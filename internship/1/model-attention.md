# 模型与注意力：从 token 到训练，再到高效注意力

阅读导航：[输入与 attention](#attention) → [GQA](#gqa) → [训练](#training) → [FlashAttention](#flash) → [线性注意力](#linear) → [手写代码](#k01)。

先读「输入与模型宽度 → 单头／多头 → Transformer 与训练 → FlashAttention → 线性与稀疏注意力」。代码在对应小节或文末附录，首次阅读可先跳过。本文统一 B 为批量、T 为序列长度、D 为模型宽度。

## 输入与模型宽度 D

token 是分词器划出的基本单位，可能是字、子词或字节片段。文本经 tokenizer 变成整数 token ID，再查 embedding 表，得到 `[B,T,D]` 的向量。经过多层 Transformer，再投影到大小为 V_vocab 的词表，输出 `[B,T,V_vocab]` 的 **logits**，即尚未归一化的类别分数。softmax 将其变成下一个 token 的概率。下节的 V 则表示注意力内容矩阵，含义不同。

embedding 表为每个 ID 保存一个可学习向量。softmax 用 `p_i=exp(z_i)/Σ_j exp(z_j)` 将一组分数变成非负、总和为 1 的概率；它保留相对偏好，指数计算的稳定处理见“训练循环与损失”。

**D 到底是什么？** D 是每个 token 的隐藏表示包含多少个数字，也称模型宽度。例如一个样本、三个 token、D=8，其表示是 `[1,3,8]`；每个 token 有 8 个特征数。它不是序列长度，也不是整个模型的参数总数（如 7B）。

一个不带 bias 的线性投影若把 D 维输入变成 M 维输出，可用 `[D,M]` 权重矩阵：`[B,T,D]×[D,M]→[B,T,M]`，参数量为 D×M；不同 token 共享这套权重，T 增长不会因此增加该投影的参数量。各头输出维数和投影的关系见下方 GQA 例子。

**参数**是要学习的权重；**激活**是一次前向的中间结果；**梯度**表示损失对参数的变化率；**优化器状态**是 Adam 等方法保留的历史统计。四者不是同一份数据。

自回归训练用前面的 token 预测后一个。文本 `[A,B,C,D]` 产生 `A→B、AB→C、ABC→D` 等目标，真实前缀已知，可一次计算多个位置。生成时新 token 尚未知，要先预测和采样，再输入模型继续下一步。

<a id="attention"></a>
## 单头注意力、MHA 与 MQA/GQA

单头中，输入 X 经不同线性投影得到 Q、K、V。Q 表示查询，K 用来匹配，V 是匹配后读取的内容：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\qquad O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

d_h 是头宽。每个 query 的分数沿 **key 位置**归一化。加性 mask M 在允许位置为 0、禁止位置为 −∞，让禁止位置的概率为 0。因果 mask 禁止看未来；padding mask 排除填充。若整行全部禁止，softmax 会出问题，应处理这种输入。

手算一行：分数 `[0,ln 2]` 对应概率 `[1/3,2/3]`；V 为 `[3,9]`，输出为 7。注意力返回加权内容，不是最大分数的位置编号。

为什么除 √d_h？若各维近似独立、均值零、方差一，点积方差随 d_h 增长；缩放缓和分数过大、softmax 过尖。这是统计动机，不是任意数据的恒等保证。

**多头 MHA** 为各头学习不同的 Q/K/V 投影，允许模型并行组合不同位置与表示子空间的信息；每头投影都可以使用完整 D 维输入，不能将它理解成只看输入的一个固定切片。投影结果组织为 H 个头，常取 d_h=D/H：`[B,T,D]→[B,H,T,d_h]`，各头独立注意，再合为 `[B,T,D]` 并作输出投影。例如 B=2、T=4、D=8、H=2，分数为 `[2,2,4,4]`。多头学习不同匹配方式，不预先指定某个头的语义。

代码统一 **True=允许关注**，框架接口须核对。[K03](#k03)、[K04](#k04)在附录，理解上文不必先读代码。

<a id="gqa"></a>
### MHA、MQA、GQA：多个查询头如何共享 K/V

**先记住一条主线：Q 决定“怎样查询”，K/V 提供“可匹配的信息和被读取的内容”。MQA 和 GQA 保留多个 Q 头，减少独立的 KV 头数量。** 它们是注意力结构，通常属于模型架构，并不只是推理时切换的一种缓存格式。

设模型宽度 D，查询头数 Hq，每头宽度 d=D/Hq；KV 头数 Hkv，假设 Hq 能被 Hkv 整除。输入 X 为 `[B,T,D]`，Q 投影为 `[B,Hq,T,d]`，K 和 V 各投影为 `[B,Hkv,T,d]`。每个查询头使用一个对应的 KV 头计算注意力，最后仍将 Hq 个输出合并为 `[B,T,D]`，再经过输出投影。

| 结构 | Hkv | Q 和 KV 的对应关系 | 特点 |
|---|---:|---|---|
| MHA：多头注意力 | Hq | 每个 Q 头有独立 K/V 头 | KV 表达更独立，缓存较大 |
| MQA：多查询注意力 | 1 | 所有 Q 头共享同一组 K/V | KV 缓存最小，共享更强可能影响质量 |
| GQA：分组查询注意力 | 1 到 Hq 之间 | 每 `g=Hq/Hkv` 个 Q 头共享一组 K/V | 在 KV 数量与表达之间取折中 |

例如 Hq=8、Hkv=2：Q0—Q3 使用 KV0，Q4—Q7 使用 KV1。Q0 和 Q1 即使共用 K/V，**查询向量不同，得到的分数、概率和输出仍可以不同**；共享也不是把所有头的输出平均。

设 Q 的长度为 Tq、K/V 的长度为 Tk，每个查询头 i 使用第 `floor(i/g)` 个 KV 头：

$$
O_i=\operatorname{softmax}\!\left(\frac{Q_iK_{\lfloor i/g\rfloor}^{\mathsf T}}{\sqrt d}+M\right)V_{\lfloor i/g\rfloor}.
$$

Q、K 和 V 的矩阵形状分别是 `[Tq,d]`、`[Tk,d]` 和 `[Tk,d]`；M 表示允许或屏蔽的位置。最终拼接 O₀…Oₕq₋₁，形成 `[Tq,D]`。自回归预填充通常 Tq=Tk；增量解码常为 Tq=1、Tk=历史长度加当前输入长度。

**节省在哪里？** 假设无 bias、d=D/Hq：Q 和输出投影各 D² 个参数；K、V 投影各 `D×Hkv×d`，所以注意力投影参数总数为 `2D²+2D Hkv d`。MHA 是 4D²；Hq=8、Hkv=2 的 GQA 是 2.5D²。减少的是 K/V 投影和缓存；Q 头数仍是 8，完整注意力的查询头数及其 QK、PV 计算不因此变成 2 头。实际计算、质量与延迟取决于具体模型、内核与硬件，不承诺相同倍数加速。

KV cache 的数量随 Hkv 线性增长：固定层数、长度、头宽和精度，8 头 MHA、2 头 GQA、1 头 MQA 的 KV 元素数之比为 **8:2:1**。[推理页](inference-service.md)有 32 头、8 KV 头的实际字节手算；页中的 2 GiB、512 MiB、64 MiB 只计算 KV，不是总显存或总推理速度。

**性能上的关键细节：** decode 时每步读取历史 K/V，共享 KV 有机会减少从显存读取的数据；但需要内核实际复用同一份 KV。教学代码若用 `repeat_interleave` 真正复制 KV，可能增加显存访问，不能因此声称实现了高性能 GQA。长 prefill 还涉及大量矩阵计算；瓶颈与 decode 可能不同。

**面试易错：**① MQA 不是只有一个 Q 头；② GQA 的 Q 头不是按组相加；③ 直接把已有 MHA 的多个 K/V 头合并会改变模型输出，通常需训练或继续适配，不能把它当无损开关；④ 因果缓存中 Tq 与 Tk 不等，mask 要按真实绝对位置构造。[MQA 原文](https://arxiv.org/abs/1911.02150)、[GQA 原文](https://arxiv.org/abs/2305.13245)

#### 手写 GQA 核心计算｜先会 shape，再考虑高性能内核

下例输入已投影好的 Q/K/V；用分组维度广播，不显式复制 KV。允许位置为 True。没有投影、RoPE、dropout、缓存分配，也不是生产 kernel。

```python
import math
import torch

def grouped_attention(q, k, v, allowed):
    # q: [B,Hq,Tq,d]; k: [B,Hkv,Tk,d]; v: [B,Hkv,Tk,dv]
    # allowed: [Tq,Tk], bool; True 表示可见
    b, hq, tq, d = q.shape
    bk, hkv, tk, dk = k.shape
    assert bk == b and dk == d and hq % hkv == 0
    assert v.shape[:3] == (b, hkv, tk)
    assert allowed.shape == (tq, tk) and allowed.dtype == torch.bool
    if not allowed.any(dim=-1).all().item():
        raise ValueError('每个 query 至少需要一个可见 key')
    group = hq // hkv
    qg = q.reshape(b, hkv, group, tq, d)
    scores = torch.matmul(qg, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(d)
    scores = scores.masked_fill(~allowed[None, None, None, :, :], float('-inf'))
    probs = torch.softmax(scores, dim=-1)  # [B,Hkv,group,Tq,Tk]
    out = torch.matmul(probs, v.unsqueeze(2))  # [B,Hkv,group,Tq,dv]
    return out.reshape(b, hq, tq, v.shape[-1])
```

例如有 3 个历史 token、本次输入 2 个 token 时，Tk=5、Tq=2，两行 mask 分别为 `[1,1,1,1,0]` 和 `[1,1,1,1,1]`。不能把不等长矩阵简单套用左上角三角 mask，否则第一行会错误屏蔽历史。此代码中的 `.item()` 是便于讲解的检查，GPU 上会引入同步；真实实现还需处理批内不同长度、数值精度及内核选择。

## Transformer block：FFN、归一化与位置

**FFN** 对每个位置分别做非线性变换，常为 `Linear(D,D_ff)→GELU→Linear(D_ff,D)`，D_ff 是中间宽度；SwiGLU 等门控形式用一个分支调节另一个分支。**残差**把分支输出加回输入，让信息和梯度有直接通路。

Linear 将最后一维从输入宽度映射到输出宽度。中间非线性防止多层仅等价于一次线性变换；SwiGLU 的 SiLU 门为 `z·sigmoid(z)`，再与另一线性分支逐元素相乘后投影。

**LayerNorm** 对一个 token 的 D 维特征求均值 μ、方差 σ²：

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

γ、β 可学习，ε 防止分母过小。RMSNorm 常用 `x/√(mean(x²)+ε)` 再缩放，不减均值。它们不同于在 batch 等维度统计的 BatchNorm。[K02](#k02)

### Pre-Norm／Post-Norm：看归一化与残差的位置

令 F 为 attention 或 FFN 分支，N 为归一化：

$$
\text{Pre: }y=x+F(N(x)),\qquad \text{Post: }y=N(x+F(x)).
$$

完整 block 对 attention、FFN 各做一次。Pre 的 Jacobian 含 `I+J_F J_N`，有显式恒等残差项；Post 为 `J_N(I+J_F)`。这是不同的梯度路径，不能据此保证 Pre 永不爆炸或完全不需 warm-up。

原始 Transformer 用 Post；本页模板默认 Pre，可用 `norm_first=False` 切换。PyTorch TransformerEncoderLayer 的 norm_first 默认 False，与本模板不同。模板的 Pre 小模型末尾有 final norm、Post 用 Identity，是本模板的选择，不是通用定义。[K05](#k05)

### 位置怎样表达

**RoPE** 按位置旋转 Q/K 的二维分量，让点积带有相对位置信息。对位置 m、一对分量和频率 θ：

$$
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}
=\begin{bmatrix}\cos(m\theta)&-\sin(m\theta)\\\sin(m\theta)&\cos(m\theta)\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}.
$$

不同维度对采用不同频率，不旋转 V。带缓存时新 Q/K 用真实位置，旧 K 不重复旋转；RoPE 不保证无限外推。[K10](#k10)

<a id="training"></a>
## 训练循环与损失

交叉熵是正确目标概率的负对数 `−log p_y`。通常把 logits 直接交给损失函数，内部使用稳定 log-softmax：

$$
\log\operatorname{softmax}(z)_i=z_i-m-\log\sum_j e^{z_j-m},\quad m=\max_j z_j.
$$

减最大值不改变概率，却能避免指数溢出。不要先 softmax 再传入要求 logits 的接口。[K01](#k01)

以下是监督模型的最小训练片段，x、target、criterion 需按任务定义：

~~~python
model.train()
optimizer.zero_grad(set_to_none=True)
prediction = model(x)
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
~~~

backward 按链式法则计算并累加梯度，step 才更新参数，zero_grad 清上次梯度。train/eval 改 Dropout、BatchNorm 等模块行为；no_grad/inference_mode 控制梯度记录，不能替代 eval。验证通常同时 eval 并关闭梯度。

例如 Dropout 训练时随机屏蔽部分激活、评估时关闭随机屏蔽；只使用 no_grad 并不会切换这种模块行为。

最简单的 SGD 更新为 `θ←θ−η∇θL`：θ 是参数，梯度指向损失局部上升的方向，学习率 η 控制向反方向走多大一步。Adam 利用梯度的一阶、二阶滑动统计调整更新；AdamW 将权重衰减与自适应梯度更新分开。这也是训练为何需要额外优化器状态的原因。

语言模型位置 t 的输出监督下一 token；框架内部已移位，就别在外部再移。attention mask 决定能读谁，loss mask 决定谁参与监督，例如 padding 不算损失。

## 训练显存与优化方法

$$
M_{train}=M_{weights}+M_{grad}+M_{optimizer}+M_{activations}+M_{temporary}.
$$

temporary 包括通信、工作区与分配开销。反向需用前向中间量，激活随 batch、长度和结构变化；朴素注意力显式分数有 T² 项。

8 bit=1 byte；FP32 每数 4 bytes，FP16/BF16 每数 2 bytes。一个明确假设：每参数低精度权重 2 bytes、梯度 2、FP32 主权重 4、Adam 两个统计 8，共 **16 bytes/参数**。7B 权重约 14 GB，模型训练状态约 112 GB，仍未计激活。实际实现可能没有主副本或采用不同精度，不能统一套 16。

GB=10⁹ bytes，GiB=2³⁰ bytes；112 GB 约 104.3 GiB。能加载权重不等于反向和优化器更新也能完成。

| 手段 | 怎样省 | 代价／边界 |
|---|---|---|
| AMP 混合精度 | 合适算子用 FP16/BF16，敏感运算保留较高精度 | 不是所有状态都降精度，FP16 常配 loss scaling |
| 梯度累积 | 小 microbatch 多次反向，最后更新 | 降低单次激活峰值，不分片参数／优化器状态 |
| 激活重计算 | 少存激活，反向重新算 | 更多计算，需处理随机性与实现一致性 |
| ZeRO/FSDP | 分片训练状态 | 更多通信和临时聚合；见[多卡与 MoE](parallel-moe.md) |
| LoRA | 冻结基础权重，学习低秩增量 | 基础权重和前向仍在，激活不会消失 |

BF16 指数范围接近 FP32，FP16 指数范围更窄；它们的范围与有效数字不同，不能认为 BF16 处处更精确。量化与混合精度的对象、误差也不同。

累积例：4 个**等大小、等权重**微批的平均 loss 各除 4 后反向，最后 step 一次。若每 DP 副本微批 b、DP 数 N、累积 K 次，有效全局 batch 为 bNK；变长文本需按有效监督 token 正确加权。

LoRA 为 `W_eff=W₀+(α/r)BA`，W₀ 冻结，A/B 可学习，新增参数约 `r(d_in+d_out)`。QLoRA 还量化冻结的基础权重，不表示全部训练运算都是低比特整数。[K09](#k09)

<a id="flash"></a>
## FlashAttention：同一注意力，减少显存读写

### 先分清算什么、存什么

单头 Q 为 `[T_q,d_h]`，K 为 `[T_k,d_h]`，V 为 `[T_k,d_v]`。分数 `S=QKᵀ/√d_h`，概率 `P=softmax(S)`，输出 `O=PV`。普通整段自注意力 T_q=T_k=T，S/P 各有 T² 个数。

例如 B=1、32 个头、T=4096、每数 2 bytes，**单张**分数矩阵就是 `1×32×4096²×2=1 GiB`，还没算概率矩阵。它们是中间张量，不是历史 KV。

先分清数据放在哪里。GPU 的设备显存（例如 HBM）容量较大；GPU 芯片上的共享内存和寄存器容量小得多，但访问更快。下面的“写入显存”指把中间张量存入 GPU 的 HBM，供后续计算读取，不是写硬盘或打印文字。

一种朴素实现可能依次执行：① 从显存读取 Q/K，计算完整分数矩阵 S，并将 S 存入显存；② 从显存读 S，计算 softmax，将完整概率矩阵 P 存入显存；③ 再从显存读 P 和 V，计算并存储输出 O。因此 S、P 不仅各占 T² 规模的空间，还会产生对应的显存读写。实际实现可以融合操作，普通 attention 不一定逐步保存这两个完整矩阵。

**FlashAttention 将 Q/K/V 分块从显存读入片上存储，在片上处理局部分数，用少量累计量逐块合并输出。** 它避免将完整 S/P 存入显存后再读取；最终输出 O 仍存入显存，训练也可能保存必要的逐行统计，以便反向计算。它优化数据搬运和内核组织，并未删掉合法注意力关系。

### 在线 softmax：为什么分块后还算得对

固定一个 query，记第 j 个 key 的分数为 s_j，对应内容向量为 v_j。稳定的完整输出为：

$$
m=\max_j s_j,\qquad
\ell=\sum_j e^{s_j-m},\qquad
u=\sum_j e^{s_j-m}v_j,\qquad o=u/\ell.
$$

m 是最大分数，ℓ 是指数和，u 是未归一化的内容加权和。u 是向量，不是概率。减 m 只为数值稳定，分子分母同时乘相同因子，结果不变。

已有旧块的 m、ℓ、u，新块 J 到来，先合并最大值，再把旧累计量换到同一基准：

$$
\begin{aligned}
m'&=\max(m,\max_{j\in J}s_j),&\alpha&=e^{m-m'},\\
\ell'&=\alpha\ell+\sum_{j\in J}e^{s_j-m'},\\
u'&=\alpha u+\sum_{j\in J}e^{s_j-m'}v_j.
\end{aligned}
$$

初始化 m=−∞、ℓ=0、u=0。每块只需更新这三个量，最终输出 u/ℓ。为何旧量乘 α？因为 `e^(s−m')=e^(s−m)×e^(m−m')`，所以旧指数和、旧加权和都需同样缩放。

带 mask 时这里只合并可见 key；全被屏蔽的块可跳过，整行没有可见 key 要单独处理，不能直接计算 −∞−(−∞)。附录教学代码对这种情况显式检查。

手算：分数 `[0,ln2,ln4]`，V 简化成 `[10,20,30]`，完整答案为 `(10+40+120)/(1+2+4)=170/7≈24.286`。

| 处理到哪 | m | ℓ | u | u/ℓ |
|---|---|---:|---:|---:|
| 第一块：前两项 | ln2 | 1.5 | 25 | 16.667 |
| 第二块：再看最后一项 | ln4 | 1.5×0.5+1=1.75 | 25×0.5+30=42.5 | 24.286 |

新最大值升高，旧量乘 `exp(ln2−ln4)=0.5`。若两块各自 softmax 再平均，会得 23.333，错在没有保留两块在全局中的概率质量。

### 哪些省了，哪些仍要做

标准 dense attention 的算术量仍为 O(T²d_h)；少的是完整二次中间矩阵的保存和相关显存访问。Q/K/V、输出和 KV 内容仍存在。训练反向可保存逐行统计等，再按块重算局部量，用重算换存储。

“exact”指实数数学上同一个注意力目标，浮点次序不同可有数值误差，不保证逐位一致。实际速度依硬件、shape、dtype 和内核；FlashAttention-2 进一步优化工作分配与非矩阵乘开销，不必背统一速度倍数。

PyTorch `scaled_dot_product_attention` 是接口，实际后端需确认。等长因果输入可用 `is_causal=True`；有历史缓存、Q/K 长度不等时应核对绝对位置与 mask 对齐，不能直接照搬。推理显式使用 `dropout_p=0`。

依据：[FlashAttention 原论文](https://arxiv.org/pdf/2205.14135)、[在线 softmax](https://arxiv.org/abs/1805.02867)、[FA2](https://arxiv.org/abs/2307.08691)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。附录的单 query Python 分块循环只验证累计关系，不实现 GPU 高效内核。

### 单 query 在线 softmax

q=[d]、k=[N,d]、v=[N,dv]，FP32/FP64、同设备、实际分数有限；allowed 为 [N] 布尔向量。Python 循环与同步检查不体现正式内核性能。

~~~python
import math
import torch

def online_attention_one_query(q, k, v, block_size=64, allowed=None):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if q.ndim != 1 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("需要 q=[d], k=[N,d], v=[N,dv]")
    if q.numel() == 0 or k.size(0) == 0:
        raise ValueError("输入不能为空")
    if k.size(1) != q.numel() or k.size(0) != v.size(0):
        raise ValueError("shape 不匹配")
    if allowed is not None:
        if allowed.shape != (k.size(0),) or allowed.dtype != torch.bool:
            raise ValueError("allowed 必须是 [N] 布尔向量")

    maximum = q.new_tensor(float("-inf"))
    denominator = q.new_zeros(())
    numerator = q.new_zeros(v.size(1))

    for start in range(0, k.size(0), block_size):
        stop = min(start + block_size, k.size(0))
        scores = (k[start:stop] @ q) / math.sqrt(q.numel())
        if allowed is not None:
            block_allowed = allowed[start:stop]
            if not block_allowed.any().item():
                continue
            scores = scores.masked_fill(~block_allowed, float("-inf"))

        new_maximum = torch.maximum(maximum, scores.max())
        scale = torch.exp(maximum - new_maximum)
        weights = torch.exp(scores - new_maximum)
        denominator = scale * denominator + weights.sum()
        numerator = scale * numerator + weights @ v[start:stop]
        maximum = new_maximum

    if denominator.item() == 0:
        raise ValueError("没有可关注的 key")
    return numerator / denominator
~~~

## 窗口、稀疏、MLA 与多卡：各自改变什么

| 方法 | 主要针对的问题 | 怎样改变 | 代价／边界 |
|---|---|---|---|
| 滑动窗口 attention | 全局配对数随 T² 增长 | 每个位置只看局部 W 个位置 | 单层无法直接看窗口外全部历史；常配合全局 token 等机制 |
| 稀疏 attention | 不必计算所有位置对 | 预定义或构造局部／全局等稀疏连接 | 减少了可直接访问的位置，效果与稀疏模式有关 |
| 线性 attention | 逐位置访问全部历史的成本 | 用可分解特征和累计状态代替显式逐对 softmax | 一般改变或近似原注意力，固定状态可能受容量限制 |
| MLA，多头潜变量注意力 | KV cache 太大 | 把 KV 表示压到共享低维潜变量，推理时利用等价代数计算 | 结构和位置编码设计更复杂；不是仅把 GQA 头数改小 |
| Ring Attention | 单卡承载不了长序列及其计算 | 序列分卡，本地 Q 与环上传来的 KV 块计算 | 有跨卡通信、负载和同步开销；不自动降低全局 dense 算术量 |

滑窗 attention 若每个位置最多看 W 个 token，主要匹配/加权量约为 O(TWd)，但不同实现、边界和全局连接会改变开销；不能把它说成保留原先所有可见关系。[Longformer](https://arxiv.org/abs/2004.05150)、[DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434)、[Ring Attention](https://arxiv.org/abs/2310.01889)

MLA 的一个理解方式：把每个 token 原本多个头所需的 KV 信息先编码为较低维的潜变量，缓存该潜变量及具体架构需要的位置相关部分；使用时可通过上投影及矩阵结合等方式计算各头。实际 DeepSeek-V2 还对位置编码作专门设计，不能简单认定“所有 K/V 被完全丢弃”或“任何 RoPE 都能直接吸收”；缓存到底多少需要按模型实际维度和实现算。

**较新的两个例子，先认识要解决什么。** Gated DeltaNet 通过门控与更新规则调节线性状态的保留和写入，并研究与其他层组合的混合架构；下文“线性注意力”的简单累加只用来理解基础，不能当作它的完整公式。[论文](https://arxiv.org/abs/2412.06464)

DeepSeek-V3.2-Exp 的 DSA 属于稀疏注意力方向：用轻量索引器为历史位置评分，挑选部分位置进行后续更昂贵的注意力计算；它建立在特定模型设计上，也要计索引和选择的开销，不能仅因只选 top-k 就断言整个系统对序列长度严格线性或一定无质量损失。[vLLM 技术说明](https://vllm-project.github.io/2025/09/29/deepseek-v3-2.html)

<a id="linear"></a>
## 线性注意力：从核函数到因果状态

### 先看瓶颈和替换的东西

对于 T 个位置，标准 attention 的每个 query 都与多个 key 点积；全局 dense 情况涉及约 T² 个位置对。它的 softmax 权重含 `exp(q·k/√d)`，一般不能只靠普通矩阵乘法结合律把 softmax 括号直接去掉：

```text
softmax(QKᵀ)V ≠ Q(KᵀV)   # 通常不相等
```

一种线性 attention **改变相似度核**：用特征映射 `φ` 将 q、k 映射为 r 维向量，并用 `φ(q)ᵀφ(k)` 作为相似度，保留归一化。以非负 φ 为例，避免权重出现负值；下面的等式对这种选定的核准确成立，但不等同于原来的 softmax 核。

### 先求公共汇总，再回答 query

对于一段可见历史，定义：

$$
S=\sum_j\phi(k_j)v_j^\mathsf{T}\in\mathbb R^{r\times d_v},\qquad
z=\sum_j\phi(k_j)\in\mathbb R^r.
$$

S 把每个 key 的特征和对应 V 汇总，z 为归一化所需的 key 特征之和。r 是特征维度，d_v 是 value 宽度；S 是矩阵而不是注意力分数矩阵。对于任意 query：

$$
o_i=\frac{\phi(q_i)^\mathsf{T}S}{\phi(q_i)^\mathsf{T}z}.
$$

为什么对？把 S、z 的求和展开，分子就是 `Σ_j [φ(qᵢ)ᵀφ(kⱼ)] vⱼ`，分母是 `Σ_j φ(qᵢ)ᵀφ(kⱼ)`。它先累计所有 key/value 的共同信息，再用 query 读取，因此不必存储全部 query–key 配对矩阵。

### 自己算一遍

设 `φ(q)=[2,1]`，两个 key 的映射分别为 `[1,0]`、`[0,1]`；对应标量 V 为 2、6。这些都是示意数值，省略映射本身：

```text
S = [1,0]×2 + [0,1]×6 = [2,6]
z = [1,0] + [0,1] = [1,1]
输出 = ([2,1]·[2,6]) / ([2,1]·[1,1]) = 10/3
```

直接逐项算也一样：两个相似度分别为 2 和 1，权重为 2/3、1/3，输出 `2×2/3+6×1/3=10/3`。这里计算的是这个线性核的归一化输出，不能当作原始 softmax attention 的数值。

### 因果语言模型：可递推的状态

第 t 个 token 只能读取自己及过去。每来一个 token 更新：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\mathsf{T},\qquad
z_t=z_{t-1}+\phi(k_t),\qquad
o_t=\frac{\phi(q_t)^\mathsf{T}S_t}{\phi(q_t)^\mathsf{T}z_t}.
$$

与普通 KV cache 保存每个历史 token 的 K/V 不同，这个示例只保留每层每头固定形状的 S、z；固定 r、d_v 时，状态大小不随历史长度增长。它把历史压进汇总状态，不能直接逐个取回任意旧 token，也不自动支持任意 attention mask；有限精度和长期状态容量需要验证。不同线性注意力可能有衰减、门控或不同状态，不能把这个简单公式套到全部方法。

如果 r、d_v 固定，构建与查询这些汇总对序列长度的主要复杂度可写为 O(T·r·d_v)，另计特征映射、投影等成本。**线性是指相对于 T，不是相对于所有维度。** r 大或序列短时，常数与内核效率可能使其不如优化过的 dense attention 快。

有些方法如 Linear Transformer 选用不同于 softmax 的核；Performer 用随机特征近似 softmax 核。二者都需要解释选择的特征及误差，不能把所有线性 attention 都说成对原 softmax 精确无误。[Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html)、[Performer](https://arxiv.org/abs/2009.14794)

### 手写：非负核的因果递推

教学代码输入 Q/K `[T,d]`、V `[T,dv]`，三者应在同一设备且使用兼容的浮点 dtype；使用 `φ(x)=ELU(x)+1`。它实现上面的核，不实现标准 softmax attention，也没有 padding、批量、GPU 优化或混合精度。

```python
import torch
import torch.nn.functional as F

def causal_linear_attention(q, k, v):
    # 单样本单头，q/k: [T,d], v: [T,dv]
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape == k.shape and q.shape[0] == v.shape[0]
    if q.shape[0] == 0:
        return v.new_empty((0, v.shape[1]))
    qf, kf = F.elu(q) + 1, F.elu(k) + 1
    state = q.new_zeros((q.shape[1], v.shape[1]))
    normalizer = q.new_zeros(q.shape[1])
    outputs = []
    for t in range(q.shape[0]):
        state = state + torch.outer(kf[t], v[t])
        normalizer = normalizer + kf[t]
        numerator = qf[t] @ state
        denominator = qf[t] @ normalizer
        outputs.append(numerator / denominator.clamp_min(1e-12))
    return torch.stack(outputs)
```

`clamp_min` 是防止有限精度下除零的教学保护；不是线性 attention 的效率来源。只缓存 state/normalizer 的增量实现可跨 token 更新；训练时仍有自动求导与中间激活成本，不要把推理状态常数误写成整个训练显存恒定。

## 代表性方法出现的时间与动机

下面是帮助定位的代表性时间线，年份为论文最初公开年份，不代表前者已经被淘汰：

| 时间 | 代表工作 | 着眼的问题 |
|---|---|---|
| 2017 | Transformer／MHA | 多个投影头组合不同位置和表示信息 |
| 2019 | MQA | 自回归推理时多头 KV 的读取与缓存成本 |
| 2020 | Linear Transformer、Longformer | 长序列全局注意力的二次开销；分别从可分解核、稀疏连接出发 |
| 2022 | FlashAttention | 相同 dense attention 的设备显存读写与中间存储 |
| 2023 | GQA、Ring Attention | 分别关注 KV 共享的质量折中、长序列多设备分摊 |
| 2024 | DeepSeek-V2 的 MLA、Gated DeltaNet | 分别关注 KV 潜变量压缩、带门控的线性状态更新及混合架构 |
| 2025 | DeepSeek-V3.2-Exp 的 DSA | 用轻量索引器选择部分历史位置，再做稀疏注意力 |

## 合上资料回答

| 问题 | 核对要点 |
|---|---|
| softmax 在哪一维？ | 每个 query 沿 key 位置归一化 |
| 8 个 Q 头、2 个 KV 头怎样分组？ | 每 4 个 Q 头共享 1 个 KV 头；不同 Q 输出可以不同 |
| GQA 减少 KV 头后 QK 算术量按头数等比例减少吗？ | Q 头数不变，注意力计算仍由各 Q 头执行；KV 投影、缓存减少 |
| Pre/Post 差什么？ | `x+F(N(x))` 与 `N(x+F(x))` |
| train/eval 与 no_grad？ | 模块行为与梯度记录分别控制 |
| backward 后参数变了吗？ | 通常只累计梯度，step 才更新 |
| 累积能分片模型状态吗？ | 不能；它降低单个微批的激活需求 |
| 7B×16 bytes？ | 112 GB 模型状态，不含激活和临时量 |

依据：[Transformer](https://arxiv.org/abs/1706.03762)、[归一化分析](https://arxiv.org/abs/2002.04745)、[RoPE](https://arxiv.org/abs/2104.09864)、[PyTorch 训练](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、[CUDA 内存](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[LoRA](https://arxiv.org/abs/2106.09685)。基础机制可信度高，内存／效果按实际实现验证。

## 附录｜完整手写代码，第一遍可以先跳过

按 K01→K02→K03→K04→K05→K06 运行可满足依赖；K09/K10 备用。张量默认 FP32，布尔 True=允许。缓存类是固定 batch、无 padding 的单层教学实现，完整模型还需逐层缓存和位置管理。

<a id="k01"></a>

### K01｜稳定 softmax／交叉熵

减最大值，再作 log-softmax；logits=[N,C]、targets=[N]。

~~~python
import torch
from torch import nn
import math

def stable_softmax(x, dim=-1):
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy_from_logits(logits, targets):
    # logits: [N, C]，targets: [N]，类别编号，不含 ignore_index
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_probs = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    correct_log_probs = log_probs.gather(1, targets[:, None]).squeeze(1)
    return -correct_log_probs.mean()
~~~

<a id="k02"></a>

### K02｜LayerNorm／RMSNorm

输入最后一维归一化，解释为何 RMSNorm 不减均值。

~~~python
def layer_norm_last(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(var + eps)
    return normalized * weight + bias

def rms_norm_last(x, weight, eps=1e-6):
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight
~~~

<a id="k03"></a>

### K03｜单头注意力

Q/K/V 为 [B,H,T,d]；布尔 True 表示允许。全屏蔽行报错；.item() 检查会同步，教学使用。

~~~python
def scaled_attention(q, k, v, allowed=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if allowed is not None:
        # allowed 可广播到 [B,H,Tq,Tk]，True 表示允许
        allowed = torch.broadcast_to(allowed, scores.shape)
        if not allowed.any(dim=-1).all().item():
            raise ValueError("某个 query 没有可关注的 key")
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
~~~

<a id="k04"></a>

### K04｜多头因果注意力

输入 [B,T,D]，投影、分头、mask、合头、输出投影；依赖 K03。

~~~python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model 必须为 n_heads 的正整数倍")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _project(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split_heads(z):
            return z.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return split_heads(q), split_heads(k), split_heads(v)

    def _merge(self, y):
        B, _, T, _ = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(y)

    def forward(self, x):
        _, T, _ = x.shape
        q, k, v = self._project(x)
        allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        return self._merge(scaled_attention(q, k, v, allowed))
~~~


~~~python
attn = CausalSelfAttention(d_model=8, n_heads=2)
x = torch.randn(2, 4, 8)
y = attn(x)
assert y.shape == (2, 4, 8)
~~~

<a id="k05"></a>

### K05｜Pre/Post block 与小模型

norm_first=True 为 Pre；TinyDecoderLM 使用可学习的绝对位置 embedding。RoPE 在 K10 单列，没有自动接入这个小模型，两种位置机制不应混同。

~~~python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        if self.norm_first:  # Pre-Norm：LN 在残差分支内部
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        else:               # Post-Norm：LN 包住相加的结果
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
        return x
~~~


~~~python
pre = DecoderBlock(d_model=8, n_heads=2, norm_first=True)
post = DecoderBlock(d_model=8, n_heads=2, norm_first=False)
x_norm_demo = torch.randn(2, 4, 8)
assert pre(x_norm_demo).shape == post(x_norm_demo).shape == (2, 4, 8)
~~~


~~~python
class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len,
                 norm_first=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, norm_first=norm_first)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.shape
        if not 0 < T <= self.position_embedding.num_embeddings:
            raise ValueError("序列长度超出本模型范围")
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))
~~~


~~~python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gated = torch.nn.functional.silu(self.gate(x)) * self.up(x)
        return self.down(gated)
~~~

<a id="k06"></a>

### K06｜单层 KV Cache

继承 K04，复用 K03。旧长度 L、本轮 S 时，新 query 位置 L…L+S−1；每步 cat 是语义示范。

~~~python
class CachedSelfAttention(CausalSelfAttention):
    def forward(self, x, cache=None):
        q, k_new, v_new = self._project(x)
        if cache is None:
            past_len = 0
            k, v = k_new, v_new
        else:
            past_k, past_v = cache
            past_len = past_k.size(-2)
            k = torch.cat([past_k, k_new], dim=-2)
            v = torch.cat([past_v, v_new], dim=-2)

        query_positions = past_len + torch.arange(x.size(1), device=x.device)
        key_positions = torch.arange(k.size(-2), device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        y = self._merge(scaled_attention(q, k, v, allowed))
        return y, (k, v)
~~~

<a id="k09"></a>

### K09｜LoRA

基础权重冻结；B 零初始化、A 非零初始化。不是 QLoRA 或量化实现。

~~~python
class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base  # 假设传入普通、尚未量化的 nn.Linear
        self.base.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        self.A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.B.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))
~~~

<a id="k10"></a>

### K10｜RoPE

相邻偶奇维旋转 Q/K，positions 为实际位置；输入 [B,H,T,d]。

~~~python
def apply_rope(x, positions, base=10000.0):
    d = x.size(-1)
    if d <= 0 or d % 2 != 0 or base <= 0:
        raise ValueError("需要正偶数头宽与正 base")
    dimensions = torch.arange(0, d, 2, device=x.device, dtype=torch.float32)
    inv_freq = base ** (-dimensions / d)
    positions = positions.to(device=x.device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack(
        [even * cos - odd * sin, even * sin + odd * cos], dim=-1
    )
    return rotated.flatten(-2).to(dtype=x.dtype)
~~~

**缓存 mask 自检**：已有 3 个 token，再来 2 个，允许矩阵应为 `[1,1,1,1,0]` 和 `[1,1,1,1,1]`。先验证整段前向与 prefill＋decode 近似一致，再谈优化。模板不代表已完成模型训练或部署。

