# C｜深度学习与 LLM 基础

更新：2026-10-20，新增基础讲解。P0：C01、02、03、06、08、09、12、17。先理解一次预测与参数更新，再学 Transformer 和问答。

**10/24 手写补充**：本页 C06、C08、C11—C14 已就近插入代码；[模型与系统代码总表](05-model-system-coding.md)另含完整 MoE 与双向链表 LRU。先背 attention／MHA、MoE、LRU，算法题本轮不扩充。

## 知识地图

| 模块 | 基础内容 | 需要输出 |
|---|---|---|
| 训练 | 链式法则、计算图、反传、初始化、优化器、正则、学习率 | 写小训练循环，解释梯度来源 |
| PyTorch | shape／broadcast、dtype／device、autograd、train／eval、冻结、保存加载 | 能定位维度、模式和梯度错误 |
| 网络 | 激活、残差、BN／LN／RMSNorm、CNN、深度展开 | 将已有 ICCV 网络讲到实现与消融 |
| Transformer | QKV、多头、mask、位置编码、FFN、残差与归一化 | 手写简化 attention 并标维度 |
| LLM 训练 | tokenization、自回归目标、SFT、LoRA／QLoRA、mask | 区分理解目标与已有训练实践 |
| LLM 推理 | prefill／decode、KV cache、MHA／GQA／MQA、采样 | 解释成本、时延和缓存显存 |
| 扩展 | AMP、梯度累积／检查点、DDP／FSDP | 有岗位反馈再深入，不虚构多卡经历 |

## 基础知识：从一个神经元到语言模型

前置知识只需要基础 A-6／A-7 的向量与梯度、基础 B-1—B-5 的样本与训练。遇到忘记的词可以回查对应小节，不要求先完成整份数学题。

### 基础 C-1｜神经网络、神经元和“深度”

一个简化的**神经元**先把输入按权重相加，再经过一个函数：$h=\phi(w^\top x+b)$。w 是权重，b 是偏置，$\phi$ 是激活函数。这里的“神经元”是计算单元的名称，不需要用生物神经细节理解。

把多个单元组织为一层，再把多层连接，就得到**神经网络**。**深度学习**使用多层可学习变换来建立输入到输出的关系。中间各层计算出的数叫**激活／隐藏表示**，它们可以逐步表达有用特征。

**前向计算 forward**是从输入逐层算到输出；**反向传播 backward**是根据损失求各参数的梯度。**推理 inference**通常是使用已训练好的参数做前向预测。

### 基础 C-2｜张量、shape、batch：代码里的数据长什么样

**张量 tensor**在这里可以先理解为有形状的多维数值数组。单个数、向量、矩阵都是其简单情况。shape 描述各轴大小；维度顺序必须与代码约定一致。

| 数据 | 常见 shape | 各轴含义 |
|---|---|---|
| 8 个样本，每个 10 个特征 | `[8,10]` | 样本数、特征数 |
| 4 张 RGB 图片 | `[4,3,H,W]` | 常见 PyTorch 约定：样本数、通道、高、宽 |
| 2 段文字，每段 5 个 token 的表示 | `[2,5,d]` | 样本数、序列长度、每个 token 的表示宽度 |

**batch**是一批一起计算的样本，B 常表示批大小。**广播 broadcasting**让某些不同 shape 的数组按规则扩展后运算；它不保证扩展方式就是你想要的。比如预测为 `[B,1]`，标签为 `[B]`，直接相减可能得到 `[B,B]`，因此必须检查 shape。

这里“有几条轴”与线性代数里矩阵的秩不是同一概念。**dtype**是元素的数值类型，**device**是数据所在的计算设备。

### 基础 C-3｜完整算一次前向、反向和更新

用一个单样本练习：输入 x=2，目标 y=6，模型 $\hat y=wx+b$，起始 w=1、b=0。为了方便求导，这次定义损失 $L=\tfrac12(\hat y-y)^2$。

1. **前向**：预测 $\hat y=1\times2+0=2$。
2. **计算损失**：$L=(2-6)^2/2=8$。
3. **对预测求导**：$\partial L/\partial\hat y=\hat y-y=-4$。
4. **对参数求导**：$\partial\hat y/\partial w=x=2$，$\partial\hat y/\partial b=1$。
5. **链式法则**：$\partial L/\partial w=(-4)\times2=-8$；$\partial L/\partial b=-4$。
6. **学习率为 0.1，同时更新**：$w=1-0.1(-8)=1.8$，$b=0-0.1(-4)=0.4$。
7. **重新前向**：预测为 $1.8\times2+0.4=4$，损失为 $(4-6)^2/2=2$，比原来小。

**链式法则**就是把“输出怎样随中间量变化”和“中间量怎样随参数变化”连接起来。反向传播将这个规则扩展到整个网络，并在共享路径处累加贡献。优化器再使用这些梯度更新参数。

读到这里，只需能解释为什么损失梯度是 -8，以及为什么更新时减去一个负数。后面的复杂网络仍遵循这个过程。

### 基础 C-4｜一次训练到底循环什么

**iteration／迭代**通常指处理一批数据的一轮过程；**step**常指一次优化器更新；**epoch**表示遍历一遍训练集。若 1000 个样本、每批 100 个、每批更新一次，一轮 epoch 有 10 次更新。使用梯度累积后，二者的对应会变化。

```text
对每轮 epoch：
    对每个训练 batch：
        清理上一次的梯度
        前向得到预测
        计算损失
        反向求参数梯度
        优化器更新参数
    在验证数据上计算指标，不做参数更新
```

在 PyTorch 中，这对应常见的 `zero_grad`、模型调用、loss、`backward`、`step`。这是各步骤的分工，先不要求死记接口参数。

**训练模式 train**与**评价模式 eval**影响 Dropout、BatchNorm 等模块的行为；**no_grad**控制是否记录梯度计算。验证一般切换 eval 并关闭梯度记录。`eval()` 本身不等于冻结所有参数或关闭求导。

### 基础 C-5｜为什么有激活函数，为什么用残差

如果每层都只做线性变换，多层连起来仍等价于一个线性变换。**非线性激活函数**让网络能够表达更复杂的关系。

**ReLU**是 $\max(0,z)$：输入 -2 得到 0，输入 3 得到 3。**GELU**是另一种平滑的激活；**SwiGLU**采用门控结构，让一条分支调节另一条分支传递的信息。这一遍先理解它们改变的是计算形式。

**残差连接 residual**把输入与经过若干层的变化相加，写作 $y=x+F(x)$。网络可以学习“在原表示上改多少”，也为梯度传播提供直接路径；不意味着任意深的网络都不会训练出问题。

### 基础 C-6｜优化器、学习率与梯度问题

**SGD**按当前梯度更新参数；**Momentum**累积过去的方向，像带有惯性的调整；**Adam**使用梯度的一阶／二阶历史统计，按参数坐标调整步长；**AdamW**将权重衰减和自适应梯度更新分开处理。

学习率过大，可能越过低损失区域并震荡；过小，可能变化很慢。**学习率调度**是在训练中按规则改变学习率。具体选择要用验证而非单凭优化器名字。

**梯度消失**指传回某些位置的梯度很小，参数难以有效调整；**梯度爆炸**指梯度过大，更新不稳定。**梯度裁剪**限制梯度大小；**NaN**表示不是合法数值，**Inf**表示无穷大，出现时要定位除零、溢出等来源。

### 基础 C-7｜归一化与 Dropout 在解决什么

网络内部数值的尺度可能影响训练。**归一化 normalization**按指定轴对数值做变换，具体不等于一律缩到 0—1。

- **BatchNorm，BN**：常见图像用法按通道、跨 batch 和空间位置估计统计量，训练和评价的统计使用方式有所不同。
- **LayerNorm，LN**：常对一个样本／token 的特征维处理，不依赖同批其他样本的统计。
- **RMSNorm**：按均方根缩放，典型形式不减去均值。

**Dropout**训练时随机屏蔽部分激活，并按约定缩放，作为一种正则；常规评价时不再随机丢弃。它作用于计算过程，不是把训练样本的正确标签随机改错。模块的训练模式以及具体函数的参数都要核对。

### 基础 C-8｜token、词表与 embedding

语言模型先将文本切成 **token**，再转成编号。token 可能对应词、子词、字符或其他片段，不固定等于一个汉字或英文单词。**tokenizer**负责这种编码，**词表 vocabulary**保存可用 token 与编号的对应。

教学上假设“我／喜欢／科研”得到编号 `[7,12,9]`，这只是自拟例子，不是真实分词器的结果。编号本身不是语义距离：编号 12 不意味着比编号 7 “多理解五分”。

**embedding，嵌入向量**将每个编号映射到一组可学习数字。词表有 V 个 token、表示宽度为 d 时，可以用 V×d 的表保存。长度为 T 的文字由此变成 T×d 的表示；带 batch 后常为 `[B,T,d]`。

**隐藏状态 hidden state**是网络计算中每个位置的表示，会随上下文和层数变化，不只是一开始查到的向量。

### 基础 C-9｜注意力先理解“怎样加权取信息”

假设当前位置需要从两个位置取信息，它们的内容数值 V 分别为 1 和 5，相关性分数分别为 0 和 $\log3$。

**softmax**先将分数取指数，再除以指数总和：$e^0=1$、$e^{\log3}=3$，因此权重是 1/4 和 3/4。加权结果为 $1\times1/4+5\times3/4=4$。

注意力把这种过程推广到向量：

- **Query，Q**：当前要找什么，用于发出查询。
- **Key，K**：各位置用于与查询匹配的表示。
- **Value，V**：各位置被加权汇总的内容表示。

Q 与各 K 计算匹配分数，缩放后经 softmax 变成权重，再对 V 加权。Q/K/V 通常通过可学习投影得到；在自注意力里，它们可以来自同一段输入的不同投影。

这些权重描述这一计算步骤的混合方式，不能单凭权重就完整解释模型为何作出最终判断。完整公式和维度见 C08，先确认能算出这个结果 4。

### 基础 C-10｜多头、位置、mask 与 Transformer

**多头注意力**使用多组可学习投影，分别取信息，再将结果拼接、变换。这里的注意力头，与 4a 里提供动作建议的策略头，定义和作用要分别说明。

**位置编码**给模型提供位置关系；只汇总内容不足以自然区分词序。**RoPE**是一种在 Q/K 上按位置旋转的表示方法，具体数学放在第二遍。

**mask**表示哪些位置允许读取。**causal mask，因果遮罩**禁止当前位置看到未来 token，服务于逐步生成；**padding mask**排除为凑齐 batch 长度而填充的位置。

一个常见 Transformer 模块包含注意力、**FFN（前馈网络）**、归一化和残差连接。FFN 对各位置的表示做进一步非线性变换；注意力负责跨位置取信息。多个模块叠加，得到更深的表示变换。具体顺序和结构有不同变体。

### 基础 C-11｜语言模型怎样预测下一项，又怎样生成一句话

常见**自回归语言模型**根据已有 token 预测下一个 token 的分布。输出层先为词表里各 token 计算分数，叫 **logits**，再用 softmax 得到概率分布。

训练时已知完整示例，可以让“我”预测“喜欢”，“我喜欢”预测“科研”，并用因果 mask 同时计算多个位置的损失。这种把正确历史作为输入的方式常叫 **teacher forcing**。

生成时则把刚得到的新 token 加入已有序列，继续预测下一个，直到结束标记或预算耗尽。可以每次选最高分，也可以按一定规则采样。**temperature**调节概率分布的尖锐程度，**top-k／top-p**限制采样候选；这些参数不会自动保证答案正确。

**loss mask**决定哪些位置计入训练损失，与“注意力能读取哪些位置”的 mask 是不同控制。

### 基础 C-12｜prefill、decode 与 KV cache

**prefill**处理已有上下文；**decode**是在此之后逐步生成新 token。因果生成中新位置会读取历史表示，过去 token 的 K/V 可以保存复用，这些存储叫 **KV cache**。

这样可以避免每步重复计算历史 K/V，但新 token 仍需与历史缓存交互，缓存越长通常也占更多内存。它不是保存“所有问题的答案”，也不等于长期记忆系统。

**MHA／MQA／GQA**在 K/V 是否由多个 query 头共享方面不同，影响缓存和计算权衡。先理解“缓存哪些中间量，省掉哪些重复计算”，显存公式留到 C12。

### 基础 C-13｜预训练、SFT、冻结与 LoRA

**预训练 pretraining**常在大量文本上学习 next-token 预测。**SFT，监督微调**用示范输入／输出或轨迹进一步训练，帮助模型形成目标任务行为。

**冻结 freeze**指不更新指定参数。**全量微调**更新基础模型的相应全部参数；**LoRA**保留基础权重，额外学习一个用小矩阵表示的低秩更新，减少可训练参数和相关存储。

**量化 quantization**用较少位数表示数值，节省存储但带来近似误差；**QLoRA**在量化的冻结基础权重上训练适配器。这里先认识更新对象，实际训练设置和低秩公式在 C14。

### 基础 C-14｜代码里其余常见词与本节自检

| 词 | 含义 |
|---|---|
| CPU／GPU、device | 计算设备及数据所在的位置；相关张量需位于兼容设备上 |
| FP32／FP16／BF16、dtype | 不同数值表示，精度、范围和存储开销有差异 |
| OOM | 内存／显存不足，需区分权重、梯度、激活和缓存等开销 |
| checkpoint | 保存的训练／模型状态；保存哪些内容取决于用途 |
| 梯度累积 | 多个微批累积梯度后再更新；要正确处理归一化 |
| 激活检查点 | 保存较少中间结果，在反向时重算，以计算换存储 |
| DDP | 常见数据并行方式：各进程处理不同数据并同步梯度 |
| FSDP | 对参数、梯度等作分片管理的一类方案 |

自检只做四件事：复算 C-3 的更新；说出 train／eval 与 no_grad 的分工；解释 token 编号为何不是语义向量；复算注意力例子的 1/4、3/4 和结果 4。完成后再读 C01、C02、C08，其他公式逐步补。

这里解释的是常见机制，具体库行为以对应接口和实际版本为准；本文件原有问答与一手资料提供进一步核对入口。

## 20 道典型问答

### C01 [P0] 反向传播和梯度下降是一回事吗？

反向传播用链式法则高效求损失对中间量／参数的梯度；优化器利用梯度更新参数。前者是求导过程，后者是优化过程。计算图分叉时各条路径梯度相加，张量操作的反向还要考虑广播维度上的求和。

追问：`backward()` 通常把梯度累积到参数的 `.grad`，并不自动调用优化器或清零；错误的 detach、转成非图内数据或在 no_grad 中前向会断开梯度路径。

### C02 [P0] `eval()`、`no_grad()`、参数冻结分别改变什么？

`model.eval()` 改变模块训练模式，例如 Dropout、常规 BatchNorm 的行为；不自动关闭 autograd。`torch.no_grad()` 控制作用域内梯度记录，不会切换模型模式。`requires_grad_(False)` 冻结参数梯度，通常还应核对优化器参数与状态。

常规推理同时使用 eval 和 no_grad；`inference_mode` 还施加额外限制，不能无条件替代所有 no_grad 场景。冻结权重也不表示必须切断对输入的梯度。[PyTorch 官方说明](https://docs.pytorch.org/docs/stable/notes/autograd)

### C03 [P0] SGD、Momentum、Adam、AdamW 有什么区别？

SGD 沿负梯度更新；Momentum 累积方向，减轻某些方向的震荡。Adam 维护梯度一阶／二阶矩的指数平均及偏差修正，按坐标调整有效步长，常见形式为 $\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$。

AdamW 把权重衰减与自适应梯度更新解耦；在 Adam 中把 L2 加进损失，通常不等价于同样系数的解耦衰减。优化器没有对所有任务统一更好的选择，比较要考虑调参预算和泛化。[AdamW 原论文](https://arxiv.org/abs/1711.05101)

### C04 [P1] 梯度消失／爆炸从哪里来，怎样排查？

深层链式乘积、激活饱和、初始化、长序列和数值尺度都可能影响梯度。检查各层激活／梯度范数和非有限值；结合残差、合适初始化、归一化、梯度裁剪和学习率处理。

裁剪限制梯度范数，不修复错误标签或损失实现。先定位开始异常的层／步骤，不要先随意换所有组件。[数值稳定与初始化](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)

### C05 [P1] 为什么需要非线性？ReLU、GELU、SwiGLU 怎样区别？

多个纯线性层仍可合成一个线性映射。ReLU 截断负值，存在部分神经元长期不激活的问题；GELU 平滑地依赖输入调节通过程度；SwiGLU 用带 SiLU 激活的分支与另一分支逐元素门控，再投影。

比较还要看隐藏维度、参数量和 FLOPs，不把单一激活视为性能保证。先画一个 FFN 的输入、扩展、非线性和输出维度。

### C06 [P0] BatchNorm、LayerNorm、RMSNorm 在哪些维度归一化？

常见图像 BN 对每个通道跨 batch 和空间维统计，常规设置训练用当前统计并更新 running statistics，eval 用运行统计。LN 对每个样本／token 指定的特征维归一化，不依赖其他 batch 样本。

RMSNorm 使用均方根缩放，典型形式 $x/\sqrt{\operatorname{mean}(x^2)+\epsilon}$ 再乘可学习尺度，不执行减均值。实际统计轴取决于张量与模块参数。LLM 使用 LN／RMSNorm 与序列和 batch 特性有关，不表示 BN 在所有场景不可用。[RMSNorm 论文](https://arxiv.org/abs/1910.07467)

~~~python
import math
import torch
from torch import nn
~~~

<a id="k02"></a>

#### 手写 K02｜LayerNorm 与 RMSNorm

输入 [B,T,D]，这里只对最后 D 维归一化。weight、bias 为 [D]，用广播作用到每个 token。先背最后一维版本，不混进 BatchNorm 的 batch 统计。

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

**最小检查**：x=[1,3]，LayerNorm 无 affine 时约为 [-1,1]；RMSNorm 除以约 sqrt(5)，不减均值。所有值相同的输入也不应除以零。

**要解释**：LayerNorm 此处用总体式方差，即除以 D，不是 D-1；eps 在平方根内部。RMSNorm 不减均值，标准形式通常只有缩放。上面固定 FP32 教学；低精度实际实现常需更高精度累积统计，并核对输出类型与实现差异。

依据：[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)、[RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)。


### C07 [P1] Dropout 在训练与推理时怎样处理？

常见 inverted dropout 训练时以概率 p 丢弃，保留值除以 1-p，以保持条件期望；推理时直接使用完整激活。它引入随机正则，但不能把每次推理都当作重新独立训练的模型。

模块 Dropout 受训练模式控制；某些 functional API 需要明确传训练开关。SDPA 按传入的 `dropout_p` 应用 dropout，推理调用时应传 0。[官方 SDPA 说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

### C08 [P0] 写出注意力公式并标注维度。mask 放在哪里？

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V.$$

单头 Q 为 `[B,Tq,dk]`，K 为 `[B,Tk,dk]`，V 为 `[B,Tk,dv]`，分数为 `[B,Tq,Tk]`，输出 `[B,Tq,dv]`。mask 在 softmax 前加到 logits：允许位置加 0，屏蔽位置用负无穷等适当表示。

Q 表示当前查询，K 用于匹配，V 提供被加权的内容。自注意力常由同一 X 经不同可学习线性投影得到 Q/K/V；多头将投影输出拆成 `[B,h,T,dh]`，各头计算后拼接并作输出投影。朴素完整自注意力的交互计算量约 O(BT²d)，显式分数矩阵存储约 O(BhT²)，投影／FFN 开销另计。

causal mask 防读取未来，padding mask 排除无效 token；整行被屏蔽要单独定义行为。各 API 的布尔含义可能不同，SDPA 中 True 表示允许参与，不能把其他接口的 mask 原样复制。[公式来源](https://arxiv.org/abs/1706.03762)、[接口说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

~~~python
import math
import torch
from torch import nn
~~~

<a id="k03"></a>

#### 手写 K03｜缩放点积注意力，先写这一个函数

输入 q=[B,H,Tq,d]，k=[B,H,Tk,d]，v=[B,H,Tk,dv]。分数 [B,H,Tq,Tk]，softmax 沿最后的 key 维，输出 [B,H,Tq,dv]。

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

**记忆口令**：QK 转置、除根号、屏蔽未来、按 key 归一、乘 V。

**最小检查**：设 q=k=全零，v 第一个 token 为 1、第二个为 3。无 mask 时输出都为 2；使用下三角因果 mask 时，第一个输出为 1、第二个为 2。

**要解释**：

1. mask 在 softmax 前使用负无穷，使禁止位置概率为零；不能简单乘零，因为零分数仍可能分到概率。
2. 整行全屏蔽没有合法分布，本模板报错；真实 padding／空序列需定义有效位置与输出规则。
3. 普通注意力的 T×T 分数矩阵带来平方级空间；这一版没有实现 FlashAttention。分块、在线 softmax 和为什么结果不变，见 [FlashAttention 机制讲解](06-attention-serving-basics.md)。
4. 这里的有效行检查在 GPU 上会触发主机同步，是教学防错，不是高性能 kernel 写法。

**复杂度**：忽略投影，等长且 dv=d 时，时间 O(BHT²d)，分数中间量 O(BHT²)。

依据：[注意力原论文](https://arxiv.org/abs/1706.03762)、[PyTorch SDPA 的运算及 mask 约定](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。框架 SDPA 与 nn.MultiheadAttention 的部分布尔 mask 语义不同，移植时要核对。

<a id="k04"></a>

#### 手写 K04｜完整多头因果自注意力 MHA

输入输出都是 [B,T,D]。假设每个 token 都有效，且 T>0；本版不含 padding mask、GQA、位置编码和 attention dropout。位置表示由 K05 另加，不说它是完整工业 LLM。

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

**背的顺序**：Linear 得 QKV → [B,T,H,d] → [B,H,T,d] → 注意力 → 转回 [B,T,H,d] → 合成 D → 输出投影。

**最小调用**：

~~~python
attn = CausalSelfAttention(d_model=8, n_heads=2)
x = torch.randn(2, 4, 8)
y = attn(x)
assert y.shape == (2, 4, 8)
~~~

**常见追问**：transpose 之后为何用 contiguous 再 view？因为 stride 改了，直接 view 不总合法；reshape 有时会复制。头数增加而 D 固定时，每头宽度减小，并非总投影参数也翻倍。完整复杂度还包括约 O(BTD²) 的投影。

**正确性检查**：固定输入前缀，只修改未来 token，前缀输出应在浮点容差内不变。只检查输出 shape 不够。

<a id="k05"></a>

#### 手写 K05｜Transformer：区分 Pre-Norm 与 Post-Norm

**10/25 补充**：昨天的模板默认采用 Pre-Norm；现在显式区分两种结构，并提供同一份可切换代码。FFN 仍是对每个位置独立的非线性映射。

**先明确“前／后”相对于什么**：看一个子层中，归一化与子层 F、残差相加的先后。F 可以是注意力，也可以是 FFN。两种写法的 LayerNorm 都仍沿特征维 D 统计，不是换了归一化轴。

| 比较 | Pre-Norm，先归一化分支输入 | Post-Norm，残差相加后归一化 |
|---|---|---|
| 单个子层 | y = x + F(LN(x)) | y = LN(x + F(x)) |
| 分支顺序 | LN → F → 与原 x 相加 | F → 与原 x 相加 → LN |
| 残差干路 | 原 x 可以绕过本子层的 LN | 相加的结果整体还要过 LN |
| 本模板开关 | norm_first=True，保持旧版默认 | norm_first=False |

一个完整 block 有注意力和 FFN 两个残差子层，用 z 表示第一步结果：

- **Pre-Norm**：z = x + Attention(LN1(x))；y = z + FFN(LN2(z))。
- **Post-Norm**：z = LN1(x + Attention(x))；y = LN2(z + FFN(z))。

LN1、LN2 是两套独立参数，第二步处理的是更新后的 z。原始《Attention Is All You Need》使用 Post-Norm。[原论文 §3.1](https://arxiv.org/html/1706.03762#S3.SS1)

**默写口令**：Pre 是“先 LN、算分支、再相加”；Post 是“算分支、先相加、再 LN”。下面两条分支共享组件定义，只改变执行顺序。


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

**最小调用**：

~~~python
pre = DecoderBlock(d_model=8, n_heads=2, norm_first=True)
post = DecoderBlock(d_model=8, n_heads=2, norm_first=False)
x_norm_demo = torch.randn(2, 4, 8)
assert pre(x_norm_demo).shape == post(x_norm_demo).shape == (2, 4, 8)
~~~

只检查 shape 不够。把注意力与 FFN 的参数都置零，同时保持 LN 的初始 weight=1、bias=0：Pre-Norm 的输出就是 x；Post-Norm 的输出是 LN2(LN1(x))，一般不是 x。这个例子能检查你有没有把 LN 的位置写错。两者即使用相同权重，也不应期待输出相等。

**为什么会影响梯度？**

对单个子层，按列向量的 Jacobian 约定，Pre 的导数为 I + J_F J_LN，Post 为 J_LN(I + J_F)，各 Jacobian 在各自前向输入处取值。Pre 的残差干路保留一个不经该 LN 的恒等分量，Post 相加后的路径仍要经过 LN。这是结构上的直觉，不表示 Pre 全网络的梯度恒为 I。

在论文研究的初始化与设置下，Post-LN 靠近输出层的参数梯度较大，warm-up 有助稳定；Pre-LN 初始化梯度更温和。它支持训练稳定性的差异，不支持“Pre 永不梯度爆炸”“Pre 永远不需要 warm-up”或“Pre 一定效果更好”。[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

**final norm 怎样处理？**

常见 Pre-Norm 堆叠在全部 block 后再加一次 final norm，规范化送入输出头的表示；Post-Norm 的每个子层输出已经经过归一化。本教学模型选择 Pre 时加 final LayerNorm，Post 时用 Identity；这是一种明确约定，不是所有模型必须如此。是否有 final norm，不能替代检查每个子层的 LN 位置。

注意：这里说的归一化输出统计，还受 eps 和可学习 affine 参数影响；不能无条件断言输出永远严格零均值、单位方差。LayerNorm／RMSNorm 是计算方式，Pre／Post 是放置位置，两个维度分别判断。

若要求写一个能从 token ID 输出 logits 的简化 Transformer，再补以下部分。它是 **decoder-only 教学模型**，不是原论文完整 encoder-decoder；用可学习绝对位置 embedding，不是 RoPE。

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

构造时用 TinyDecoderLM(..., norm_first=True) 或 norm_first=False 显式选择。旧调用不传参数时仍为 Pre-Norm。切换结构并非无损转换训练权重；本例两种完整模型还因 final norm 约定而有少量参数差异，但单个 block 的参数形状与数量相同。

**PyTorch 接口对照**：nn.TransformerEncoderLayer 的 norm_first=True 表示 Pre，False 表示 Post，其默认是 False。此处自写类为延续昨天模板而默认 True，不能把两者默认值混为一谈。该官方层是接口顺序参考，不代表本例的 decoder-only 因果结构与其全部配置相同。[官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)

**P1：若追问现代 LLM 的 SwiGLU**，可把 FFN 换成以下门控形式。先写对两个并行升维、SiLU 和逐元素相乘，再降维；hidden_dim 要单独设定，不能改成三组矩阵后还直接声称与原 FFN 参数量相同。

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

依据：[GLU Variants 作者论文](https://arxiv.org/abs/2002.05202)。

**最小检查**：vocab=20、D=8、H=2、2 层，输入 [2,4] 的 token ID，输出应为 [2,4,20]。取 logits[:, :-1] 预测 token_ids[:, 1:]，再计算交叉熵；至少有两个 token 才能按此构造 next-token loss。

**要解释**：因果 mask 负责不看未来，位置 embedding 负责表达位置，二者不能替代。这里省略 dropout、权重共享等细节；不要直接把它称为某个现成大模型的精确复刻。


### C09 [P0] 为什么除以 $\sqrt{d_k}$？

若 q、k 各维近似独立、零均值、单位方差，点积是 d_k 项乘积之和，方差约 d_k。按平方根缩放使 logits 尺度较稳定，避免 softmax 过于尖锐带来的梯度问题。

这是解释动机的假设，不是实际网络每层都严格独立标准正态；缩放也不是对 q、k 分别做 L2 归一化。[原始论文](https://arxiv.org/abs/1706.03762)

### C10 [P1] MHA、MQA、GQA 的区别是什么？

MHA 通常每个 query 头对应独立 K/V 头；MQA 让多个 query 头共享一组 K/V；GQA 在两者之间，把 query 头分组共享 K/V。主要影响 KV cache、内存带宽与模型表现的权衡，query 头本身并不因此全部相同。

推理时节省多少要根据 KV 头数、长度与实现测量，不把减少缓存等同于整机等比例加速。[GQA 原论文](https://arxiv.org/abs/2305.13245)

### C11 [P1] RoPE 怎样表达位置？有 RoPE 就能无限外推吗？

RoPE 按位置在二维子空间旋转 Q/K，使点积中出现相对位置结构。它在注意力计算中引入位置关系，不是把一个位置向量简单加到 token embedding。

训练上下文分布、旋转频率和长度扩展方法都会影响长上下文表现；不能只因数学上可以计算更大位置就保证超长序列可靠。[RoPE 原论文](https://arxiv.org/abs/2104.09864)

<a id="k10"></a>

#### 手写 K10｜RoPE，P1 备用

下面按相邻偶／奇维成对旋转，输入 x=[B,H,T,d]，positions=[T]，d 为正偶数。对 Q、K 使用，不旋转 V。

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

**背的顺序**：按维度生成频率 → 位置乘频率 → cos／sin → 两维旋转 → 交错合回。位置为 0 时是恒等变换，旋转前后向量范数应近似不变。

**与 cache 连接**：当前 Q、新 K 在写入缓存前按实际绝对位置旋转，旧 K 已旋转，不重复处理；decode 不能每次把 positions 从 0 重置。某些模型采用前后半维配对，不能直接把这版用于那些权重而不转换约定。RoPE 不等于已保证无限长度外推。

依据：[RoFormer 作者论文](https://arxiv.org/abs/2104.09864)。


### C12 [P0] KV cache 缓存什么？prefill 与 decode 有什么不同？

10/26 专项加强：[逐 token 缓存、显存推导、PagedAttention 块表与量化](../1/inference-service.md)。先掌握本题基础，再用那里的位置与容量例子练追问；现有 K06 代码保留。

缓存各层已处理 token 的 K、V，使因果自回归生成新 token 时复用历史投影。prefill 同时处理已有上下文，decode 通常每轮推进一个新 token；decode 仍需让新 Q 读取历史 K/V，成本随上下文增加，并非 O(1)。

常规全长缓存近似占用 $2LBTn_{kv}d_hs$ 字节，L 为层数，B 为批量，T 为缓存长度，s 为每元素字节。例：L=32、B=1、T=4096、KV 头数 8、头维度 128、两字节存储，仅 K/V 约 512 MiB。权重、临时激活、分页／量化元数据另计。[GQA 对缓存的讨论](https://arxiv.org/abs/2305.13245)

~~~python
import math
import torch
from torch import nn
~~~

<a id="k06"></a>

#### 手写 K06｜KV cache：最容易写错的是位置偏移

本类继承 K04，复用投影和合头。cache 为 (历史 K, 历史 V)，均为 [B,H,L,d]；本轮输入 x=[B,S,D]，输出 y=[B,S,D]，新缓存长度 L+S。

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

**关键算例**：已有 3 个 token，本轮来 2 个，query 的绝对位置是 3、4，key 位置是 0—4；mask 两行为 [1,1,1,1,0] 与 [1,1,1,1,1]。直接做一个不带偏移的 2×5 下三角会屏蔽本该看到的历史。

**最小检查**：同一个 CachedSelfAttention，在 eval＋no_grad 下，整段一次前向的结果，应与“前 3 个 prefill，再逐 token decode”拼接结果近似一致。实际测试还应包含一次追加多个 token。

**要解释**：

- 缓存历史 K/V，新 token 只投影新的 K/V，但 Q 仍需读历史 K/V。
- 这是单层、无 padding、batch 不变、位置表示已在输入正确处理的教学例子。完整模型要维护各层缓存和位置偏移，不能直接以这段代替完整模型生成接口。
- 每步 cat 会复制缓存，适合解释语义，长序列应考虑预分配／分页管理；不是 PagedAttention 实现。
- 推理使用 eval＋no_grad／inference_mode，别让跨步图累积。新请求开始必须清空旧请求缓存。


### C13 [P1] 自回归训练为什么能并行，生成却通常串行？

teacher forcing 时已知目标序列，可以用因果 mask 并行计算各位置的 next-token loss；生成时下一个输入依赖刚采出的 token。常见对齐为位置 t 的 logits 预测位置 t+1 的 token，避免标签错位。

padding mask 与 loss mask 作用不同。SFT 可只监督助手回答，具体取决于目标与数据方案；工具返回通常作为上下文，是否对其 token 算损失必须明确。

追问：tokenizer 决定文本怎样变成 token，padding 与截断会影响训练。按 token 平均负对数似然的 perplexity 是 $\exp(L)$；不同分词器／数据处理下的困惑度不能直接视为同一尺度的模型质量。

<a id="k01"></a>

#### 手写 K01｜稳定 softmax 与交叉熵

**背的顺序**：减最大值 → 指数 → 求和 → 归一化。算交叉熵时直接算 log-softmax，避免先 softmax 再 log 导致下溢。

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

**最小检查**：logits 为 [[1000,1001]]，softmax 应约为 [[0.269,0.731]]，不溢出；标签为 [1]，loss 约 0.313。

**要解释**：减同一常数不改变概率；分类的输入是 logits；gather 取每行真类对应的 log 概率。前提是非空样本、合法标签、有限 logits；整行负无穷属于另一种情况，要按 K03 显式处理。

**复杂度**：N×C 输入时间 O(NC)。生产中通常用框架稳定实现，而不是为了“手写”替换已有算子。

依据：[PyTorch softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。


### C14 [P1] LoRA 和全量微调有什么区别？QLoRA 改了什么？

LoRA 冻结基础权重 W，学习低秩更新 $\Delta W=BA$；若 W 是 $d_{out}\times d_{in}$，新增参数约 $r(d_{in}+d_{out})$。减少可训练参数及部分优化器／梯度存储，不保证按同比例减少全部激活开销。

QLoRA 在低比特量化的冻结基础权重上训练适配器，并采用相应量化和存储技术；不等于把全部训练运算都变成低比特整数。[LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)

~~~python
import math
import torch
from torch import nn
~~~

<a id="k09"></a>

#### 手写 K09｜LoRA，P1 备用

基础线性层冻结，额外学习 A、B 两个低秩矩阵；输出为 base(x)+α/r·B(A(x))。

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

**要解释**：基础权重不更新，新增参数 r(d_in+d_out)；B 初始为零使初始增量为零，A 仍应有非零初始化。如果 A、B 同时全零，两边可能都得不到起步梯度。代码没有实现 QLoRA，也未实现权重合并／dropout／adapter 管理。

依据：[LoRA 作者论文](https://arxiv.org/abs/2106.09685)。


### C15 [P1] 预训练、SFT、偏好优化分别改变什么？

常见自回归预训练以大量文本的 next-token 目标学习分布；SFT 用示范输入／输出或轨迹塑造任务行为；偏好优化利用回答偏好或奖励调整策略，具体流程见 D16—18。

RAG 在推理时加入外部内容，与更新参数作用不同。SFT 不保证只学格式，RAG 也不自动消除幻觉；是否需要训练要看任务、数据、成本和评价。

追问生成：temperature 缩放 logits，top-k 保留最高的 k 项，top-p 保留累计概率达到阈值的高概率集合再采样；参数影响随机性与候选覆盖，低温不保证事实正确，也不能在比较 pass@k 时隐去采样设置。

### C16 [P1] 显存不够时有哪些办法？梯度累积等于大 batch 吗？

先区分权重、梯度、优化器状态、激活和缓存的开销。可缩短序列／微批、用混合精度、梯度累积、激活检查点、适配器或分片。检查点以重算换存储，累积不会自动减少单个样本的激活需求。

累积要正确归一化，样本／有效 token 数不同不能总按同一个固定步数简单平均；还受 BN、随机层和更新频率影响，并非任何情况下都严格等价于一次大 batch。[官方训练教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

### C17 [P0] loss 不下降或变成 NaN，按什么顺序排查？

先用很小数据尝试过拟合，查输入、标签、mask、loss reduction、shape／dtype／device；再查参数是否在优化器、梯度是否存在、清零与 step 是否执行。NaN 则定位首次非有限值，检查 log(0)、除零、全屏蔽 softmax、溢出和异常梯度。

之后调整学习率、初始化、混合精度或裁剪；验证 train／eval 切换。单纯把 loss 打印改小、无依据 clamp 或忽略异常，不等于修复训练。[训练过程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[数值稳定](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)

### C18 [P1] DDP、FSDP 大致解决什么问题？

DDP 通常每个进程持有完整模型，分不同数据并同步梯度；FSDP 对参数、梯度、优化器状态采用分片策略，并在计算时执行相应通信。数据并行、张量并行、流水线并行的切分对象不同。

相关通信先分清三种结果：**AllGather** 将各卡分片拼成完整张量并让大家都持有；**AllReduce** 对各卡张量的对应位置归约、让大家持有完整结果；**ReduceScatter** 则让各卡持有归约结果的一片。理想 p 卡 Ring 中，完整逻辑张量为 S bytes 时，每卡发送量分别为 `(p−1)S/p`、`2(p−1)S/p`、`(p−1)S/p`，接收另算。这里 AllGather 的 S 指完整输出，另外两者指每卡输入，不能直接把三种输入大小当成相同。具体数据流、四卡手算，以及长序列分片的 **Ring Attention**，见[通信与并行专项](../1/parallel-moe.md)。

还要考虑有效 batch、sampler、梯度同步和保存加载。首次面试到概念层即可；没有实际做过就说明，没有必要把名词写成多卡经验。

### C19 [P1] CNN、普通 Transformer、深度展开怎样联系你的 ICCV？

CNN 的局部连接与权重共享形成归纳偏置，注意力可建立输入相关的交互；复杂度和感受范围要看具体结构。深度展开把迭代优化步骤映射为可学习模块，将已知物理／退化结构与数据学习结合。

准备解释 ICCV 中退化模型进入哪一步、哪些量固定／学习、Transformer 子模块做什么、为何需要条纹机制、用什么消融验证。不是有物理模块就自动可解释或泛化，仍需要实验与条件说明。

常见编码追问：普通二维卷积不分组、带 bias 时参数量为 $k_hk_wC_{in}C_{out}+C_{out}$；stride、padding、dilation 改变输出形状／感受野，但不改变同一层的上述参数量。别把参数量和 FLOPs 混同。

### C20 [P1] 设随机种子就能保证复现吗？怎样做有效消融？

还需要版本、数据、预处理、硬件／算子、并行设置和检查点等；部分算子可能非确定。不同 seed 评估不同随机性，重复测试同一权重不等于重复训练。

消融应说明删了什么、其余如何控制、参数／FLOPs／数据／训练预算是否变化。若改变多项组件，不能把全部提升归因于单项；见 B20。

## 手写练习

1. 写完整训练／验证循环，包含 `zero_grad → forward → loss → backward → step`，验证时 `eval＋no_grad`；按样本或有效 token 正确汇总 loss。解释每一步缺失的后果。
2. 写单头因果注意力；输入 `[B,T,d]`，softmax 在 key 维，未来位置屏蔽；对照未屏蔽版本检查未来 token 是否影响早期输出。这里先不用多头／缓存。
3. 手算 C12 的缓存例子，再分别把 T 翻倍、KV 头数改为 1，区分缓存变化与总体显存变化。
4. 从自己的 ICCV 或 4a 代码取一个能解释的函数，逐项标输入／输出维度、计算量、异常路径与验证方式；借助工具实现的部分也要能说明。

核心推导参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，PyTorch 练习按[官方教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)与你实际安装版本核对。这里没有要求为准备面试训练大模型或购买算力。
