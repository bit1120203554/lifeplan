# MoE 负载优化｜从 token 怎样分配讲起

补充日期：2026-10-26。面向第一次学习 MoE 的 AI Infra 首轮准备。先读 **1—5、7、9、12**；动态偏置、梯度公式和通信流水线用于追问。先会解释机制，再练手写，不要求临时实现多卡系统。

本文的 MoE 是**神经网络专家层**。旧 project4a 组合程序评分头的做法不构成这种 MoE 的训练、专家并行或通信调优经历。

## 1. 专家、router、assignment 分别是什么

Transformer 先把 token 变成向量。这里的“专家”通常是一套有自己参数的 FFN，而不是一个完整聊天机器人，也不是预先标注为“数学专家”的角色。

假设有 4 个专家 E0—E3。router 是根据 token 向量输出 4 个分数的小网络；若采用 softmax，可得到 `[0.6, 0.25, 0.1, 0.05]`。top-2 选择 E0 和 E1，将向量各送一份，分别计算，再按选中权重合并输出。是否重新归一化选中权重取决于具体模型。

**assignment 是“一次 token→专家的分配”。** 8 个 token、top-1 是 8 个 assignment；100 个 token、top-2 是 200 个 assignment。讨论负载时先数这个，不能只数原始 token。

```text
router：选谁、给多少权重
dispatch：按专家整理并送出 token 向量
expert：用自己的 FFN 参数计算
combine：按原 token 位置加权合并
```

所有专家可放同一张卡，也可通过 EP（专家并行）放在不同卡上。只有分布到不同设备时，才需要考虑相应的跨卡 token 交换。

## 2. 为什么负载不均会慢：一个贯穿全文的例子

8 个 token、4 个同样大小的专家、top-1，实际路由为：

```text
E0：t0 t1 t2 t3 t4 t5   → 6 次计算
E1：t6                   → 1 次计算
E2：t7                   → 1 次计算
E3：                     → 0 次计算
```

平均应该是 `8/4=2`，最忙专家是平均的 `6/2=3` 倍。

**只为理解，假设每专家独占一张同速卡、每次计算耗时相同且随数量线性增加，并暂不计通信。** 这轮要等 E0 完成 6 份工作；若分成 `[2,2,2,2]`，只需等待 2 份工作。实际 GPU 有并行、启动开销和不同 GEMM 效率，不能据此宣称真实系统必然加速 3 倍。

失衡常见原因包括：初始化时的微小偏好；部分专家被选得多、训练得多，进一步吸引 token；输入领域改变，例如推理请求大多是代码；小 batch 的随机波动。专家擅长不同模式是正常现象，目标是兼顾任务质量与资源利用，**不要求每个 token 对所有专家都给相同概率**。

还要区分两个层次：

| 层次 | 问的问题 | 为什么有区别 |
|---|---|---|
| 逻辑专家负载 | 每个专家收到了多少 assignment？ | 反映 router 的选择分布 |
| 设备负载 | 每张卡合计要处理多少计算／通信？ | 一张卡可能放多个专家，也可能有热门专家的副本 |

专家数平均分到各卡，不等于工作量平均。比如 `[6,1,1,0]`，卡 A 放 E0/E1、卡 B 放 E2/E3，负载就是 7 对 1。

## 3. 训练方法一：负载均衡辅助损失，到底怎样起作用

### 3.1 先用语言解释

训练原本主要优化预测正确与否，例如下一个 token 的交叉熵。增加一项辅助目标，给“总把大量 token 送去少数专家”的倾向施加压力：

\[
L_{total}=L_{task}+\lambda L_{balance}.
\]

`λ` 是两项目标的权衡。它不是对某次前向直接执行“把 E0 的 token 搬给 E3”，而是**通过梯度更新 router，影响后续路由**。当前已经完成的 top-k 不会因为多算一个 loss 当场重新分配。

### 3.2 最常见的入门公式：先限定 top-1

令 E 为专家数，T 为本次统计范围内的有效 token 数：

\[
f_i=\frac{\text{实际选中专家 }i\text{ 的 token 数}}{T},\qquad
P_i=\frac1T\sum_{t=1}^{T}p_{t,i},
\]

\[
L_{balance}=E\sum_{i=1}^{E} f_iP_i.
\]

- `p[t,i]`：token t 的 router 对专家 i 的 **top-k 前完整 softmax 概率**。
- `f[i]`：实际接到多少，是离散的硬选择统计。
- `P[i]`：平均分到多少概率，是连续、可微的量。
- 外面的 E 用来统一量级：两者都均匀时，这个表达式等于 1。

这是 [Switch Transformer §2.2](https://arxiv.org/html/2101.03961v3#S2.SS2) 的入门形式，本文将 λ 单独写在总损失中。**均匀时等于 1 是参考值，不是仅凭一个 loss 数值就能证明负载已均衡，也不要把 1 背成所有可行状态下的严格下界。**

### 3.3 用同一个 8-token 例子算一遍

假设前 6 个 token 的概率都是 `[0.7,0.1,0.1,0.1]`；t6 为 `[0.1,0.7,0.1,0.1]`；t7 为 `[0.1,0.1,0.7,0.1]`。

top-1 得 `[6,1,1,0]`，所以：

```text
f = [0.75, 0.125, 0.125, 0]
P = [0.55, 0.175, 0.175, 0.10]
L_balance = 4 × (0.75×0.55 + 0.125×0.175 + 0.125×0.175)
          = 1.825
```

作为对照，若每个专家恰好被 2 个 token 选择，每个 token 仍给被选专家 0.7、其余各 0.1，则 `f=P=[0.25,0.25,0.25,0.25]`，结果为 1。这里每个 token 仍有明确偏好，**不是把每一行概率都改成 0.25**。

### 3.4 top-k 不可微，为什么还能训练？

`f` 的硬计数通常当作常数，不对 top-k 索引求导；梯度通过 P 回到 softmax、router 参数。固定这次的 f 看，最热门专家的 f 大，提高它的 P 会付出较大的辅助代价。

进阶公式：对 token t、专家 j 的 logit，固定 f 时有

\[
\frac{\partial L_{balance}}{\partial z_{t,j}}
=\frac ET p_{t,j}\left(f_j-\sum_i f_ip_{t,i}\right).
\]

这说明它调节的是相对偏好，而不是给离散索引求梯度。梯度下降之后，部分 token 的专家排名可能改变；仍需与任务目标共同优化。

**为什么不直接最小化计数方差？** 方差能描述失衡，但直接从硬 top-k 得到的计数通常没有可用的普通梯度路径。不能写一个计数方差就默认 router 会被训练；需要可微代理或其他路由优化方法。

### 3.5 三个必须保留的边界

1. λ 太小可能约束不足，太大可能牺牲任务适配与专家分工；不承诺越大越好。
2. 统计范围可能是一条序列、一个 microbatch 或跨 rank 的范围，结论不同。多个 rank 的长期总量均衡，不保证某一步或某张卡不堵。
3. top-2 时可以把 `f_i=count_i/(T×2)`，沿用 `EΣf_iP_i` 作一种明确归一化的扩展。实现也可能把 k、E 的因子放在别处，需连同 λ 看，不能只比较 loss 数值。后面的代码采用这一约定。

## 4. 容量 capacity：限制这轮最多接多少，与“学会均衡”不同

常见教学估算：

\[
\text{平均 assignment 数}=\frac{Tk}{E},\qquad
C=\left\lceil c\frac{Tk}{E}\right\rceil.
\]

- C：每个专家这一轮的容量上限。
- c：capacity factor，给平均数量留出的余量。
- 取整、最小容量、按组统计等以实现为准；公式中的 T 与 E 必须属于同一个路由范围。

8 个 token、top-1、4 专家，若 c=1.5，则 C=3。原始 `[6,1,1,0]` 下，E0 有 3 个 assignment 超额。**虽然总容量为 12，大于总需求 8，热门专家仍会超额；容量不是所有专家共享的一个池子。**

超额怎么办？具体方法不同：

| 选择 | 实际含义 | 代价 |
|---|---|---|
| 丢弃超额 assignment | 不计算被溢出的那条专家分支 | 改变本层计算，可能影响质量 |
| 改送其他专家 | 根据明确的替代路由规则重新选择 | 额外调度，可能降低路由适配性 |
| 提高容量 | 给热点更多空间 | 更多缓冲／padding 成本，也没有消除热点工作量 |
| dropless | 接受实际被选中的所有 assignment | 处理动态形状、实际峰值内存与长尾工作量 |

**“丢 token”通常不是把文字从整段序列删除。** 在一些带残差的实现中，top-1 溢出时跳过该 token 的专家分支，残差路径仍在；top-2 也可能只少算其中一个 assignment，如何合并需看实现。

再练题设：100 token、top-2、4 专家，平均 50；若 c=1.2，教学容量为 60。不能算成 `100/4×1.2=30`。

容量／drop／padding 是不同开关。例如 [Megatron 官方说明](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)分别列出 capacity、drop policy 与 pad-to-capacity。限制容量不等于一定给空位做 GEMM。

## 5. padding 与 dropless 为什么会被反复提到

GPU 喜欢规整的大矩阵，但每个专家收到的 token 数不同。

假设容量都是 3、采用上节的丢弃方案，实际保留 `[3,1,1,0]` 共 5 份工作。**若进一步采用“每专家填到容量”的实现**，会准备 12 个位置，其中 7 个是假填充；是否真的为所有空位执行乘法，仍取决于 kernel。

另一种做法不因这个容量截断路由，保留原始 `[6,1,1,0]` 的 8 份工作，再用能处理不等长输入的算子执行。**dropless 只说明不因专家容量丢弃路由分配，不意味着无 padding、无峰值内存、无热点，也不保证更快。** 对齐硬件块大小时仍可能做少量 padding。

[MegaBlocks](https://arxiv.org/abs/2211.15841) 是用块稀疏计算支持动态 MoE 工作量的一种代表性系统。块稀疏是将有用计算组织成可由内核处理的非零块，不必逐专家填满同一个大容量；不是“把 token 均匀分了”这一训练策略。

## 6. [P1] 不用主要的均衡辅助梯度，怎样调路由？

一种方法给每个专家维护路由校正偏置 b：选择专家时比较 `score[t,i]+b[i]`。热点专家下调 b，空闲专家上调 b，让后续选择更容易分散。

```text
观测这一训练步的负载 → 热门减偏置、冷门加偏置
下一步：用“适配分数＋偏置”选择 top-k
```

例如原始分数 `[0.50,0.48]` 选择 E0；校正偏置 `[-0.03,+0.03]` 后，用于选择的分数变成 `[0.47,0.51]`，会选 E1。这里分数不要求是 softmax 概率。

[Auxiliary-Loss-Free 方法](https://arxiv.org/abs/2408.15664)通过负载反馈更新偏置，避免用同一项均衡辅助梯度直接干扰主任务梯度；更新速度过大仍可能振荡。其关键是反馈控制，不是“没有损失，随便平均分”。

[DeepSeek-V3 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)采用相关思路：**偏置只影响选谁，最终专家输出的门控权重仍来自原始适配分数**。该报告还保留很小的序列级均衡辅助项，因此不能背成“DeepSeek-V3 完全没有任何辅助损失”。这是训练机制，不应直接作为给任意已部署模型随意修改 router 的建议。

## 7. grouped GEMM：分配好了，怎样把不同专家算得更高效

GEMM 就是矩阵乘法。一个专家 FFN 的第一层可写成：

\[
X_iW_i,\quad X_i\in\mathbb{R}^{M_i\times D},\quad
W_i\in\mathbb{R}^{D\times F}.
\]

`M_i` 是该专家收到的 assignment 数。专家不同，W 不同，M 也可能不同；激活函数和第二个线性层还要继续计算。

比如同一卡上 3 个专家收到 `[2,7,0]`，就有 2 行与 7 行的两项有效矩阵乘，空专家可跳过。分别启动很小的计算，启动和调度成本可能突出，GPU 也可能吃不满。

**grouped GEMM 将多项、允许不同形状的矩阵乘组织为一组，由分组内核调度其工作块**，有机会减少多次启动和细碎任务的开销。它保留各专家自己的权重；不能把所有 token 接成一个矩阵，统一乘 W0，就声称等价。

一个直观的工作列表是：

```text
任务 A：X0 的 2 行 × W0
任务 B：X1 的 7 行 × W1
任务 C：0 行，跳过
→ 分组内核处理 A/B 的有效工作块
```

依据：[CUTLASS 分组调度](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)。**它主要改善执行效率，不修改 `[2,7,0]` 的路由分配，也不能独自解决跨设备热点。** 性能还取决于维度、对齐、分块与设备。

## 8. [P1] All-to-All、拓扑与重叠：工作送过去也要花时间

EP 中，token 源卡与目标专家所在卡可能不同。dispatch 常先按目标整理，再发送 token 向量；专家计算后，输出返回源卡，恢复原顺序并加权。需要传送的主要是激活向量和相应元信息，不是每次请求把专家的全部参数传过去。

**All-to-All** 描述多个参与者向不同目标交换各自分片的集合通信。它可以实现这种分发，但调用它不会自动让各专家负载相等。热点卡的接收量、计算量和返回量可能都更大。

这里有三种不同优化：

- **减少重排／拷贝**：将排列、打包等步骤融合，减少中间缓冲的读写与启动。
- **拓扑感知**：考虑节点内与跨节点链路，把布局或受约束路由与网络结构一起设计；更偏本地的路由可能影响模型适配。
- **通信计算重叠**：一个可独立处理的块／microbatch 已到达时开始计算，同时推进另一个的通信。必须有可用工作与正确依赖；不能在输入未到时计算同一项专家输出。

自拟时间例：dispatch 2 ms＋计算 5 ms＋返回 2 ms，串行 9 ms。存在独立工作且资源允许时可隐藏一部分通信；实际依赖、争用和尾部等待决定还能露出多少。不能一律改成 `max(2,5,2)=5 ms`。

MoE 系统的分发、分组计算和重叠是独立的设计维度，参考 [Megatron MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)。不要把“开异步参数”当作已证明端到端变快。

## 9. 推理时为什么更应该谈“专家摆在哪里”和“热门副本”

训练过的 router 可能在训练分布上总体均衡，但在线请求集中于某个领域、decode 每轮 token 很少，仍会失衡。**推理时临时加辅助 loss，没有参数更新，就不会自动使当前路由变均匀。**

先区分“逻辑专家”与它的“物理副本”：逻辑 E0 是一套固定参数；可以在两张卡上各放相同权重的 E0 副本，让已经选中 E0 的不同 token 去不同副本。

继续两卡例子：

```text
原布局：A 放 E0/E1 → 6+1=7；B 放 E2/E3 → 1+0=1
增加副本：B 额外放一份 E0
把 E0 的 6 个 assignment 分成 A 算 3 个、B 算 3 个
结果：A 为 3+1=4；B 为 3+1+0=4
```

这是自拟负载算例，假设 B 有额外显存、复制与调度成本可承受。**token 仍送给逻辑 E0，不是把 E0 随意换成 E3。** 固定权重、相同计算设置下可以保持专家函数语义，浮点结果仍可能有舍入差异。冗余权重挤占显存，也可能减少 KV cache 容量；迁移／更新映射也有成本。

[vLLM EPLB 官方说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#expert-parallel-load-balancer-eplb)描述了按前向收集负载、周期调整专家映射及冗余专家的部署机制。不同版本支持和参数会变，这里先学原理，不要求记命令。若用于训练，还会有副本梯度与参数一致性问题，不能直接照搬这个固定权重例子。

## 10. 面试官问“MoE 很慢，你怎么优化”，按什么顺序回答

先确认训练还是推理、top-k、专家维度、EP 范围、batch／长度、硬件与目标。在真实工作负载上保留基线，再看：

| 先观测 | 帮助判断 | 对应改动 |
|---|---|---|
| 每专家 assignment 数、每卡总数、最忙/平均 | 专家或设备是否有热点 | 训练均衡目标；推理专家布局／副本 |
| drop 数、容量、padding 位置、峰值显存 | 是否因限制少算或填充太多 | 容量、dropless、内核处理方式 |
| GEMM shape 与设备时间线 | 小矩阵／细碎启动是否突出 | grouped GEMM、融合、适当批量 |
| dispatch／combine 字节、链路与暴露的通信时间 | 网络和重排是否在关键路径 | 拓扑、打包、通信实现、有效重叠 |
| 端到端 step 时间或 tokens/s、TTFT／TPOT／p99 | 优化有没有改善用户目标 | 正确性、任务质量及延迟约束下取舍 |

`[6,1,1,0]` 的最忙/平均=3。这个比值描述分布，不是自动等于加速上限；不同专家大小或设备速度不同时，数量也不能直接等同时间。辅助 loss、GPU 利用率都不能单独代替端到端证据。

## 11. 手写补充：硬计数＋可微概率＋辅助损失

这段与 [K07 Top-K MoE](05-model-system-coding.md#k07)衔接：K07 负责专家前向，下面负责解释并计算均衡项。只处理一个非空本地路由范围的有效 token，**没有跨 rank 统计、capacity 执行、token 丢弃或高效 kernel**。

本版统一 `f=count/(T*k)`、`P=全体专家 softmax 概率的 token 均值`。k=1 对应上面的 Switch 形式；k>1 为本页明确的归一化扩展，不冒充所有模型的实现。

~~~python
import torch

def moe_balance_loss(router_logits, top_k=1):
    # router_logits: [T,E]，只含有效 token，未做 top-k 截断
    if router_logits.ndim != 2:
        raise ValueError("router_logits 必须是 [T,E]")
    T, E = router_logits.shape
    if T == 0 or not 1 <= top_k <= E:
        raise ValueError("需要 T>0 且 1<=top_k<=E")
    if not router_logits.is_floating_point():
        raise ValueError("router_logits 必须是浮点张量")

    # 半精度输入时用 FP32 统计；FP64 检查保留 FP64。
    work = router_logits
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    probs = work.softmax(dim=-1)                 # [T,E]，保留梯度
    with torch.no_grad():
        ids = probs.topk(top_k, dim=-1).indices  # [T,k]，硬选择
        counts = torch.bincount(ids.flatten(), minlength=E)
        fraction = counts.to(probs.dtype) / (T * top_k)
    mean_probs = probs.mean(dim=0)               # [E]，可微
    loss = E * (fraction * mean_probs).sum()
    return loss, counts

# 6 个 token 选 E0，一个选 E1，一个选 E2。
p = torch.tensor([[0.7, 0.1, 0.1, 0.1]] * 6 + [
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
], dtype=torch.float64)
logits = p.log().requires_grad_()
balance, counts = moe_balance_loss(logits, top_k=1)
assert counts.tolist() == [6, 1, 1, 0]
assert torch.allclose(balance, balance.new_tensor(1.825))
balance.backward()
assert logits.grad is not None and torch.isfinite(logits.grad).all()
~~~

训练时将返回的 `balance` 以 λ 加入原任务 loss，再一起反传；**不要对整个 balance 或 mean_probs 做 detach／item 后再加入，否则辅助项的梯度会丢失。** `counts` 的硬统计不求导是刻意的。`.item()` 可以用于日志，但日志数值不能替代用于训练的张量。

这里的概率必须是 top-k **之前**的完整分布。如果用 K07 中“选中后重新归一化”的权重代替，尤其 k=1 时全是 1，就不是上面这个损失了。K07 的原版仍没有自动接入均衡项。

容量只需先会手算。下面代码展示的是**按专家数额做截断后的数量统计**，并没有挑选具体 token，也没有执行残差／重新归一化等模型逻辑。

~~~python
import math

def capacity_counts(counts, num_tokens, top_k, capacity_factor):
    if not counts or num_tokens <= 0 or not 1 <= top_k <= len(counts):
        raise ValueError("参数不合法")
    if capacity_factor <= 0 or any(n < 0 for n in counts):
        raise ValueError("容量系数须为正，计数不得为负")
    if sum(counts) != num_tokens * top_k:
        raise ValueError("计数之和必须等于 token 数乘 top-k")
    capacity = math.ceil(capacity_factor * num_tokens * top_k / len(counts))
    kept = [min(n, capacity) for n in counts]
    overflow = sum(counts) - sum(kept)
    return capacity, kept, overflow

assert capacity_counts([6, 1, 1, 0], 8, 1, 1.5) == (3, [3, 1, 1, 0], 3)
assert capacity_counts([100, 50, 30, 20], 100, 2, 1.2) == (60, [60, 50, 30, 20], 40)
~~~

## 12. 60—90 秒面试表达＋自检

可以练习这样讲：

> MoE 每个 token 只选少数专家，但 router 可能把大量 token 分到同一专家。比如 8 个 token、4 专家、top-1，分成 6、1、1、0 时，热点专家可能拖住其他设备。训练侧可以用负载均衡辅助损失，结合实际分配比例和平均路由概率，通过梯度影响后续选择；它不会直接重排当前 token。容量是每轮每专家的接收上限，需要权衡溢出、padding 和质量，dropless 也不等于没有热点。执行侧用 grouped GEMM 改善不等长小矩阵计算，并分析重排、All-to-All 和可隐藏的通信。推理侧还会依据实时负载调整专家布局，或给热门逻辑专家增加同权重副本。最终看每专家／每卡分布、显存、端到端时延与吞吐，并检查输出和任务质量。

初学先做到前四句，然后补执行与推理；不把这段表述成自己已经做过多卡优化。

| 自检题 | 应能解释的关键 |
|---|---|
| 100 token、top-2、4 专家，平均多少？c=1.2 时容量多少？ | 50，60；数 assignment |
| 总容量大于总需求为什么仍溢出？ | 容量分属每个专家，热点不能直接借空闲名额 |
| top-k 不可微，辅助损失梯度从哪走？ | 硬计数停止梯度，完整平均概率保留梯度 |
| balance=1 能保证均衡吗？ | 不能；须看实际计数和统计范围 |
| 每个 token 概率都均匀，是否最理想？ | 会削弱选择与分工，且平分概率遇到 tie 不保证硬选择均衡 |
| grouped GEMM 会改变每个专家收到多少 token 吗？ | 通常不改变路由，优化已分配工作的执行 |
| dropless 会让所有卡一样忙吗？ | 不会，仍须处理不均匀实际工作量 |
| 给热门专家副本，为什么比换一个冷门专家更能保留语义？ | 同一个逻辑专家的同权重副本，函数相同 |
| 推理时加一项 balance loss 有用吗？ | 不更新参数就不会自动改变路由；用部署策略处理负载 |

## 13. 依据、可信度与校验

- 辅助损失：[Switch Transformer §2.2](https://arxiv.org/html/2101.03961v3#S2.SS2)。公式可信度高，本文将 λ 与未加权 loss 分开，top-k 扩展口径单独声明。
- 训练反馈路由：[Auxiliary-Loss-Free](https://arxiv.org/abs/2408.15664)、[DeepSeek-V3 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)。讲机制，不将特定模型配置推定为本岗位技术栈。
- 不规则执行：[MegaBlocks](https://arxiv.org/abs/2211.15841)、[CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)。具体速度取决于工作负载，未套用论文的加速数字。
- 实现与部署：[Megatron MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)、[vLLM EP／EPLB](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)。机制可靠，main／latest 接口随版本变动。
- 文中的 token 分配、容量、损失和时间例子均为自拟教学算例；不是本人的项目指标或讯飞内部配置。

2026-10-26 在现有 PyTorch 2.13.0+cpu 环境校验：两个代码块完整执行；40 组不同 token／专家／top-k 的硬计数、参考损失和解析梯度对照通过；均衡分配、均匀概率但硬选择集中、loss 小于 1 的反例通过；FP16／BF16／FP32 梯度有限且存在；非法输入检查及两组容量算例通过。相关文档代码语法与本地链接已核对。CPU 数值校验不构成 GPU 性能证据，也不说明本人已掌握或实践过这些内容。
