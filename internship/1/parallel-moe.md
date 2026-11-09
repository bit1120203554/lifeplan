# 多卡通信与 MoE：数据怎样分片、传递与计算

阅读导航：[集合通信](#collective) → [并行与分片](#parallel) → [MoE 与手写代码](#moe) → [Ring Attention](#ring)。

先按 rank 和集合通信建立数据流，再读 DP／TP／PP／EP／CP、MoE 路由和负载，最后看 Ring Attention 与挂起排障。通信量的 S、每卡发送与接收口径在原位定义；计算按理想模型估计。

## 通信组、rank 与设备

进程是运行中的程序实例。常见部署让一个进程管理一张卡。rank 是进程在组内的编号，world size 是组内进程数，local rank 常为节点内编号。两机各四卡时，全局 rank 可为 0—7，各机 local rank 为 0—3。

process group 决定一次通信由谁参与。不同协作方式可有各自子组，因此一次集体通信的规模不必等于总卡数。torchrun 等启动器设置进程环境，程序仍需初始化组、绑定设备，并让参与者按兼容次序通信。

**rank 是卡吗？** rank 首先是通信组内参与者的编号；常见的一进程一 GPU 部署中，可以先把 rank 0—3 理解为对应四张卡。编号、进程和设备并非同一概念；一个进程也可以管理多个设备。以 NCCL communicator 为例，每个 rank 绑定一个 CUDA 设备，具体绑定由初始化决定。[NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)

<a id="collective"></a>
## 集合通信：操作语义、Ring 过程与流量

collective（集合通信）规定一组进程怎样交换或归约数据。**归约**是对对应位置执行 sum/max 等操作；**收集**保留原数据并拼接；**分发分片**让不同 rank 保留结果的不同部分。下面先用 SUM 归约、等长分片解释。

### AllGather：大家各有一片，最后大家都有全部

四卡各有一段不同的数据：

```text
输入：rank0 [A]，rank1 [B]，rank2 [C]，rank3 [D]
输出：每个 rank 都有 [A, B, C, D]，按 rank 顺序排列
```

A/B/C/D 是张量分片，不是要相加的数。AllGather **不做对应位置求和**，输出元素数是单卡输入的 p 倍，p 为参与通信的 rank 数。参数分片训练中，一张卡原本只保存部分参数，计算前可用 AllGather 取得该计算单元所需的完整参数；TP 也可能用它拼接不同输出列。

Ring（环形）实现可将四卡按 `0→1→2→3→0` 连接。每轮向下一卡发送一片，同时从上一卡接收一片，并在下一轮继续转发刚收到的数据：

| rank0 的进度 | 新收到 | 目前已有 |
|---|---|---|
| 初始 | — | A |
| 第 1 轮 | D，由 rank3 发来 | A、D |
| 第 2 轮 | C，经 rank3 转发 | A、C、D |
| 第 3 轮 | B，经 rank2、rank3 转发 | A、B、C、D |

各卡同时工作，最终将数据放进按 rank 排列的输出位置。p 卡需要 p−1 轮，每轮发送一片。这里描述的是 Ring 算法；AllGather 这个操作本身也可以由其他算法实现。

也可以同时追踪四张卡每一轮收到的块：

| 时刻 | rank0 | rank1 | rank2 | rank3 |
|---|---|---|---|---|
| 初始持有 | A | B | C | D |
| 第 1 轮收到 | D | A | B | C |
| 第 2 轮收到 | C | D | A | B |
| 第 3 轮收到 | B | C | D | A |

例如 A 依次从 rank0 → rank1 → rank2 → rank3；rank1 收到后既保存供自己使用，也在下一轮转发。所有卡并行进行发送和接收，并非等 A 走完整圈之后再传 B。这个“环”是通信顺序，不要求设备在物理上摆成圆圈；GPU 互连、PCIe、跨机网络等决定实际路径。对 ReduceScatter，沿环传递的是对应片的部分归约结果，接收者还需把自己的贡献加进去；Ring Attention 传递的是 KV 块，由各卡用本地 Q 计算。

### ReduceScatter：对应位置先归约，每卡只保留一片结果

四卡各有一个**相同长度、相同位置含义**的向量，例如各自算出的参数梯度：

```text
rank0：[   1,    2,    3,    4]
rank1：[  10,   20,   30,   40]
rank2：[ 100,  200,  300,  400]
rank3：[1000, 2000, 3000, 4000]

对应位置 SUM：[1111, 2222, 3333, 4444]
ReduceScatter 输出：rank0 [1111]，rank1 [2222]，rank2 [3333]，rank3 [4444]
```

每个输出元素都包含所有 rank 对这个位置的贡献。它不是把各卡的原向量各自切掉一部分，也不是按整卡把向量压成一个数。

实际实现不必先让每卡得到完整和再切分：将每张卡的输入划成 p 片，沿环逐步发送、接收并累加，p−1 轮后每卡只留下归约完成的一片。例如把第一片叫 A，四卡的局部贡献分别为 A₀…A₃，一种安排为：

```text
rank1 的 A₁ → rank2 加 A₂ → rank3 加 A₃ → rank0 加 A₀
最终第一片完整和留在 rank0；其他片也同时沿环累加，分别到达自己的持有者。
```

分片训练中，每卡只负责一部分参数的优化器更新，因此可以用 ReduceScatter 将梯度汇总到对应的持有者。**操作输出变小，不代表输入和通信工作区从来没有占过显存。**

### AllReduce：对应位置归约，大家都保留完整结果

用上面的四个输入向量，AllReduce SUM 后，**每个 rank 都得到 `[1111,2222,3333,4444]`**，输出大小与每卡原输入相同。DDP 用它同步梯度；某些 TP 矩阵切分用它合并部分和。SUM 与平均不同，需要平均时还要按训练语义处理除数。

一种常见实现就是：

$$
\operatorname{AllReduce}(X_0,\ldots,X_{p-1})
=\operatorname{AllGather}\bigl(\operatorname{ReduceScatter}(X_0,\ldots,X_{p-1})\bigr).
$$

第一阶段每卡拿到完整和的一片；第二阶段互相交换这些**已经归约好的片**，恢复完整和。Ring 实现合计 2(p−1) 轮。这里是操作语义和一种实现分解，不能据此认定通信库一定执行两个独立 API 调用；浮点加法顺序也可能造成微小舍入差异。

### 通信量：先约定 S 是完整张量大小

以下按**等长分片的理想 Ring、一次操作**计数，不含协议头、额外拷贝和物理网络的多跳开销：

- p：参与的 rank 数。
- S：**完整逻辑张量的字节数**。AllReduce／ReduceScatter 中是每卡输入大小；AllGather 中是每卡最终输出大小。
- S/p：一片的大小。AllGather 每卡初始只有这一片。
- “每卡通信量”本表统一指**发送字节数**；接收量相同，若问发送＋接收，需再乘 2。
- “全组发送量”是所有 rank 发送量之和。同一次传输在接收端不再重复计数，也不代表唯一数据内容的大小。

| 操作 | 每卡输入 → 输出 | 轮数 | 每卡发送量，接收量也相同 | 全组发送量 |
|---|---|---:|---:|---:|
| AllGather | S/p → S | p−1 | (p−1)S/p | (p−1)S |
| ReduceScatter | S → S/p | p−1 | (p−1)S/p | (p−1)S |
| AllReduce | S → S | 2(p−1) | 2(p−1)S/p | 2(p−1)S |

为什么乘 `(p−1)/p`？一轮发 S/p，一共 p−1 轮；AllReduce 做两个阶段，所以再乘 2。**不是每卡把完整 S 都发给其余每一张卡。**

手算 p=4、S=8 MiB，1 MiB=2²⁰ bytes：

| 操作 | 每卡输入 → 输出 | 每卡发送 | 每卡接收 | 每卡发送＋接收 | 全组发送 |
|---|---|---:|---:|---:|---:|
| AllGather | 2 → 8 MiB | 6 MiB | 6 MiB | 12 MiB | 24 MiB |
| ReduceScatter | 8 → 2 MiB | 6 MiB | 6 MiB | 12 MiB | 24 MiB |
| AllReduce | 8 → 8 MiB | 12 MiB | 12 MiB | 24 MiB | 48 MiB |

如果题目说“每卡初始有 8 MiB，执行 AllGather”，则这里的 S 是 **4×8=32 MiB**，每卡发送 3×8=24 MiB，不能套成 6 MiB。这是最常见的口径错误。

### 通信时间：传输量相同，延迟和拓扑仍影响性能

记 α 为每轮启动时延，β 为每卡有效单向带宽，单位 bytes/s。在可同时发送和接收的简化 Ring 模型下：

$$
\begin{aligned}
t_{\mathrm{AG}}\approx t_{\mathrm{RS}}&\approx(p-1)\alpha+\frac{(p-1)S}{p\beta},\\
t_{\mathrm{AR}}&\approx2(p-1)\alpha+\frac{2(p-1)S}{p\beta}.
\end{aligned}
$$

这是启动＋传输模型，尚未单列归约计算、拥塞、协议等成本；相同通信量也不保证实测时间完全相同。大消息中 Ring 利于带宽利用，小消息中随 p 增长的轮数可能不划算；通信库可选树、分层、多通道等方式，不能把 Ring 当作所有情况的固定实现。

**[P1] nccl-tests 的两个带宽。** `algbw=S/t` 是按逻辑张量大小计算；`busbw` 再乘表中的发送系数：AllReduce 乘 `2(p−1)/p`，AllGather／ReduceScatter 乘 `(p−1)/p`。这里 AllGather 的 S 仍是完整输出、ReduceScatter 的 S 是完整输入。它是方便比较链路利用的归一化指标，不是又产生了一份额外流量。[NVIDIA 性能口径说明](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

### All-to-All 等其他操作，以及调用条件

**All-to-All** 向不同目标发不同分片：rank0 的 `[a,b]` 分别给 rank0/rank1，rank1 的 `[c,d]` 也如此。最终 rank0 得 `[a,c]`，rank1 得 `[b,d]`。MoE 可借此将 token 送往目标专家。AllGather 要让大家获得同一份完整集合，All-to-All 则让各目标获得各自那一份。

Broadcast 将 root 数据复制给大家；Reduce 只让 root 得归约结果。通信要匹配组、调用次序、操作、dtype 和相应 shape/count。可变长 All-to-All 要令一方向某对端的发送量与对端接收量对应，并非所有 rank 都必须发送相同长度。[NCCL 操作定义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## DDP：样本与梯度怎样同步

DP 是数据并行思想，DDP 是 PyTorch 常用实现。各副本对不同数据前向反向，合并梯度，再按相同规则更新。

两卡等样本数、本地 loss 都是均值时，梯度为 2、4，等权同步后是 3。原始 SUM AllReduce 得 6，平均属于训练语义，不能混同：

$$
g=\frac1N\sum_{r=1}^{N}g_r,\qquad \theta\leftarrow\theta-\eta g.
$$

N 为 DP 副本数，η 为学习率，更新式以 SGD 举例。样本量／有效 token 数不同时，本地均值的简单平均未必等于全局均值，需加权。

DDP 不自动将模型状态分片，也不完整替你划分数据。常用 DistributedSampler 分配样本、每 epoch 按约定更新 shuffle；尾部补齐／丢弃会影响重复和遗漏。

反向中梯度可按 bucket（通信批组）组织，一组就绪便通信，与后续反向重叠。过小有启动开销，过大可能启动晚。异步 API 不等于所有通信已隐藏。

<a id="parallel"></a>
## DP、TP、PP、EP、CP 分别切什么

| 方式 | 每卡保存／计算什么 | 为什么通信 |
|---|---|---|
| DP | 完整模型副本，不同样本 | 合并梯度 |
| TP，张量并行 | 同一层矩阵的一部分 | 拼接输出或归约部分和 |
| PP，流水线并行 | 网络的一段层 | 传递阶段间激活与梯度 |
| EP，专家并行 | 一部分专家 | token 分发与结果返回 |
| CP，上下文并行 | 同一长序列的不同 token 段及其激活 | 本地 Q 仍需访问其他段的 K/V；Ring Attention 是一种实现思路 |

TP 例：Y=XW。按 W 输出列切，两卡得到 Y 的不同列，后续可能需拼接；按输入维切，得到同一 Y 的部分和，需归约。因此 TP 不只使用一种 collective。

PP 例：四层放两卡，A 算前两层、B 算后两层。样本必须先过 A 才能到 B；拆 microbatch 后可让两卡处理不同微批，但起止仍可能空闲，称流水线气泡。

CP 例：同一条 8,192-token 序列分到四卡，每卡处理 2,048 个 token；跨段注意力仍要计算。TP 切同一层的矩阵，CP 切输入的序列位置，二者可以组合。不同框架中 sequence parallelism 的范围可能不同，回答时先说明具体切了哪一维、哪些层和激活。

混合并行的 batch 只乘独立数据副本。每 DP 副本微批 3、DP=4、累积 2，则全局 batch=24；TP=2 是协作处理这些样本，不能再乘一次。

## ZeRO／FSDP：模型状态分片

ZeRO-1 分优化器状态；ZeRO-2 再分梯度；ZeRO-3 再分参数。计算某层时需临时取得所需参数，梯度则归约到对应分片。它与按网络层切 PP 不是同一件事。

沿用[模型与注意力](model-attention.md#training)中的显存假设：每参数低精度权重 2、梯度 2、主权重 4、Adam 统计 8 bytes。参数数 P、数据并行组大小 N 时，仅模型状态近似：

$$
\begin{aligned}
\text{DDP}:&\ 16P,\\
\text{ZeRO-1}:&\ P(4+12/N),\\
\text{ZeRO-2}:&\ P(2+14/N),\\
\text{ZeRO-3}:&\ 16P/N.
\end{aligned}
$$

7B、N=4 时分别为 112、49、38.5、28 GB。不是峰值显存：激活、临时聚合和通信工作区另算。FSDP 也组织参数分片与临时聚合，不能认为它和每种 ZeRO 实现完全相同。

将前面的集合通信放回这里：ZeRO-2 可用 ReduceScatter 将梯度交给负责该分片更新的 rank，再同步更新后的权重；ZeRO-3／全分片 FSDP 还会在计算所需模块时 AllGather 参数，并用 ReduceScatter 汇总梯度。前向、反向是否重新聚合和聚合多少次，依重分片、预取与保留策略而定，不能只把“一次 AllGather”当成整个训练 step 的通信量。

<a id="moe"></a>
### MoE：路由、专家计算和负载优化

### 专家是什么，为什么每个 token 不算全部参数

典型 Transformer 稀疏 MoE 把部分 FFN 换为多套参数独立的 FFN，称为**专家**。它不是多个完整聊天模型，也不必预先指定“数学专家”等角色。

设一个 token 表示为 x，共 E 个专家。router 给各专家打分，选 top-k，专家分别计算后加权合并。一种明确的 softmax／重归一约定是：

$$
p_i(x)=\operatorname{softmax}(W_rx)_i,\quad I(x)=\operatorname{TopK}(p(x),k),
\qquad y=\sum_{i\in I(x)}\frac{p_i(x)}{\sum_{j\in I(x)}p_j(x)}E_i(x).
$$

W_r 是路由参数，E_i 是第 i 个专家函数。比如概率 `[.6,.3,.08,.02]`，top-2 选前两个，重归一权重为 2/3、1/3；若输出简化为 3、9，组合为 5。模型之间的打分、门控和归一化不统一，回答时先声明约定。

每 token 只激活 k 个专家，**全部专家权重仍需存放**。更少的激活参数并不消除路由、重排和通信，也不保证比 dense 模型更快。

### dispatch／combine 在做什么

**assignment** 是一次 token→专家分配。100 token、top-2 共 200 个 assignment。dispatch 按专家聚集这些输入，专家批量计算，combine 再按原 token／槽位加权累加；同一个 token 可能收到多份输出，不能简单 concat。

如果都在同一卡，主要是本地分组与重排。采用 EP 时，不同专家分布在不同卡，需将 token 向量发送过去并收回输出，常见 All-to-All；通常不是每一步把全部专家参数送来送去。

### 为什么负载不均：一个例子贯穿后面

8 token、top-1、4 个相同大小的专家，收到 `[6,1,1,0]`，平均 2，热点是平均的 3 倍。若每专家独占同速卡，整轮可能等热点卡；真实矩阵乘不严格随数量线性耗时，不能直接宣称均衡后快 3 倍。

还要区分**逻辑专家负载**与**设备合计负载**。卡 A 放 E0/E1、卡 B 放 E2/E3，工作量为 7 对 1；每卡两个专家不等于一样忙。偏好、训练反馈、输入领域和小 batch 波动都可能造成失衡。

### 训练：均衡辅助损失怎样影响后续路由

在任务 loss 上增加 `λL_balance`。top-1 的常见形式：

$$
f_i=\frac{\text{选中专家 i 的 token 数}}T,\quad
P_i=\frac1T\sum_t p_{t,i},\quad
L_{balance}=E\sum_i f_iP_i.
$$

T 是本次统计范围的有效 token 数；f 是硬分配比例，P 是 **top-k 前完整概率的均值**。固定本轮 f 看，热门专家继续获得很高 P，会付出更大辅助代价。硬 top-k 计数不求导，梯度通过 P 回到 router，改变后续选择；当前 token 不会因为算出 loss 就立即重排。

例：6 个 token 的概率是 `[.7,.1,.1,.1]`，另外两个分别以 .7 偏好 E1/E2，其余均 .1。则 `f=[.75,.125,.125,0]`、`P=[.55,.175,.175,.1]`，L=1.825。每专家恰选两个、各自仍有 .7 偏好时，f/P 都为 .25，L=1。

**目标是批量不过度集中，不是每个 token 的概率都均匀。** L=1 只是参考，不能单独证明硬分配均衡。λ 太大可损害任务质量，统计范围也影响结论；跨 rank 总量均衡未必保证某卡某一步不堵。

本页附录对 top-k 使用 `f=count/(Tk)` 的扩展，系数与 λ 需一起解释；不能把选中后重归一的权重代替完整概率。[Switch 依据](https://arxiv.org/html/2101.03961v3#S2.SS2)

### capacity、丢弃、padding、dropless

常见教学容量：

$$
C=\left\lceil c\frac{Tk}{E}\right\rceil.
$$

c 是 capacity factor，C 是每专家本轮最多接收的 assignment 数，具体取整／最小值按实现。100 token、top-2、4 专家，平均 50；c=1.2 时 C=60。

贯穿例子 `[6,1,1,0]`，T=8、k=1、E=4、c=1.5，C=3，E0 溢出 3。总容量 12 大于需求 8，仍不能把空闲专家的名额直接借给不同参数的 E0。

| 策略 | 做什么 | 代价 |
|---|---|---|
| 丢弃超额分配 | 跳过溢出的专家分支 | 改变该层计算，可能损害质量 |
| 重路由 | 按明确规则送其他专家 | 调度成本与适配损失 |
| 增加容量 | 给每专家更多余量 | 缓冲／padding 和峰值压力，热点工作仍在 |
| dropless | 保留全部已选分配 | 接受动态尺寸与不均匀工作量 |

“丢 token”通常不删除序列里的文字；带残差时，溢出分支可跳过而残差仍在。top-2 少一支后怎样合并以模型为准。

padding 是为规整形状填空位。若截断后 `[3,1,1,0]` 并各填到 3，则 12 槽位只有 5 有效；是否真的计算所有空位看 kernel。dropless 不等于无 padding、无热点或更快，块对齐仍可能填充。MegaBlocks 用块稀疏算子支持不规则工作量，是一种实现路线。

### 执行优化：grouped GEMM、重排和通信

GEMM 就是矩阵乘。专家第一层为 `X_i[M_i,D] @ W_i[D,F]`，M_i 是它收到的数量，各 W_i 不同。若同一卡专家收到 `[2,7,0]`，就有两项不同大小的有效矩阵乘，空专家跳过。

grouped GEMM 将多项矩阵乘放进分组内核调度，减少细碎启动、改善利用率；它保留不同权重，不改变 `[2,7,0]` 的路由，也不能独自消除跨卡热点。

进一步可融合排列／打包、减少拷贝，结合节点内／跨节点链路布局，并用独立 microbatch 或数据块重叠通信计算。输入没到不能算同一项工作；dispatch 2 ms、算 5 ms、返回 2 ms 的串行 9 ms，不会仅因异步就一律变 5 ms。

### 推理侧：让热点逻辑专家有多个物理副本

训练总体均衡，在线代码请求集中或 decode batch 很小，仍可失衡。固定参数推理只加辅助 loss，不更新参数，就不会自动改变路由。

两卡原负载 7 对 1，若 B 能额外放一份 E0 的**相同权重**，可让 E0 的 6 份工作各卡算 3 份，合计变成 4 对 4。token 仍选逻辑 E0，不是换成不同功能的冷门 E3。代价是权重复制、调度与显存占用，可能挤占 KV 空间。

这解释了推理的专家布局／冗余副本与训练 router 均衡的区别。训练若也用副本，还需处理参数和梯度一致性，不能直接套固定权重例子。

#### [P1] 动态偏置：另一种训练均衡思路

选择 top-k 时比较 `适配分数 s_i+b_i`，热点下调 b_i、冷门上调，按负载反馈控制后续选择。DeepSeek-V3 的这一偏置用于**选谁**，输出门控仍来自原始适配分数；报告另保留很小的序列均衡辅助项，不能说没有任何辅助损失。更新速度太大也可能振荡。

### 面试回答与自测

“先看每专家／每卡 assignment、drop／padding、GEMM shape 和通信时间线。训练侧考虑均衡目标，推理侧考虑布局与副本；执行侧再做分组计算、融合和有效重叠，最后检查端到端质量、显存与时延。”

| 问题 | 核对要点 |
|---|---|
| 100 token、top-2 的平均数？ | 200/E 个 assignment，不是 100/E |
| top-k 不可微，辅助项怎么训练？ | f 不求导，完整平均概率 P 保留梯度 |
| dropless 等于均衡吗？ | 不等于，它保留不均匀的真实工作 |
| grouped GEMM 会改路由吗？ | 通常不改，优化已有分配的执行 |
| 换冷门专家和复制热点相同吗？ | 不同，复制同权重逻辑专家可保留函数语义 |

依据：[Switch](https://arxiv.org/abs/2101.03961)、[MegaBlocks](https://arxiv.org/abs/2211.15841)、[CUTLASS 分组](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)、[Megatron MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)、[EPLB](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[DeepSeek-V3](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)。机制可信度高，性能依实际负载。个人旧程序策略组合不作为神经 EP 实战。

### 附录｜单设备前向与辅助损失

前向模板选中概率重新归一；k=1 时门控恒为 1，不能依靠该输出路径得到通常期待的 router 梯度。均衡项单独使用完整概率。以下无 EP、capacity 执行或高性能内核。

<a id="k07"></a>

#### K07｜Top-K MoE

~~~python
import torch
from torch import nn

class TopKMoE(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, top_k=2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("需要 1 <= top_k <= num_experts")
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model),
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        tokens = x.reshape(B * T, D)
        probs = torch.softmax(self.router(tokens), dim=-1)  # [BT,E]
        weights, expert_ids = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # [BT,K]
        output = torch.zeros_like(tokens)

        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(expert_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_input = tokens[token_ids]                 # [M,D]
            expert_output = expert(expert_input)             # [M,D]
            contribution = expert_output * weights[token_ids, slots, None]
            output.index_add_(0, token_ids, contribution)    # 同 token 要累加

        return output.reshape(B, T, D)
~~~

#### 均衡项：f=count/(Tk)，P 保留完整 softmax 的梯度

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

训练时将返回的 balance 乘 λ 加入任务 loss，一起反向。不要对训练用的概率或整个 loss 执行 detach/item；硬计数不求导是刻意的。本地版本没有跨 rank 统计。

## 扩展效率、带宽与重叠

相同任务量单卡 100 ms、两卡 65 ms，加速 `S=100/65≈1.54`，效率 `S/2≈77%`。分摊计算同时增加通信、同步、调度和数据压力，最慢 rank 也会拖住全体。

强扩展固定全局工作量；弱扩展保持每卡工作量近似不变、增加总量，不能将两类 step 时间混作同任务加速。

点对点传输 S 字节的粗略成本是 `α+S/β`；集合通信还要考虑算法的轮数和每轮传多少，不能把完整张量大小直接除以带宽就结束。三种核心操作的推导和四卡算例见前面的集合通信部分。Gb/s 与 GB/s 要换算，1 byte=8 bits。

节点内可能经 PCIe／专用互连，跨节点经网卡与网络。RDMA 可减少数据路径的 CPU 介入和内核拷贝，但仍有注册、控制与同步。TCP/IP 是传输／寻址层，NCCL/HCCL 是通信库，AllReduce 是操作；不能混为一种技术。

重叠要求有独立可计算的工作和正确依赖。例如一组梯度通信时计算后面的梯度；不能计算尚未收到输入的同一份工作。看真实时间线上尚未隐藏的通信，而不是只看异步开关。

<a id="ring"></a>
## Ring Attention：序列分卡与 KV 传递

### 解决什么问题，每张卡持有什么

FlashAttention 避免存完整的注意力分数矩阵，但很长的序列仍有大量 Q/K/V 和其他激活。Ring Attention 将**同一条序列沿 token 维分到多卡**，每张卡保留本段 Q，并让 K/V 块沿环传递。这样，本段 query 仍能关注全部合法的 key，不必让每张卡同时保存全序列 KV。

先看全可见自注意力的一层、一个头：序列长 L，p 卡等分，每段 c=L/p，Qᵣ、Kᵣ、Vᵣ 都为 `[c,d]`。rank r 最终输出 Oᵣ 为 `[c,d]`，对应它自己的 c 个 query。**Qᵣ 留在原卡，远端的 K/V 依次送过来。** 实际多头或 GQA 需按各自头数计算大小。[原论文](https://arxiv.org/html/2310.01889v4#S3)

四卡按 `0→1→2→3→0` 传递时，rank0 的计算如下；其他卡同时做各自的计算：

| 阶段 | rank0 使用的 Q | 本次使用的 K/V | 做什么 |
|---|---|---|---|
| 初始 | Q₀ | K₀、V₀ | 处理本地 query 对本地 key 的注意力 |
| 第 1 次交换后 | Q₀ | K₃、V₃ | 加入第 3 段 key 的贡献 |
| 第 2 次交换后 | Q₀ | K₂、V₂ | 加入第 2 段 key 的贡献 |
| 第 3 次交换后 | Q₀ | K₁、V₁ | 加入第 1 段，得到最终 O₀ |

这里的“加入贡献”需要下面的归一化合并。各段输出仍留在对应卡，后续可继续按序列分片计算，不必每层把全部 O 收集到每张卡。

### 分块结果怎样合并

本卡 Q 与每个收到的 KV 块分别计算局部分数；不能把每块单独 softmax 的输出直接平均。需要维护每个 query 的最大值、指数和及未归一化加权和，并按新最大值重标定旧状态，才能得到全局归一化结果。递推和数值例子在[模型与注意力的 FlashAttention](model-attention.md#flash)；Ring 只是将 KV 块跨设备传过来。

### 通信量、显存与重叠：三个维度分开看

设 S_KV 为**该层全序列 K 和 V 合计的字节数**。标准全可见前向中，本地先算一次，再做 p−1 次必要交换，每次发送／接收一份大小为 S_KV/p 的 KV 块：

$$
V_{\mathrm{send,rank}}=\frac{p-1}{p}S_{KV},\qquad
V_{\mathrm{send,group}}=(p-1)S_{KV}.
$$

接收量同样大。本式只算这一层前向的 KV 流动，假设等长分块、不额外回环；不含反向、QKV 投影或其他并行通信。反向还需重算／访问相关块并汇总 K/V 梯度，不能直接把前向量当整步总量。

自拟手算：batch=1，L=8,192，K/V 各 token 的总宽度均为 1,024，每数 2 bytes，则 `S_KV=2×8192×1024×2=32 MiB`。四卡各持有 8 MiB KV，每卡三轮共发送 24 MiB、接收 24 MiB；全组发送 96 MiB。

| 看哪个维度 | Ring Attention 的作用与边界 |
|---|---|
| 显存 | 逐块消费 KV，保留本地 Q、累计量和有限收发缓冲；无需常驻完整远端 KV。仍有模型状态、其他激活和工作区 |
| 通信量 | 上述规则环的前向发送量，与一次收集全 KV 的 Ring AllGather 同阶；优势不能简单写成“KV 不传了” |
| 通信时间 | 计算当前块的注意力，同时向下一卡发送该块、接收下一块；时间线允许时将传输掩盖在计算中 |
| 计算量 | 仍计算所有合法的 Q–K 关系，全可见注意力总量仍为 O(L²d)，没有变成线性注意力 |

粗略看一头、相同块长 c，QKᵀ 和乘 V 合计约 `4c²d` FLOPs，发送 K/V 为 `2cds` bytes，s 为每个数的字节数。计算吞吐 F、有效带宽 β 时：

$$
t_{\mathrm{compute}}\approx\frac{4c^2d}{F},\qquad
t_{\mathrm{transfer}}\approx\alpha+\frac{2cds}{\beta}.
$$

若块计算够长、设备支持并行传输且依赖安排正确，通信可能大部分被计算掩盖；块太小、网络慢、卡间争用或因果负载不均时，传输仍会暴露在总时延中。不能从异步调用就推出“零通信开销”。这里是教学成本模型，不是具体硬件测速。

### 因果 mask 与几种技术的区别

因果注意力必须使用**全局 token 位置**判断可见性。连续分四段时，第 0 段 query 不能看后面的段；第 3 段可看所有较早段，本段内部仍是下三角 mask。整块全被屏蔽就跳过该块的累计，避免直接算 `−∞−(−∞)`；跳过计算不代表可任意取消其他 rank 仍依赖的转发。

连续分段会使前面设备的合法注意力计算较少、后面较多，产生负载不均。Striped Attention 将序列位置交错分配给各卡，让每卡混合处理较早和较晚位置；分配改变后，仍需维持原始位置和因果关系。上面的标准全可见通信估算不能直接冒充所有因果优化实现的精确流量。[Striped Attention](https://arxiv.org/abs/2311.09431)

| 名称 | 主要处理什么 | 是否在做归约 |
|---|---|---|
| Ring AllReduce | 沿环汇总各卡同一位置的数据，让大家取得完整归约结果 | 是，例如梯度求和 |
| Ring Attention | Q 留本地，KV 沿环传递，并计算、合并本地 query 的注意力 | 不是对各卡 Q/K/V 做 AllReduce；在线归一化在本地进行 |
| FlashAttention | 在设备内分块计算，优化显存读写与中间存储 | 不是多卡通信操作，可作为 Ring Attention 的局部计算基础 |
| PagedAttention | 管理、寻址分页 KV，使缓存按需分配并支持相应注意力读取 | 不是跨卡上下文分片；实际能否组合取决于后端 |

面试可这样概括：**Ring Attention 沿序列切分 QKV，让 Q 留在本卡、KV 块逐次流过，用在线 softmax 得到全局注意力，并尝试让计算与传输重叠。它主要缓解单卡长上下文的存储压力，效果取决于分块、带宽和负载分配。** 先用长序列训练／prefill 理解；单 token decode 的计算／通信比例不同，不能保证同样收益。

## 通信挂起与超时排查

一 rank OOM 退出，其他 rank 可能等它通信而超时。先查看全部 rank 的最早异常和最后完成阶段，再核对组、设备、dtype/count、collective 次序与各 rank 步数。

以单卡、同机两卡、跨机两卡的最小任务逐步缩小范围，再查设备、网络与版本。加 timeout 只改变等待窗口，随意加 barrier 还可能引入不一致。

## 合上资料自测

| 问题 | 核对要点 |
|---|---|
| AllGather 与 AllReduce？ | 原分片拼接与对应位置归约不同 |
| 四卡，完整 S=8 MiB，三种操作每卡发送多少？ | AllGather／ReduceScatter 各 6 MiB，AllReduce 为 12 MiB；接收量另算 |
| 四卡每卡各有 8 MiB 做 AllGather，每卡发多少？ | 完整 S=32 MiB，每卡发送 24 MiB，输出 32 MiB |
| ReduceScatter 能直接“把本地梯度切一片”吗？ | 不能，持有的一片须包含所有 rank 对这些位置的贡献 |
| 为什么 AllReduce 常分成两个阶段？ | ReduceScatter 先获得和的各片，AllGather 再让每卡得到完整和 |
| DDP 能自动使模型状态除以卡数吗？ | 不能，主要是副本与梯度同步 |
| 多两个 TP rank，独立样本加倍吗？ | 同层协作不等于新 DP 副本 |
| ZeRO-3 严格把峰值显存除 N 吗？ | 不能忽略激活和临时聚合 |
| Ring Attention 里什么在动，什么留本地？ | KV 沿环动；本段 Q 和输出累计量留本地 |
| 各 KV 块独立 softmax 后直接求平均行吗？ | 不行，须用最大值／指数和等统计保留全局归一化 |
| Ring Attention 减少了二次算术量吗？ | 没有，分摊计算与存储，并尝试隐藏通信；因果 mask 还要按全局位置处理 |
| 卡在通信先改 timeout？ | 先找全部 rank 的更早异常与通信不匹配 |

依据：[DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)、[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)、[NCCL 测试通信量口径](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)、[ZeRO](https://arxiv.org/abs/1910.02054)、[Megatron](https://arxiv.org/abs/1909.08053)、[Ring Attention](https://arxiv.org/html/2310.01889v4)、[作者实现](https://github.com/lhao499/llm_large_context)、[Striped Attention](https://arxiv.org/abs/2311.09431)。定义与基本机制可信度高；流量、时间和数字例子按正文假设推导，真实性能依拓扑与实现，未运行多卡验证。

