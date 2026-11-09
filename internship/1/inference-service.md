# 推理与服务：KV、调度、P/D 分离及 vLLM

阅读导航：[Prefill、Decode 与 KV](#kv) → [vLLM 请求路径](#vllm) → [调度](#schedule) → [分页与前缀](#paged) → [P/D 分离](#pd) → [量化](#quant)。

按一次请求的时间顺序阅读：先算 prefill/decode 和 KV，再看显存管理、调度、跨实例交接与配置，最后处理量化和排障。vLLM 细节以 0.11.0 官方文档为示例，不同版本须重新核对；这里的命令和示例没有在本机部署运行。

<a id="kv"></a>
## 一次生成：Prefill、Decode 和 KV Cache

prefill 是处理已知提示文本（prompt）的阶段；decode 是此后逐步生成的阶段。两者使用同一模型，但本次处理的位置数和可复用的历史不同。

在普通因果 Transformer 的每一层，输入表示分别投影为 Q、K、V。生成新 token 时，只需用新位置的 Q 匹配历史及当前位置的 K，再对 V 加权。

**过去位置看不到未来，因此在模型、前缀、位置与推理设置相同的条件下，过去各层的 K/V 不因追加 token 而改变。** 将它们缓存，就不必每步重新运行整个历史前缀来生成这些表示。历史 Q 不必留在下一步缓存里，因为下一步要计算的是新位置的输出。

以 prompt `[A,B,C]` 为例，各层都独立维护自己的缓存：

| 本次输入模型的内容 | 本次计算与缓存更新 | 输出用于什么 |
|---|---|---|
| prefill：A、B、C | 计算三个位置的表示，缓存各层 K/V，长度 3 | 最后位置的 logits 用于采样 D |
| decode：D | 计算 D 的 Q/K/V；追加 K/V，长度 4；Q 读取 A—D 的 K/V | 采样 E |
| decode：E | 计算 E 的 Q/K/V；追加 K/V，长度 5；Q 读取 A—E 的 K/V | 采样下一个 token |

注意采样 D 后，要到 D 真正进入下一次前向时，才有 D 的各层 K/V；不要把“已经生成了这个 token”和“已计算它的缓存”混同。

**节省了重算，没有省掉读缓存。** 标准全注意力中，单个新 query 对长度 T 的历史仍需约 O(T×头维度) 的匹配与加权，长上下文下读取 KV 也可能成为瓶颈。

来源：[Transformers 缓存说明](https://huggingface.co/docs/transformers/main/en/kv_cache)。该解释针对普通因果 Transformer；滑窗、潜在注意力等结构需另外算。

## 为什么两阶段资源需求不同

- **Prefill**：prompt 已知，一个请求可同时处理很多位置，矩阵运算规模通常较大；长 prompt 同时带来较高的计算和激活压力。
- **Decode**：普通自回归生成每条请求一次只推进一个新 token，需要多步串行；每步要读取模型权重和相应历史 KV，单请求时可能受内存带宽与调度开销影响。

这是常见趋势，不是“prefill 永远算力受限、decode 永远访存受限”的定理。batch、输入／输出长度、模型架构、并行与硬件都会改变瓶颈；应使用实际时间线与性能计数器检查。

同池服务中，新来的长 prompt 和已有请求的 decode 竞争同一组计算资源。若调度长 prefill 一次运行太久，已有流式请求的两个输出 token 之间可能等待变长；如果一直优先 decode，新请求也可能等待更久。问题既是硬件资源竞争，也是调度与延迟目标的取舍。[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

## KV 占用：从头数与长度算容量

单层、单条请求，K/V 各有约 `T×n_kv×d_h` 个数；再乘层数、请求数和每个数的字节数：

$$
M_{KV}\approx 2LBTn_{kv}d_hs.
$$

L：层数；B：同时保留的请求数；T：每条缓存长度；n_kv：KV 头数；d_h：每头维度；s：每元素字节数。**系数 2 来自 K 和 V，不是前向和反向。**

以 `L=32、B=1、T=4096、n_kv=8、d_h=128、s=2` 为例：

```text
每个 token 的全部层 K/V：2×32×8×128×2 = 131072 bytes = 128 KiB
4096 个 token：128 KiB×4096 = 512 MiB
8 条同长度请求：4 GiB
```

如果 Q 有 32 头，MHA 的 KV 也为 32 头，则同例为 2 GiB；GQA 用 8 个 KV 头为 512 MiB；MQA 用 1 个为 64 MiB。这是 **KV 部分**的比较，不是整机显存或速度的等比例变化。三种结构的 Q/K/V 分组、投影参数量、shape 和手写代码见[模型页的 MHA／MQA／GQA](model-attention.md#gqa)。[GQA 论文](https://arxiv.org/abs/2305.13245)

不同请求长度不同时，将 BT 改成 `ΣT_i`。普通无共享分页缓存的分配槽位还要按每条 `ceil(T_i/块大小)×块大小` 计；有前缀共享时应数实际唯一物理块。权重、workspace、块表、量化 scale 和临时激活另算；并行分片或复制方式也影响单卡占用。

## 服务指标：TTFT、ITL、TPOT、吞吐与 goodput

| 指标 | 定义与关注点 |
|---|---|
| TTFT | 客户端发起请求到收到首个输出 token 的时间；注明是否包括排队、网络、分词 |
| ITL | 相邻输出 token 的间隔；可看每次间隔及 p95/p99 |
| TPOT | 每输出 token 的平均时间；常用首 token 之后的生成时间除以后续输出 token 数，具体口径应明确 |
| 吞吐 | 每秒输出 token、输入 token 或完成请求数；三者不是同一单位 |
| Goodput | 在给定 TTFT、TPOT/ITL 等 SLO 约束下成功满足要求的请求／token 速率；报告定义与阈值 |

例：请求从 t=0 开始，首 token t=100 ms，之后在 t=130、160 ms 到达。按此口径 TTFT=100 ms；两个 ITL 是 30、30 ms；首 token 后平均 TPOT=30 ms。若输出只有一个 token，这种 TPOT 分母为 0，应标为不适用。

吞吐高不一定对用户体验好：即使每秒输出总 token 多，部分请求可能排队久或 token 间隔波动大。比较时固定输入长度、输出长度、到达速率／并发、采样设置和统计口径，再报告分位数。

<a id="vllm"></a>
## vLLM 如何处理请求

```text
客户端请求（文本、采样参数、输出长度等）
   ↓
API 服务接收、校验、处理 chat 模板和 tokenization
   ↓
推理引擎维护请求队列、生成状态和 KV 缓存
   ↓
调度器选择这一轮运行哪些请求及多少 token
   ↓
worker / model runner 准备输入，调用模型和设备计算
   ↓
采样下一个 token，更新请求状态、KV；流式返回或结束
   ↖───────────────── 未结束请求进入后续调度轮次 ─────────────┘
```

这是理解职责的数据流，不是保证每个版本都按同样进程/线程拆分。多卡时可能有多个 worker，模型层内或不同层在设备间分工。prefill、decode 的状态转移及采样时点见本文开头。[v0.11.0 架构概览](https://docs.vllm.ai/en/v0.11.0/design/arch_overview.html)

**tokenization 和 detokenization**：前者把文本转 token ID；后者把生成的 ID 转回文本。CPU 端处理、排队、通信、GPU 模型前向和网络返回都可能影响端到端延迟，不能只计 kernel。

<a id="schedule"></a>
## 连续批处理、调度预算与 chunked prefill

静态 batching 要先凑一批，再整体运行；若长短请求混在一起，等待所有请求结束可能浪费容量。**Continuous batching**（连续批处理）在运行过程中按轮次管理活动请求：某个请求结束后释放或回收资源，新请求可以在后续轮次进入，不必等原批次全部结束。

示意：

| 轮次 | 正在服务的请求 | 当轮变化 |
|---|---|---|
| 1 | A、B | 两者各推进生成 |
| 2 | A、B | A 达到结束条件，输出完成并释放其可回收资源 |
| 3 | B、C | B 继续，等待中的 C 可进入 |

表格简化了 prefill/decode 和调度成本；实际是否能接纳 C 取决于 token 预算、KV 空间、并行配置和调度策略。连续批处理一般提高高并发资源利用机会，也可能因排队和每步批量变大影响单请求时延；请求越多并不总越好。

**调度预算**可限制一轮要处理的 token 数和活动序列数。例如 token 预算 8，3 个 decode token 已占 3，若策略允许，剩余 5 可给长 prefill 的一部分；下轮重新调度。`max_num_batched_tokens` 不是“该服务可接收的总 token 数”，它是具体版本中的每轮调度/批处理预算；`max_num_seqs` 与同时调度的序列数量相关，二者含义不同。

vLLM 0.11.0 的 V1 优化文档描述 chunked prefill 在该版本默认开启、调度优先 decode 后用剩余预算处理 prefill。预算变化会影响长 prompt 完成时间、输出间隔和吞吐；这是**该版本文档中的策略**，其他版本或配置需重新确认。[官方优化指南](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html)

理解 V1 的另一层：其统一调度器按请求已经计算的位置和本轮 token 预算分配工作，底层不必把全部请求强行分成两个互斥的大阶段。教学上仍可把处理 prompt 称 prefill、后续生成称 decode；这种**逻辑阶段的区分**与是否采用**独立 P/D 实例部署**不是同一件事。[v0.11.0 V1 指南](https://docs.vllm.ai/en/v0.11.0/usage/v1_guide.html)

<a id="paged"></a>
## PagedAttention：KV 块管理与前缀共享

请求长度未知、不断增长，且不同请求结束时间不同。如果每条都预留很长的连续空间，会有预留浪费；连续扩容还可能涉及搬移和碎片。

PagedAttention 将请求的 KV 划为固定 token 数的**逻辑块**，用块表映射到 KV 存储池中的**物理块**；注意力内核按块表读取，不要求一个请求的所有 KV 连续存放。它包含配套的存储管理与读取方式，不只是把张量切几段。

### 手算：每块 4 个 token，请求有 10 个 token

这里 token 位置从 0 开始：

| 逻辑块 | 存哪些 token 位置 | 物理块 |
|---|---|---:|
| 0 | 0—3 | 7 |
| 1 | 4—7 | 2 |
| 2 | 8—9，另有两个空位 | 9 |

块表为 `[7,2,9]`。寻找位置 9：`9//4=2`，块内偏移 `9%4=1`，所以读取**物理块 9 的偏移 1**。具体张量还需层、K/V、头和维度等索引。

再追加位置 10、11，可填入尾块；追加位置 12 时才需要下一个块。此时原来的三个块无需为保持整条请求连续而搬家。结束后按引用情况回收块，供其他请求使用。

**减少的是预留与分配浪费，KV 数值本身没有被压缩。** 上例分配 12 个槽位、使用 10 个，仍有 2 个尾块空位。PagedAttention 也不意味着自动将缓存换到 CPU，更不意味着忽略历史 token。

依据：[PagedAttention 论文](https://arxiv.org/abs/2309.06180)、[作者的块映射说明](https://vllm.ai/blog/2023-06-20-vllm)。

### 前缀共享与 copy-on-write

两条请求各有 10 个 token，若前 8 个 token 的完整前缀及执行条件一致，可以共享前两个满块，各自保留最后一个块：

```text
A 块表：[7,2,9]
B 块表：[7,2,5]
共享物理块 7、2；各自的尾块为 9、5。
不共享：6 个块、24 个槽位；共享：4 个块、16 个槽位。
```

这只是自拟容量例子。共享的是相同上下文下算出的 K/V，不是相似文字，也不是相同子串就能共享：同一个词在不同上文中，各层表示可以不同。模型、适配器、位置和相关输入条件也要匹配。

引用计数用于追踪使用者。若一种实现允许共享可继续写入的尾块，分支写入前可先复制，叫 **copy-on-write（写时复制）**。不可把它理解成每次读取都复制。vLLM 的自动前缀缓存文档明确只缓存完整块，具体前缀复用和分支共享策略按版本区分。[官方设计说明](https://docs.vllm.ai/en/latest/design/prefix_caching/)

vLLM 0.11.0 的自动前缀缓存使用父块信息、本块 token 与必要额外信息标识块，文档说明只缓存完整块；具体版本的哈希、隔离与复用策略须再确认。[官方缓存设计](https://docs.vllm.ai/en/v0.11.0/design/prefix_caching.html)

<a id="pd"></a>
## 同池、分块与 P/D 分离：怎么选择

同池由同一资源池执行 prefill/decode；chunked prefill 把长输入拆成多轮，仍可留在同池。P/D 分离则由不同实例／资源池负责两个阶段，需要交接请求状态。它们解决的层次不同，可以组合。

### P/D 分离：不同实例／设备负责不同阶段

```text
                         ┌─→ Prefill 实例/设备：处理 prompt ─┐
用户 → 网关／调度协调 ───┤                                   ├─→ Decode 实例/设备 → 流式输出
                         └──── 选择匹配的 decode 资源 ───────┘
                                        ↑
                     前缀 KV、请求状态及首 token/采样状态经协调和传递
```

不同系统可能由网关、协调器或专门的控制面负责路由，图只是功能分工，不代表 vLLM 的固定进程结构。一次可能的流程：

1. 请求抵达，协调器选择 P 端和可用的 D 端；P 端用同一模型权重处理 prompt，得到各层前缀 KV 和最后 logits。
2. **一种实现**由 P 端采样并返回首 token D，同时将前缀 KV 和继续执行所需的信息交接给 D 端。也有其他首 token 处理方式，不能假设所有实现一样。
3. D 端在接收完成、数据可用后，以 D 为输入进行下一个 decode 步，读取前缀 KV 并追加 D 的 K/V，再生成后续 token E。
4. 请求结束、取消或超时时，按系统的所有权／引用关系释放缓存；传输失败时需处理超时、回退或请求失败。

跨池需传递的不只是纯文本：**D 端继续前向所需的各层 KV、上下文位置与请求／采样等状态都要一致**。不同并行切分、dtype、块布局不同时，还可能需要转换、重分片或重排。不能把各层 KV 当成只要复制 token ID 就能无代价恢复的东西。

## P/D 分离的收益、开销与传输手算

| 可能收益 | 条件和代价 |
|---|---|
| 降低长 prefill 对 decode 的直接干扰，改善尾部 ITL | 仍有网络竞争、队列和调度开销，不能保证所有情况下都改善 |
| 可分别配置 P/D 的资源、并行与批量 | 资源池之间需匹配流量；失衡时一侧排队、另一侧空闲 |
| 针对不同阶段调优 | 两侧可能各加载模型副本，增加权重显存和运维成本 |
| 分开扩容 | KV 必须传输或远程访问；网络、转换和同步可能抵消收益 |

**手算传输下限**：一请求前缀 KV 为 512 MiB，有效传输带宽假设 25 GB/s（十进制，单方向），只算字节／带宽：

```text
512 MiB = 536,870,912 bytes
理想时间 = 536,870,912 / 25,000,000,000 ≈ 0.0215 s = 21.5 ms
```

这不含建立连接、排队、GPU 内存搬运、协议和格式转换。多个请求同时迁移会共享链路。若 P 端先返回首 token，迁移的等待可能主要出现在首 token 到第二个 token 的间隔；若首 token 需要等交接完成，也可能计入 TTFT。指标归属取决于实现和测量起止点。

不同系统、对照基线、负载及指标会导致不同结论：DistServe 的研究针对 TTFT/TPOT 约束下的服务能力；vLLM 0.11.0 实验性分离文档强调独立调优和尾延迟控制，并明确提醒不要据此承诺吞吐一定提高。应同时比较资源数、请求率、长度分布、原始吞吐和 SLO goodput，而不是背一个“分离一定更快”的结论。[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)、[vLLM 0.11.0 分离说明](https://docs.vllm.ai/en/v0.11.0/features/disagg_prefill.html)

P/D 分离可以与分页、前缀复用及连续批处理配合；vLLM 0.11.0 将该功能标记为实验性，接口、connector 和数据格式按部署版本核对。[分离文档](https://docs.vllm.ai/en/v0.11.0/features/disagg_prefill.html)

<a id="quant"></a>
## 量化：权重、激活与 KV 各减少什么

### 为什么做量化，量化谁

量化将较高精度的数映射到较少的可表示值，减少存储和搬运，并可能利用低精度运算。它会引入近似误差。

| 名称 | 具体含义 | 没有自动发生的事 |
|---|---|---|
| W4A16 | 权重约 4 bit、激活约 16 bit | KV 不一定也为 4 bit；并非全部运算都用 INT4 |
| W8A8 | 权重与激活各约 8 bit | 仍须说明 INT8／FP8 等格式和累加精度 |
| FP8 | 8 位浮点格式，仍有指数／尾数 | 与 INT8 的整数表示不同 |
| KV 量化 | 单独降低缓存 K/V 的表示位数 | 权重没有因此变小 |

7B 参数仅算权重，FP16 为 `7×10^9×2=14 GB`，理想 4-bit payload 为 `3.5 GB`。实际还要存 scale／zero-point，可能保留部分层为高精度，另有缓存与工作区。**权重 payload 缩为四分之一，不等于整个推理显存或时延缩为四分之一。**

### scale、zero-point 与手算

常见均匀仿射量化写成：

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s)+z, q_{min},q_{max}\right),\qquad
\hat{x}=s(q-z).
$$

s 是正的缩放步长，z 是让实数 0 对应某个整数位置的 zero-point；q 是低比特编码，x̂ 是反量化的近似值。round 带来舍入误差，clip 会截掉超出范围的值。

本页的**对称 INT8 示例**取 z=0、整数范围 `[-127,127]`，令 s=max|x|/127；这是明确的教学约定，不声称所有 INT8 都不用 −128。

对 `[-1,-0.25,0,0.4,1]`，s=1/127：

```text
整数编码：[-127, -32, 0, 51, 127]
反量化约：[-1, -0.25197, 0, 0.40157, 1]
```

为什么异常值难处理？少数很大的数会拉大 s，让大量小数都挤进少数几个整数格子，丢失区别。逐通道／分组缩放可以缩小各自的表示范围，但增加元数据和内核复杂度。

量化粒度：per-tensor 共用一组参数；per-channel 每通道一组；group-wise 每组若干权重一组。例如线性层 W 的 shape 为 `[输出通道,输入维度]` 时，可按每行一个 scale；具体轴和分组方向必须说清。分组不是 MoE 的专家分组。

基础依据：[量化概念与标定](https://huggingface.co/docs/optimum/concept_guides/quantization)、[TensorRT 量化方案](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)。后端支持的表示范围、粒度和格式需单独核对。

### PTQ、QAT 与校准数据

- **PTQ，训练后量化**：从已有模型出发量化，通常不做完整量化感知重训。某些方法需要代表性校准样本，估计激活范围或层输出误差；简单权重取整也可不读校准输入。
- **QAT，量化感知训练**：训练时模拟量化／反量化误差，使模型适应部署精度；常用近似梯度处理取整，成本高于直接转换已有权重。
- **校准**是在量化准备中收集统计／确定参数，不是仪器标定。输入应覆盖部署时的语言、任务与长度；最终质量测试使用独立数据。

### [P1] 三个方法名，知道解决什么即可

| 方法 | 要点 |
|---|---|
| [GPTQ](https://arxiv.org/abs/2210.17323) | 权重量化，利用近似二阶信息补偿量化误差；不是单纯对所有权重直接 round |
| [AWQ](https://arxiv.org/abs/2306.00978) | 依据激活统计识别重要权重通道，通过等价缩放降低低比特权重误差；名字有 activation-aware，不代表激活也被量化 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 用等价缩放把激活异常值的量化困难转移到权重侧，支持 W8A8；缩放本身等价，后续量化仍有误差 |

无需背论文加速倍数。被问部署选择时，先说模型与硬件支持、质量要求、batch／长度和实测瓶颈。

### 为什么量化后可能不快，KV 又能不能量化

低比特减少权重搬运，小 batch decode 可能因此受益；但解包、反量化、scale 处理及不适合形状的 kernel 也有成本。若通信、KV 读取或调度才是瓶颈，只量化权重未必明显加速。

KV 也可以用更少位数保存。上面的 512 MiB FP16 KV，若改为 8-bit payload，理想为 256 MiB，元数据和缓冲另算。K 的误差会影响注意力匹配，V 的误差会影响加权内容；需要验证长上下文和生成质量，量化／转换开销也可能抵消时延收益。[vLLM KV 量化说明](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)

**分页和量化可以分别考虑：分页减少空位与重复分配，量化减少每个数的位数。** GQA 减少头数，缩短上下文减少长度，它们修改的是不同因素；实际组合要有后端支持和质量验证。

## vLLM 配置与资源取舍

| 参数／概念 | 在官方 0.11.0 文档中的用途 | 调整时的思路 |
|---|---|---|
| `max_model_len` | 模型支持/配置的最大上下文长度 | 上下文更长可能增加缓存和资源压力；与请求长度分布一起看 |
| `max_num_seqs` | 一批中可调度的序列数上限 | 增大可能容纳更大批次，也增加 KV 和时延压力 |
| `max_num_batched_tokens` | 一轮批处理的 token 预算 | 与 chunked prefill 配合；调大/调小均需看 TTFT、ITL、吞吐 |
| `gpu_memory_utilization` | 用于 vLLM 设备内存使用规划的配置比例 | 提高可能为 KV 留更多预算，但要考虑其他进程和运行峰值，不等于永不 OOM |
| `tensor_parallel_size` | 在多个 GPU 之间切分层内权重与计算 | 可减轻单卡权重压力；跨卡通信也可能影响时延 |
| prefix caching | 复用可匹配的完整前缀块 | 相同前缀多时可能受益；要测命中率和缓存占用 |

**不要孤立调参。** 先确认目标是 TTFT、TPOT/ITL、吞吐、显存还是 SLO；固定模型、输入／输出长度分布、并发、采样及设备，再逐项改变并对照。更高并发或更大预算不保证所有指标同时改善。[v0.11.0 优化指南](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html)

KV 空间不足时可能触发 preemption：请求被暂时移出活动执行，之后恢复。vLLM 0.11.0 的 V1 文档写明默认采用 recompute，即恢复时重新计算所需内容；这会消耗额外计算，不能理解为“无成本等一等”。具体模式与支持范围以版本为准。

## vLLM 最小使用示例

以下是 **0.11.0 风格的示意**。需要事先有兼容的 vLLM 环境、设备、可访问的模型文件和使用权限；本仓库没有执行此命令，勿把阅读当作部署经历。

```bash
vllm serve /path/to/local-model \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --tensor-parallel-size 1
```

服务启动后，可用另一个终端检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

这里的 127.0.0.1 指本机，不会自动对外提供服务。实际模型路径、支持的选项、资源和接口要按安装版本核实；支持 OpenAI 风格接口描述的是**请求协议兼容**，不代表调用了 OpenAI 的云模型。离线批处理也可用 `LLM`、`SamplingParams` 等接口，具体参考版本文档。[0.11.0 Quickstart](https://docs.vllm.ai/en/v0.11.0/getting_started/quickstart.html)

## 其他服务优化：投机解码与 CUDA Graph

### Speculative decoding：让一次验证覆盖多个候选

普通自回归生成每一步依赖已生成前缀。投机解码先由较便宜的草稿模型或候选机制提出若干 token，再由目标模型一次验证多个位置；被接受的前缀可一起推进，不接受处需要按对应算法修正后继续。

以贪心为例：草稿提出 A/B/C，目标在当前前缀下认可 A、B，但在 C 位置更倾向 D，则可接受 A/B、在该位置用 D 继续；实际实现细节、额外 token、缓存回滚/更新需遵循算法。精确随机采样版要按目标/草稿分布做接受与修正，不能只说“目标模型不同意就取其 argmax”；满足算法条件时可保持目标采样分布，不保证同一随机种子逐 token 一样。

效果取决于草稿成本、接受长度、目标模型利用率、并发和实现；低接受率或草稿开销大时可能更慢。与 GQA、量化、分页、P/D 分离的优化层次不同，可以组合但需后端支持。[原论文](https://arxiv.org/abs/2211.17192)

### CUDA Graph 与 torch.compile，先讲动机

Python/CPU 每步依次提交许多小 kernel 有启动开销；CUDA Graph 可在满足条件下捕获并重放一组设备操作，减少重复提交开销。对动态 batch/shape、内存地址、采样等有适配约束，图捕获也可能占额外显存；不是把所有 Python 变成 GPU 代码，也不必然让每个 kernel 更快。`torch.compile` 则尝试通过图捕获、优化和生成代码改善计算；两者层次和适用条件不同，实际依版本/后端配置。[PyTorch CUDA Graph](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

## 服务故障与排查

| 场景 | 先核对什么 | 可能方向，需测量验证 |
|---|---|---|
| 服务启动 OOM | 权重、dtype、并行切分、图捕获／工作区、别的进程 | 缩小模型或调整并行、显存规划；不能仅清理缓存 |
| 长 prompt 才 OOM | 实际长度、prefill 激活、KV 块使用、并发 | 限制输入/并发，检查 chunked 与模型支持 |
| TTFT 高 | 请求排队、分词、前缀命中、prefill 时间、网络 | 先确认热点，再调整调度预算／缓存／资源 |
| ITL 尾部高 | 新长 prompt 是否干扰、批量变化、KV 带宽、抢占 | 测 chunked、预算或 P/D 分离，比较同资源 SLO |
| 输出不稳定或错误 | 模型/模板/tokenizer、dtype、采样参数、缓存位置与 mask | 固定输入和采样设置，对比无缓存或其他正确基线 |
| preemption 频繁 | KV 可用块、活动序列、长度分布、抢占方式 | 限制并发/长度、调整可用缓存，测重算开销 |

此外，有缓存与无缓存输出不一致时，检查位置偏移、因果 mask、层与请求的缓存归属、padding 和 eval／精度设置；decode 随长度变慢时，检查历史 KV 读取、batch 与时间线。

## 附录：分页寻址与量化代码

### 块表寻址：只计算地址，不执行 attention

~~~python
def locate_kv(token_position, block_size, block_table):
    if token_position < 0 or block_size <= 0:
        raise ValueError("位置须非负，块大小须为正")
    logical_block, offset = divmod(token_position, block_size)
    if logical_block >= len(block_table):
        raise IndexError("对应逻辑块尚未分配")
    return block_table[logical_block], offset

assert locate_kv(9, 4, [7, 2, 9]) == (9, 1)
assert locate_kv(10, 4, [7, 2, 9]) == (9, 2)
~~~

### 逐输出通道 INT8：只演示量化／反量化

返回完整反量化矩阵方便检查，整段不会体现真实低比特部署显存收益。范围取对称 [-127,127]；INT4 还需真实打包和对应内核。

~~~python
import torch

@torch.no_grad()
def quantize_weight_int8(weight):
    if weight.ndim != 2 or weight.numel() == 0:
        raise ValueError("需要非空 [输出通道,输入维度] 权重")
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("权重必须是有限浮点数")
    work = weight
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    maximum = work.abs().amax(dim=1, keepdim=True)
    step = (maximum / 127).clamp_min(torch.finfo(work.dtype).tiny)
    scale = torch.where(maximum > 0, step, torch.ones_like(maximum))
    codes = (work / scale).round().clamp(-127, 127).to(torch.int8)
    restored = codes.to(work.dtype) * scale
    return codes, scale, restored

w = torch.tensor([[-1., -.25, 0., .4, 1.], [0., 0., 0., 0., 0.]])
q, scale, restored = quantize_weight_int8(w)
assert q[0].tolist() == [-127, -32, 0, 51, 127]
assert torch.equal(q[1], torch.zeros_like(q[1]))
assert torch.allclose(restored[1], w[1])
assert ((restored - w).abs() <= scale / 2 + 1e-6).all()
~~~

无 clipping 且忽略浮点误差时，最近舍入误差不超过半个步长；发生截断时不能沿用此上界。以上算例为自拟，未假定本团队硬件或用户已有推理服务实践。

