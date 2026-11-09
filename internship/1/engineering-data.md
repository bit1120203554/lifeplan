# 工程、训练数据与排障

阅读导航：硬件与计时 → [数据与训练恢复](#data) → [故障与 Linux](#debug) → [容器](#container) → [LRU 代码](#k08)。

沿“硬件和时间线 → 数据进入模型 → 恢复训练 → 异常定位 → 容器运行”学习；最后用真实测量案例和 LRU 手写练习。

## P0：计算在哪里发生

**CPU** 擅长复杂控制流、低延迟任务和通用计算；**GPU** 用大量执行单元并行处理相似运算。Python 代码不会自动在 GPU 上加速：需要对应设备算子，数据搬运与启动算子也有开销。

**算子 operator**是矩阵乘、加法等计算操作，**kernel**是执行设备计算的具体程序。框架需要发起 kernel；很短的小运算太多时，启动／调度成本可能占较大比例。融合将多个操作放进更少的执行过程，还可能减少中间数据读写。

一次模型运行通常经过：磁盘／网络读数据 → CPU 处理 → 主存 → 设备显存 → GPU 算子 → 必要时把结果拷回 CPU。任何一环都可能拖慢整体。

**显存容量**决定放不放得下；**带宽**表示每秒搬多少字节；**计算吞吐**表示每秒多少运算。GB、GB/s、FLOP/s 不是同一种指标。

矩阵乘法 $A_{M\times K}B_{K\times N}$ 大约需要 $2MKN$ 次浮点运算，把乘加算两次。大矩阵往往更容易发挥设备能力，但还受 shape、精度、布局、实现和硬件影响，不能只看 FLOPs 判断速度。

P1：NVIDIA GPU 的 warp 通常包含 32 个线程，以 SIMT 方式执行，分支分歧可能降低有效并行度。寄存器、片上共享内存／缓存和设备显存有不同容量与访问成本；合并访存让相邻线程的访问形成更高效的内存事务。这些术语不能直接套到所有 NPU。

来源：[CUDA 最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[矩阵乘法性能](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

## P0：计算受限、访存受限

**计算受限**主要等运算完成，**访存受限**主要等数据搬运。还有启动、同步、通信和 CPU 等开销，实际不一定属于前两者。

算术强度 $I=\text{FLOPs}/\text{从某级存储搬运的字节数}$，单位 FLOP/byte。简化 roofline 上界：

$$
\text{吞吐}\leq\min(\text{计算峰值},\text{带宽}\times I).
$$

假设计算峰值 10 TFLOP/s、带宽 100 GB/s、强度 10 FLOP/byte，带宽项约为 1 TFLOP/s。计算峰值再高，这个访问模式也难达到 10 TFLOP/s。它是理想上界，不是实测保证；要说明强度对应哪一级存储。

训练大矩阵、长 prompt prefill 常更有计算密度；小 batch decode 常受权重／KV 访存影响。需结合模型、batch 和并行方式测量，不能当固定规律。

**融合算子**可能减少中间张量读写和启动，但也可能增加寄存器压力；**量化**减少位宽，是否提速仍依赖设备与算子支持。

口述：“我先看 CPU、GPU、拷贝与通信的时间线，再判断是计算、访存还是等待，优化主要开销，并检查结果是否改变。”

## P0：profiling 的工作顺序

Profiling 用来定位时间／资源消耗；benchmark 在固定条件下比较性能。测量本身有开销，因此分析和最终计时分开。

1. **定义目标**：训练 step、samples/s，或推理 TTFT、TPOT、吞吐、p95／p99。说明是否包含读数据、排队和网络。
2. **建立基线**：固定代码、硬件、数据、精度、batch／长度、并发和正确性要求，记录版本。
3. **粗定位**：分读取、CPU、主机到设备拷贝、计算、通信、同步。
4. **细定位**：找到主要函数／算子，比较替代方法，一次主要改变一个因素。
5. **复核**：重复测量，检查数值／任务结果，报告中位数、波动与范围。

一个自拟串行 step 30 ms，读取／预处理 18、拷贝 2、计算 8、等待 2：先优化输入链路，不能因为模型运行在 GPU 就先换 attention 内核。异步流水线需进一步看重叠后的关键路径。

读时间线时先找 GPU 空白段，查看是否在等 CPU、数据、同步或其他 rank；再看长 kernel、频繁小 kernel 和拷贝。算子总时长与端到端时间不总能直接相加。

| 工具 | 主要回答什么 | 不能直接推断什么 |
|---|---|---|
| Python cProfile | Python CPU 调用在哪耗时 | GPU 异步工作的实际完成时间 |
| PyTorch Profiler | CPU／设备算子、时间线、shape、可选内存记录 | 开 profiler 的吞吐等于无额外开销的吞吐 |
| Nsight Systems | NVIDIA 的 CPU、CUDA、拷贝、通信全局时间线 | 某个 kernel 的全部硬件瓶颈 |
| Nsight Compute | 单个 NVIDIA kernel 的硬件指标 | 端到端必受该 kernel 限制 |

昇腾等平台换用对应工具，先确认设备。本周先理解读时间线的方法，不需要安装所有工具。

来源：[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。

## P0：GPU 为什么要同步计时

CUDA 运算通常异步提交，CPU 返回可能只表示入队。直接 Python 前后计时可能主要测到提交开销。

以下为单进程、NVIDIA CUDA、隔离环境下的计时示意，step() 是待测函数；没有把示例当作本人已做的实验：

~~~python
import time
import statistics
import torch

for _ in range(10):
    step()
torch.cuda.synchronize()

times = []
for _ in range(30):
    torch.cuda.synchronize()
    start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
print("median_ms =", statistics.median(times) * 1000)
~~~

预热轮数根据稳定程度决定。训练 step 会更新状态，比较时要控制或说明；推理先 eval()，再用 inference_mode() 等关闭梯度。多流、多进程和分布式还需明确边界。

CUDA Event 可记录设备流中的耗时，与包含 CPU 的墙钟时间回答不同问题。同步会改变可重叠的执行，测孤立算子和测真实吞吐要分别设计，不能简单相加每个算子的同步耗时。

来源：[CUDA 异步执行与计时](https://docs.pytorch.org/docs/stable/notes/cuda.html)。

<a id="data"></a>
## P0：大模型数据完整链路

**原始语料 → 解析清理 → 去重／质量筛选 → 划分 → 分词 → 组织序列 → 分片采样 → batch → 模型。**

每一步影响质量和吞吐，不只答“清洗、去重”。

1. **解析清理**：统一编码，处理损坏／空内容／异常格式；明确过滤规则，保留来源和版本。
2. **去重**：哈希发现完全重复，近重复需其他方法；跨训练测试的重复可能让评测虚高。
3. **划分**：按文档／用户／时间等单位，不能随意把同文档的重叠片段分到训练和测试。
4. **Tokenization**：文本到 token ID；记录 tokenizer、特殊 token、聊天模板和版本。token 不等于汉字数。
5. **序列组织**：截断、padding、packing，以及 attention mask、loss mask。
6. **分片采样**：支持流式读取，按 rank 分工，控制 shuffle、seed、混合比例和恢复位置。
7. **检查**：统计过滤前后数量、长度、重复率和来源比例，抽样看内容，查测试污染和标签对齐。

**Padding**：把 [a,b,c] 与 [d] 补成等长 batch，pad 通常不参与损失，需要 mask。长度差大时，很多算力用于无效位置。

**Packing**：多个短样本装成长序列，减少 padding。但样本边界、EOS、position、attention／loss mask 要匹配目标；是否允许跨文档注意力是设计选择，不是把 ID 拼起来就一定正确。

**SFT** 可只对 assistant 回复算损失，是否对提示算损失取决于目标；**预训练**常做 next-token prediction。检查标签移位、特殊 token 与边界，框架内部已 shift 时，不要外部再 shift 一次。

口述：“先保证划分、分词和标签正确，再从长度分布、padding、读取和预处理优化吞吐，质量问题不能靠更多 GPU 补救。”

来源：[Tokenizers 组件](https://huggingface.co/docs/tokenizers/components)、[因果语言建模教程](https://huggingface.co/docs/transformers/tasks/language_modeling)。以上为教学流程，具体规则按项目定。

常见的 BPE 从基本符号出发，按语料中的频繁相邻符号对逐步合并，形成子词词表；使用时按已确定的 tokenizer 切分和编码。不是每个模型都用同一种 BPE，不能随意替换 tokenizer 后仍假定 ID 与 embedding 对应。

### 标签与 mask 的具体例子

EOS 是样本结束标记，PAD 是补齐标记。设 EOS ID=2、PAD ID=0，两条序列为 [10,11,2] 和 [20,2]，补齐后：

```text
input_ids:      [[10,11,2], [20,2,0]]
attention_mask: [[ 1, 1,1], [ 1,1,0]]
labels:         [[10,11,2], [20,2,-100]]
```

这是因果语言模型**内部负责标签移位**的一种常见接口约定；-100 表示此损失位置忽略。模型用 token 10 所在位置的输出预测 token 11、用 token 11 预测 EOS；这里 10/11 是 token ID，不是位置编号。第二条的填充目标不计损失。若模型不内移位，就需显式准备 next-token 目标，不能混用两种约定。

attention_mask 的 1 表示有效输入，不是完整的二维因果关系；因果约束仍需模型实现。SFT 的提示位置可再按目标屏蔽 loss。Packing 若要求样本独立，要处理跨样本的注意力与目标边界；仅插入 EOS 不自动形成隔离。

## P0：PyTorch 怎样把数据交给模型

- **Dataset**：定义样本，map-style 常按索引取，iterable-style 迭代生成。
- **Sampler**：决定索引和顺序；分布式需注意 rank 分工与每轮 shuffle。
- **DataLoader**：组织读取、worker 和 batch。
- **collate_fn**：多个样本组成 batch，如动态 padding、生成 mask。

长度 [100,110,1000,1010] 随机配对可能浪费 padding；按长度分桶可减少浪费，但影响顺序，要兼顾随机性。

num_workers 是读取 worker 数，CPU、解压、存储和内存有限，不是越多越快；多进程还可能复制部分数据状态。

pin_memory 使用固定页主存，常用于改善主机到 CUDA 传输条件，不是把数据放进显存。non_blocking=True 可在满足条件时允许异步拷贝；真正与计算重叠仍依赖流、依赖与硬件，要看时间线。

卡等数据时，测独立数据迭代，再用固定设备输入对照。确认不是计算／同步瓶颈后，再调 worker、prefetch、缓存和存储格式。

来源：[Dataset／DataLoader](https://docs.pytorch.org/docs/stable/data.html)、[性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。

## P0：断点恢复不只是存权重

只存参数可满足部分推理用途；要继续训练接近原轨迹，通常还需要：

- 优化器状态、学习率 scheduler。
- AMP GradScaler（使用时）、step／epoch、梯度累积进度的处理约定。
- Python／NumPy／框架和设备随机数状态。
- sampler、shuffle、流式读取位置，数据／tokenizer／代码／环境版本。
- 分布式各 rank 的分片、一致性和保存完成标记。

Adam 动量丢失，即使权重相同，下一步也可能不同。数据位置丢失，可能重复或跳过样本。

考虑保存中断和部分写入，可先写临时位置，完成后按存储系统支持的方式发布完整版本；不能跨文件系统一概承诺原子性。

验证：短训练保存，恢复后跑几步，比数据位置、loss 和参数；有非确定算子时说明容差，不承诺任何环境都逐位一致。

来源：[保存与恢复](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[可复现性](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

<a id="debug"></a>
## P0：四类常见故障

### GPU 利用率低

利用率是采样窗口中的活动指标，不等于峰值算力利用率；不能仅凭一张截图判断。

- 先确认输入规模、step 时间、CPU／I/O 和显存，再分阶段测量。
- 数据等待：查读取、解压、预处理、worker、存储／网络；用固定设备输入对照。
- kernel 很碎：查小 batch、小算子、频繁 CPU 发起与 Python 控制。
- 往返多：查循环中的 .item()、.cpu()、打印张量等同步／拷贝。
- 多卡等待：比较各 rank 数据耗时、负载和通信；卡忙也可能在通信。

不要直接增加 batch／worker／卡数。worker 过多可能争 CPU 和内存，反而变慢。

### 显存高但设备不忙

模型、优化器、缓存可能占显存却没持续计算。看归属和等待位置；显存占用不等于利用率。allocated 是框架活跃张量占用，reserved 是包含已分配部分的分配器保留池，不能相加；设备总占用还含其他进程／库。

### 一卡能跑，多卡变慢

先确认固定的是全局 batch 还是每卡 batch、处理数据量是否一致、计时是否同步。再查通信、rank 不均衡、数据、拓扑和重叠。小任务可能无法摊薄通信成本。

### loss 变 NaN／不下降

找第一个异常 step／张量；检查输入、标签、mask、空 batch、损失和梯度是否有限；再看学习率、AMP、梯度爆炸、初始化和参数是否更新。小样本过拟合检查能帮助定位。一次改十个超参会失去归因。

## P0：Linux 命令要说明“看什么”

以下只读示例不保证每台机器都安装。重点是读指标，不是报命令名。

| 现象 | 可先查看 | 看什么 |
|---|---|---|
| 进程／CPU | ps -ef、top、pidstat | 状态、线程、CPU、上下文切换 |
| 主机内存 | free -h、vmstat 1 | available、swap、换页 |
| 磁盘／I/O | df -h、df -i、iostat -xz 1 | 容量、inode、延迟和队列 |
| 端口／连接 | ss -lntp、ss -tp | 监听地址和连接状态，部分信息需权限 |
| NVIDIA 设备 | nvidia-smi | 进程、显存、利用率与驱动可见性 |
| 被杀死 | 容器状态、日志，有权限时查内核日志 | 主机／cgroup OOM 与 CUDA OOM 的区别 |

**RSS** 是驻留物理内存；**VSZ** 是虚拟地址空间，大不等于实际物理内存耗尽。共享页也使多个 RSS 相加不等于唯一物理占用。Linux 会用内存做缓存，available 通常比仅看 free 有参考价值。

**load average** 包含可运行任务和不可中断等待等，不是 CPU 使用率百分比。CPU 空而负载高也可能与 I/O 等待有关。

排查表达：“先确认是 Python 异常、CUDA OOM、容器 OOMKilled 还是挂起，记录时间与日志，再对照主机、设备和应用阶段指标。”

来源：[proc 手册](https://man7.org/linux/man-pages/man5/proc.5.html)、[Linux 内核指标说明](https://www.kernel.org/doc/html/latest/filesystems/proc.html)。

## P0／P1：进程、线程、Python 和 C++

进程有自己的地址空间；同进程线程共享地址空间，但各有栈与调度状态。共享数据要防竞争；多进程隔离强，但创建、序列化和通信有成本。

通常启用 GIL 的 CPython 中，同一时刻只有一个线程执行 Python 字节码，纯 Python CPU 计算未必靠线程加速。I/O 和释放 GIL 的 NumPy／PyTorch 原生代码不同；也存在禁用 GIL 的构建，不能推广到所有 Python。

P1 的 C++ 基础要能解释一个具体错误：

| 概念 | 小例子与含义 |
|---|---|
| RAII | 文件流对象离开作用域时自动关闭文件，异常路径也能清理资源 |
| unique_ptr／shared_ptr | 一个管理者与多个共享所有者；shared_ptr 环互持可能不释放，用 weak_ptr 表达非拥有关系 |
| 悬空／越界 | 返回局部变量地址，或访问 vector[size()]，都可能触发未定义行为 |
| reserve／resize | reserve(100) 预留容量但不产生 100 个元素；resize(100) 改元素数 |
| 迭代器失效 | vector 扩容搬移元素，旧指针／引用／迭代器可能失效 |

局部 vector 对象在当前作用域内管理资源，但元素通常另行分配，不能认为局部变量的数据全在栈上。

编码先用最熟语言，并确认允许范围；简历写了 C++，所有权、容器和复杂度仍需准备。不必三天另学 Rust。

来源：[Python 线程与 GIL](https://docs.python.org/3/library/threading.html#gil-and-performance-considerations)、[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)。

P1：**NUMA** 指非一致内存访问架构，不同 CPU 到不同内存区域的访问成本可能不同。CPU／内存放置、设备连接拓扑可能影响数据和通信路径；排查时核对实际拓扑，再测绑定前后的效果，不盲目固定到某个 CPU。来源：[Linux NUMA 内存策略](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html)。

<a id="container"></a>
## P0：镜像、容器、虚拟机

**镜像 image** 是打包的文件系统和运行配置，作为启动模板；**容器 container** 是按镜像启动的隔离进程环境。容器通常共享宿主内核，虚拟机则有自己的客体操作系统内核。

镜像像运行环境模板，容器像按模板启动的一次运行，不是把整台物理机器装进去了。

Linux 容器常通过 **namespace** 隔离进程、网络、挂载等视图，通过 **cgroup** 管理和限制 CPU、内存等资源。隔离不等于独占硬件，也不自动解决全部安全／性能问题。

复现除了 Dockerfile，还需基础镜像／依赖版本、代码、数据、参数和兼容驱动。仅用“最新镜像”不能保证复现。

GPU 容器仍依赖主机驱动、对应运行时与设备暴露机制。镜像内装了 CUDA 用户态库，不代表主机驱动兼容；其他加速器也有自己的要求。

P1：Dockerfile 的 FROM 选基础镜像，COPY 放文件，RUN 构建时执行安装等操作，CMD 给默认启动命令。数据和 checkpoint 常外部挂载，避免容器删除后丢结果；密码和令牌不放镜像。

来源：[Docker 容器](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)、[镜像](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-an-image/)。

## P0：Kubernetes 对象管什么

Kubernetes（K8s）按声明的目标状态管理容器工作负载，解决调度、生命周期和资源组织；不会自动把任意模型的分布式训练配置正确。

| 对象／组件 | 作用 | 例子 |
|---|---|---|
| Cluster | 集群整体 | 多机和管理组件 |
| Node | 工作节点，物理机或虚拟机 | 八卡服务器 |
| Pod | 可调度基本单位，含一个或多个紧密协作容器 | 训练进程＋日志辅助容器 |
| Scheduler | 为未分配 Pod 选择符合条件的 Node | 找足够资源的节点 |
| Kubelet | Node 上管理 Pod 运行的代理 | 启动和检查容器 |
| Job | 执行到完成的任务 | 数据处理／训练 |
| Deployment | 维持副本、支持更新 | 常驻推理服务 |
| Service | 提供稳定访问入口／服务发现 | 推理 API 内部访问 |
| Volume／PVC | 提供存储／申请持久存储 | 数据和 checkpoint |

Pod 不等于 Node，多个 Pod 可同 Node。普通 Job 也不自动提供多机训练全部协作机制，实际常需额外配置或训练作业控制器。

K8s 的 namespace 是资源分组／作用域，kubectl 的 `-n` 用来选择它；它与上一节 Linux 容器的 namespace 隔离机制不是同一个对象。

**requests** 是调度资源需求，**limits** 是资源约束。CPU 超限通常被节流，内存超限可能 OOM kill，依资源类型和环境而定。声明的资源不等于实时利用率。

**Running** 不等于 **Ready**：进程在，不代表模型加载完或能接请求。readiness probe 判断是否适合接流量，liveness probe 判断是否需重启，startup probe 可给慢启动留空间。

来源：[Pods](https://kubernetes.io/docs/concepts/workloads/pods/)、[资源管理](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)、[探针](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)。

### 从提交任务到真正运行

提交 Job 后，控制器创建 Pod；Scheduler 根据资源与节点条件选 Node；Node 上的 Kubelet 协同容器运行时拉镜像并启动进程；进程再加载数据／模型。训练代码自己还需配置 rank、地址、设备与通信组。K8s 把进程安排到机器上，不自动替任意训练程序完成多机协作。

例如 requests CPU=2、内存=8 GiB，limits CPU=4、内存=12 GiB：调度看资源请求，CPU 达限制可能节流，内存超限制可能被杀；这不是实时利用率。GPU 等扩展资源有对应申请规则，需要设备插件与匹配的驱动／运行时。

## P0：Pod 不启动，按状态查

先看状态、事件、退出原因和日志，再决定措施，不能只答“重启”。

| 现象 | 优先检查 |
|---|---|
| Pending／Unschedulable | 缺 CPU／内存／GPU？节点条件、污点、PVC？以调度事件为准 |
| ImagePullBackOff | 镜像名／版本、认证、网络、拉取限流 |
| CrashLoopBackOff | 命令、配置、依赖、权限；当前和上一次日志／退出原因 |
| OOMKilled | 容器主机内存限制、worker／缓存峰值；区别 CUDA OOM |
| Running 但不可访问 | readiness、模型加载、地址／端口、Service 选择器和网络 |
| 容器看不到 GPU | 主机驱动、运行时、设备插件、资源声明，先确认平台 |

只读命令示意，尖括号替换成实际对象，先确认命名空间：

~~~bash
kubectl get pods -n <namespace>
kubectl describe pod <pod> -n <namespace>
kubectl logs <pod> -n <namespace> -c <container>
kubectl logs <pod> -n <namespace> -c <container> --previous
kubectl get events -n <namespace> --sort-by=.metadata.creationTimestamp
~~~

GPU Pending 先判断资源不足、设备插件未注册，还是节点条件不匹配，不直接随意降低 requests。无实践时说：“我理解按状态查看事件和日志的过程，尚未独立维护生产集群。”

P1：NVIDIA 集群常通过设备插件暴露 nvidia.com/gpu，仍需兼容驱动与运行时。扩展 GPU 资源遵循对应声明规则，不当普通 CPU 可超卖资源；不能推断本岗一定用 NVIDIA。

来源：[调试 Pod](https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/)、[GPU 调度](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)。

## P0：把本人已有优化讲扎实

依据：[2026-10-02 记录](/home/lys/task/projects/project4a/data/results/runtime_validation_cleanup_20261002_report.md)。

| 问题 | 真实回答骨架 |
|---|---|
| 优化什么 | 固定策略库的单状态策略评估，属于程序执行开销 |
| 怎么改 | 去重复验证、复用数组、少数组／字典转换、减少重复类型和有限值检查 |
| 怎么测 | 同 checkpoint、2,026 个状态、7 轮交替配对微基准，先检查输出一致 |
| 结果 | 中位耗时 468.45 → 322.67 μs，下降约 31.12% |
| 限制 | 不能推出端到端搜索／LLM 推理／GPU 同比例加速 |

$(468.45-322.67)/468.45\approx31.12\%$。该片段约 1.45 倍快，但没有证明整体也如此。

**Amdahl 定律**：若一个片段占原总时间 40%，变为 2 倍快，整体只有 $1/(0.6+0.4/2)=1.25$ 倍。先知道热点占比；异步场景还要分析关键路径。

真实反例：[多进程记录](/home/lys/task/projects/project4a/data/results/head_process_clip_20261003_report.md)。该次配置本地约 312 μs，多进程约 2,337 μs，通信调度成本压过收益，不能据此认为所有多进程都无效。

口述：“我没有假设进程越多越快。测量发现任务短，序列化和调度代价占主导，所以撤回方案，再减少原热路径的重复工作。”

## 合上资料自测

| 问题 | 核对要点 |
|---|---|
| GPU 显存高等于算力利用率高吗？ | 占用是容量，活动与运算效率要另看 |
| 为什么 CUDA 计时前后同步？ | CPU 提交异步工作，返回不等于完成 |
| roofline 的两个限制？ | 计算峰值与带宽×算术强度的较小者 |
| 40% 热点快两倍，整体多少？ | 1/(0.6+0.4/2)=1.25 倍 |
| CPU 内存耗尽与 CUDA OOM？ | 主机／容器内存与设备内存是不同资源 |
| 第一条优化措施怎么选？ | 先建立基线和定位主要耗时，再改一个因素复核 |

## 附录｜手写 LRU

LRU 是访问淘汰策略；不要用它随意删除模型当前计算必需的历史 KV。

<a id="k08"></a>

### 手写 K08｜LRU：哈希表＋双向链表，完整手写

**需求**：get 命中后变最近使用，未命中返回 -1；put 新增／更新也变最近使用；超容量删最久未使用。容量为非负整数，key／value 为整数。

**先画顺序**：head ⇄ 最旧 ⇄ …… ⇄ 最新 ⇄ tail。head、tail 是不存真实数据的哨兵，消除空链表和首尾特殊分支。哈希表负责 O(1) 找节点，双向链表负责 O(1) 摘下与接尾。

~~~python
class Node:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.nodes = {}
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _append_latest(self, node):
        previous = self.tail.prev
        previous.next = node
        node.prev = previous
        node.next = self.tail
        self.tail.prev = node

    def get(self, key):
        node = self.nodes.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._append_latest(node)
        return node.value

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self.nodes:
            node = self.nodes[key]
            node.value = value
            self._remove(node)
        else:
            node = Node(key, value)
            self.nodes[key] = node
        self._append_latest(node)

        if len(self.nodes) > self.capacity:
            oldest = self.head.next
            self._remove(oldest)
            del self.nodes[oldest.key]
~~~

**背的顺序**：定义节点和哨兵 → 写摘除 → 写接尾 → get 命中就移动 → put 更新或新增 → 超限删最旧，且字典同步删除。

**最小走查**：

~~~python
cache = LRUCache(2)
cache.put(1, 10)
cache.put(2, 20)       # 旧 → 新：1,2
assert cache.get(1) == 10  # 2,1
cache.put(3, 30)       # 淘汰 2，变 1,3
assert cache.get(2) == -1
cache.put(1, 11)      # 更新不增加节点数，变 3,1
assert cache.get(1) == 11
~~~

**边界**：容量 0、容量 1、更新已有 key、连续 miss、value 为 -1。接口约定允许 -1 时，仅看返回值不能区分真实 -1 与 miss；题目如此约定，其他接口可改为返回命中标记。

**复杂度**：哈希操作平均 O(1)，链表摘／接 O(1)，空间 O(capacity)。若用普通列表查找或删除任意元素，会退化到 O(n)。

**追问**：不是线程安全实现，竞争会破坏字典／链表一致性；生产缓存还可能按字节限额、有 TTL 和并发规则。LRU 的访问链与 Transformer KV cache 不是同一个概念，不能直接用 LRU 随便删模型当前必需的历史 token。

若允许使用标准库，也可用 OrderedDict 实现；手写要求下优先准备本页双向链表版。

