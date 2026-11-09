# 来源、团队线索与可信度

核查日期：2026-10-24。面试为 10/27 周二。这里区分本人信息、官方岗位、个人面经和教学材料，避免把相邻岗位要求直接套成本次实习门槛。

## 1. 已知与尚未知

| 信息 | 来源 | 可信度／限制 |
|---|---|---|
| 岗位称谓、首轮邀请和本次 JD | 本人消息，[本地 JD](jd.md) | 高，尚无独立官网同一实习岗位编号 |
| 一小时以内、可能手写代码 | 本人对通知／常见流程的判断 | 是准备假设，非核实的固定流程 |
| 投递当前 rsm.docx | 本人本次确认，[原文件](../rsm.docx) | 高；原 Word 未在本次改写 |
| 北京、每周最多 4—5 天、可六个月 | 本人此前确认 | 是个人约束，尚未证实本岗位地点与安排符合 |
| 团队偏训练还是推理、设备、技术栈、mentor | 尚无团队回复 | 未知；不把平台宣传当团队配置 |

## 2. 官方招聘能告诉我们什么

**A. 词元星火相邻的星火 X 全职课题。** 官方招聘列表可检索 J14504“面向大模型高效训推的算子研发与极致优化”、J14505“面向超大尺寸 MoE 的大 EP 推理工程优化”。后者涉及跨节点 All-to-All、负载均衡和并行调度，前者涉及 GPU／NPU 算子与性能分析。[官方岗位列表](https://iflytek.zhiye.com/7/jobs?shareId=a7633258-73ca-4a77-9e3a-f5a1034ba4c3&shareSource=2)

这支持把**通信、MoE、profiling**列为准备重点，不能证明本次实习就是上述课题组，也不能要求自己三天具备全职专家水平。这是基于相邻岗位的准备推断，可信度中等。

**B. 另一条校招 AI Infra 岗位 J14428。** 2026-09-14 发布，页面写合肥，内容覆盖智能体运行平台、推理服务和调度，并要求系统与性能分析基础。[官方岗位页](https://iflytek.zhiye.com/campus/detail?jobAdId=fb1ba98a-054c-4911-b7e1-8121cf0db41a)

该条是另一校招岗位，不等于本次实习；其地点不能用于确认本岗，仍需问北京办公是否可行。

**C. 公司平台层面有异构算力。** 星火智算平台公开列出昇腾、寒武纪、海光、英伟达等资源管理／适配能力。[官方平台](https://spark-aicc.iflytek.com/aicc-portal/)

因此材料保留通用通信语义，以 CUDA／NCCL 为例并注明 HCCL；不能由“讯飞”断定本团队一定是 NVIDIA 或一定是昇腾。这里没有核实到该实习实际集群、版本或生产规模。

核查方式说明：招聘与平台网页依赖 JavaScript，直接文本读取有限，上述招聘字段与平台介绍通过**官网域名的公开搜索索引**交叉核对。官方文本来源可靠，页面当前招满情况、索引时效和与本次实习的对应关系仍需确认。

## 3. 面试流程与公开面经

没有找到足够可靠、能对应“词元星火 AI Infra 系统工程师实习首轮”的公开流程或真题。

| 公开候选人自述 | 可参考什么 | 不能推出什么 |
|---|---|---|
| [科大讯飞 26 校招多模态大模型面经](https://www.nowcoder.com/feed/main/detail/787a820436114e47ae09446c1bd4da19)，页面日期 2025-09-14 | 项目深挖、注意力、部署、C++ 等问法 | 岗位／地点／年份不同，文中没有编码不代表本岗没有 |
| [阿里 AI Infra 二面](https://api-cdn.nowcoder.com/feed/main/detail/328e2dada5f246889e354707926f6da0?sourceSSR=users)，页面显示 09-18，年份未核实 | 通信、推理同步、GEMM、最小复现等相邻方向 | 另一公司、另一轮次，不能当讯飞题目或出现频率 |

两者是一手候选人自述，真实性未经企业确认，适合作为话题线索，可信度中低。技术结论另用官方文档和论文核对。三套练习全部为本次自拟，没有冒充真题。

本次 45—60 分钟的练习流程按本人预计时长和一般技术面组织：自我—项目—基础—编码／诊断—反问。这是准备安排，真实流程以面试官为准。

## 4. Skill 的实际使用范围

此前三个 Skill 的查找记录见 [10/21 清单](../knowledge/skills-2026-10-21.md)，本次未安装其工具或读取登录态。

- **[CS 技术面试官 SKILL.md](https://github.com/CHfaithwy/cs-tech-interviewer-skill/blob/main/SKILL.md)**：读取说明、选题策略和语义评价参考。借鉴按 JD 与简历分配权重，追问个人职责、指标和取舍，分层给提示并按意思判断。此次是离线资料编写，没有创建或运行其 live session，也没有声称已执行该 Skill 的脚本选题流程。
- **[Interview Collector](https://github.com/swf2020/interview-collector)**：保留为后续面经检索工具，本次使用公开网页检索，没有执行其采集／多 Agent 流程。
- **[Agent Reach](https://github.com/Panniantong/Agent-Reach)**：是跨平台访问工具，非面试知识标准。本次现有网页工具足够，未额外安装，也未尝试访问需要登录的内容。

模拟题及 0—3 分自评表为本次教学用途，没有基于候选人尚未发生的回答生成分数或录用判断。

## 5. 六个学习单元的一手依据

2026-10-26 重排。正文给出基础解释，以下用于核对与深入。数学与机制可信度高；官网 stable/main/latest 会变动，代码验证环境为 PyTorch 2.13.0+cpu，不能推定本团队使用网页最新版。

| 单元 | 一手来源 |
|---|---|
| [01 模型／训练](../../legacy/internship/1/archive/before-consolidation-2026-11-03/01-model-pytorch.md) | [Transformer](https://arxiv.org/abs/1706.03762)、[MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245)、[Pre/Post 分析](https://arxiv.org/abs/2002.04745)、[RoPE](https://arxiv.org/abs/2104.09864)、[PyTorch 训练](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、[LoRA](https://arxiv.org/abs/2106.09685) |
| [02 推理／显存](../../legacy/internship/1/archive/before-consolidation-2026-11-03/02-inference-memory.md) | [FlashAttention](https://arxiv.org/pdf/2205.14135)、[在线 softmax](https://arxiv.org/abs/1805.02867)、[GQA](https://arxiv.org/abs/2305.13245)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[vLLM 前缀缓存](https://docs.vllm.ai/en/latest/design/prefix_caching/)、[量化基础](https://huggingface.co/docs/optimum/concept_guides/quantization)、[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[SmoothQuant](https://arxiv.org/abs/2211.10438) |
| [03 分布式](../../legacy/internship/1/archive/before-consolidation-2026-11-03/03-distributed-communication.md) | [DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)、[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)、[NCCL 测试流量／带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)、[ZeRO](https://arxiv.org/abs/1910.02054)、[Megatron](https://arxiv.org/abs/1909.08053)、[Ring Attention](https://arxiv.org/html/2310.01889v4)、[作者实现](https://github.com/lhao499/llm_large_context)、[Striped Attention](https://arxiv.org/abs/2311.09431) |
| [04 MoE](../../legacy/internship/1/archive/before-consolidation-2026-11-03/04-moe.md) | [Switch](https://arxiv.org/abs/2101.03961)、[MegaBlocks](https://arxiv.org/abs/2211.15841)、[CUTLASS 分组调度](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[vLLM EP/EPLB](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) |
| [05 性能／排障](../../legacy/internship/1/archive/before-consolidation-2026-11-03/05-linux-profiling.md) | [CUDA 最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[CUDA 计时／内存](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[Linux proc](https://www.kernel.org/doc/html/latest/filesystems/proc.html)、[Python GIL](https://docs.python.org/3/library/threading.html#gil-and-performance-considerations) |
| [06 数据／容器](../../legacy/internship/1/archive/before-consolidation-2026-11-03/06-container-data.md) | [DataLoader](https://docs.pytorch.org/docs/stable/data.html)、[语言建模数据](https://huggingface.co/docs/transformers/tasks/language_modeling)、[Docker](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)、[K8s 资源](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)、[Pod 排障](https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/)、[保存恢复](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) |

所有微型数字例子为教学自拟，不套用论文加速倍数，也不作为本人新增经验。模板只验证语义和数值，未执行 GPU kernel、分布式训练或生产部署。

## 6. 个人项目数字与当前阶段

- ICCV 方法与结果：[公开论文](https://arxiv.org/html/2506.21880v1)。Calibration 的 IHRUT-7 为 39.61 dB，MAUN-7 为 38.15 dB；不泛化成所有数据／指标提升百分比。
- 前期 project4a 已停止，机制与实现分别见[原设计](/home/lys/task/projects/project4a/docs/Id1.md)和[实现说明](/home/lys/task/projects/project4a/docs/prototype_implementation.md)。不是当前 Id1，也不冒充神经 MoE。
- 运行分析：[10/2 单状态配对测量](/home/lys/task/projects/project4a/data/results/runtime_validation_cleanup_20261002_report.md)、[10/3 多进程反例](/home/lys/task/projects/project4a/data/results/head_process_clip_20261003_report.md)、[150 局冻结对照](/home/lys/task/projects/project4a/data/results/sokoban_evo150_20261009/report.md)。旧数据不移作新方法结果。
- 当前研究：[Id1/I001](/home/lys/task/ideation/ideas/entries/I001-active-self-evolution-sparse-planning.md)为局部原型；[Id3/I003](/home/lys/task/ideation/ideas/entries/I003-neural-programmatic-co-learning.md)为方法设计。阶段依据本人说明与本地条目，详细核对见[当前研究说明](../../legacy/internship/1/current-research-2026-10-25.md)。

## 7. 本轮校验

2026-10-26 完成以下核对：

- 从未掌握概念的视角逐页复查术语首次解释、符号含义、单位、前后依赖和问答闭合；补正 token ID／位置、allocated／reserved、采样／缓存时点等易混处。
- 重排前的 16 个函数／类定义在新材料中完整保留，语法结构逐一一致；19 个 CPU 教学代码块执行通过。
- 注意力因果性、分块缓存等价、Pre/Post 两种 block／小模型反向、在线 softmax 对完整 softmax、MoE 恒等组合与均衡项梯度、LRU 参考序列通过。
- KV 显存、注意力中间矩阵、在线合并、ZeRO 状态、容量与标签移位算例复算通过；190 个活动本地链接／锚点通过。归档保留当时结构，不作为当前入口。
- self、原 Word、两份个人研究说明校验值不变。通用训练片段和 CUDA 计时只查语法与前提，未执行 GPU、多机或生产系统。

正文已整理完成，学习、自测和面试本身仍待本人实际进行。

2026-10-27 进一步补充通信：操作语义按 NCCL 文档核对，流量采用等长理想 Ring 并明确完整张量、每卡发送／接收和全组计数；Ring Attention 按原论文 §3 及作者实现核对，因果负载问题参考 Striped Attention。CPU 小数组检查了 2／4／8 rank 的 Ring 分片归约、收集结果与发送量；按环顺序合并普通／因果注意力，与完整 softmax 结果一致。通信量表与在线合并算例为教学推导，不作为多卡性能结果；未运行实际分布式通信，self 与原 Word 校验值保持不变。

2026-11-01 为 11-02 二面更新入口、MHA／MQA／GQA 内容及两套模拟与逐题答案。GQA 的共享头与权重适配依据上列两篇论文；CPU 检查了头数为 8/8、8/2、8/1 的教学代码，并与显式展开 KV 的参考计算比较、检查反向和非等长因果 mask。这不是 GPU 内核或实际模型性能测试。二面侧重和面试流程未获确认，题目据 JD 自拟。

## 8. 2026-11-03：面试后知识库扩充

本人反馈两次面试均涉及 vLLM，近期面试涉及注意力，并指出 P/D 分离缺失；不根据这几题推断岗位题频。新增 [注意力机制](../../legacy/internship/1/archive/before-consolidation-2026-11-03/07-attention-mechanisms.md)、[Prefill/Decode 分离](../../legacy/internship/1/archive/before-consolidation-2026-11-03/08-prefill-decode-serving.md)、[vLLM](../../legacy/internship/1/archive/before-consolidation-2026-11-03/09-vllm-practice.md)与[30 题及答案](../../legacy/internship/1/archive/before-consolidation-2026-11-03/10-interview-bank.md)。

- 原理：[Transformer](https://arxiv.org/abs/1706.03762)、[MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html)、[Performer](https://arxiv.org/abs/2009.14794)、[Longformer](https://arxiv.org/abs/2004.05150)、[DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[Ring Attention](https://arxiv.org/abs/2310.01889)、[Speculative Decoding](https://arxiv.org/abs/2211.17192)。
- 服务与版本：[DistServe 原论文／USENIX](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)、[vLLM 0.11.0 架构](https://docs.vllm.ai/en/v0.11.0/design/arch_overview.html)、[性能调优](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html)、[自动前缀缓存](https://docs.vllm.ai/en/v0.11.0/design/prefix_caching.html)、[实验性 P/D 分离](https://docs.vllm.ai/en/v0.11.0/features/disagg_prefill.html)、[快速入门](https://docs.vllm.ai/en/v0.11.0/getting_started/quickstart.html)。具体默认值、connector、命令和版本以运行环境文档为准。
- 可信度：公式和小样例是明确假设下的教学推导；P/D 成效取决于流量、网络和实现。没有部署 vLLM 或进行实际 GPU、多机测试；不能将阅读材料写作本人经历。
- 补充的较新代表工作：[Gated DeltaNet](https://arxiv.org/abs/2412.06464)、[DeepSeek-V3.2-Exp DSA 的 vLLM 说明](https://vllm-project.github.io/2025/09/29/deepseek-v3-2.html)；简单线性核示例不能替代这些模型的完整实现。[vLLM 0.11.0 V1 指南](https://docs.vllm.ai/en/v0.11.0/usage/v1_guide.html)用于说明统一调度。
- 校验：CPU 下四组随机输入的因果线性核递推与显式逐位置核计算一致、反向梯度有限；空输入、512 MiB 传输算例、30 题编号及活动文档本地链接通过。没有运行 GPU 内核、vLLM 服务或网络性能测试。

## 9. 2026-11-03：按知识含义合并

本人指出材料持续增量、语义相近的内容分散，要求实际合并。现整合为[模型与注意力](model-attention.md)、[推理与服务](inference-service.md)、[多卡与 MoE](parallel-moe.md)、[工程与数据](engineering-data.md)，题目和答案合并至[统一题库](interview-qa.md)；10 份旧内容及历史模拟逐字保存在[合并前快照](../../legacy/internship/1/archive/before-consolidation-2026-11-03/)。上文按旧日期保留的文件链接指向当时快照。
