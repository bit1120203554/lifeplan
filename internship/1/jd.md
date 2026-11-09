# JD｜词元星火 AI Infra 系统工程师实习

来源：2026-10-24 本人提供的招聘描述。没有将外部相邻岗位的条件套到本岗。

## 职位描述

1. 研发和优化公司的基础设施平台产品、技术方案和产品规划。
2. 组织推进公司基础设施平台的技术架构设计。
3. 与算法、数据等上下游产品及业务团队协作，优化系统端到端效率。

## 职位要求

- 计算机科学与技术、人工智能、机器学习、自动化等相关专业硕士及以上。
- 熟练使用 Linux；熟练掌握大模型结构及原理；熟练掌握 PyTorch 框架及开发。
- 掌握分布式高速通信基本原理并能运用；了解 MoE 计算及专家优化。
- 熟悉 Kubernetes、Docker、计算机架构、底层通信；具备排障经验。
- 熟悉常见大模型数据处理方法及原理。
- 熟练掌握 Python／C++／Rust 等语言，高性能计算经验优先。
- 独立分析、学习、沟通协作、技术钻研能力；代码能力强，能 profiling、分析瓶颈并跨层优化。

## 转换成面试检查点

| JD 表述 | 实际要解释的能力 | 准备文件 |
|---|---|---|
| 大模型＋PyTorch | 结构、训练循环、显存、精度 | [模型与注意力](model-attention.md) |
| 推理与端到端效率 | KV／FlashAttention／分页／量化与指标 | [推理与服务](inference-service.md) |
| 分布式高速通信 | rank、collective、DDP、并行、拓扑与 hang | [多卡与 MoE](parallel-moe.md) |
| MoE／专家优化 | 路由、均衡、容量、EP、通信与矩阵乘 | [MoE](parallel-moe.md#moe) |
| Linux／架构／排障 | CPU／GPU／I/O、时间线、内存与优化证据 | [工程与排障](engineering-data.md) |
| 容器与数据 | 分词／mask、DataLoader、checkpoint、Docker／K8s | [数据与容器](engineering-data.md#data) |
