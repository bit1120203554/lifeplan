# FlashAttention 与推理服务｜先懂机制，再记名字

整理：2026-10-26。用于 10/27 面试准备。先读 1—6 和第 9 节对照表，理解后再看代码与进阶。这里采用普通 dense／causal attention 说明，不覆盖所有注意力变体或最新版实现。

## 1. FlashAttention 到底是什么

**它是一种高效计算标准注意力的算法及相应 GPU 实现。** 核心工作是把计算分块，在小而快的片上存储里完成局部运算，并正确合并结果，减少大中间矩阵在显存中的读写。

它仍计算：

$$
O=\operatorname{softmax}(QK^\top/\sqrt d)V.
$$

Q 是查询，K 是用于匹配的键，V 是被加权的内容；d 是 Q／K 的头宽。mask 决定哪些位置允许关注，softmax 沿每个 query 对应的 key 维归一化。

FlashAttention 主要改变计算顺序、存储位置和 GPU 任务组织。普通 dense attention 所需的合法注意力关系仍参与计算，不能说“只看局部，所以更快”。[原论文](https://arxiv.org/pdf/2205.14135)

## 2. 普通 attention 为什么会浪费时间和显存

朴素实现的计算片段可以写成三步；这里假定 q／k／v 已给出，d 是头宽：

~~~python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
probabilities = torch.softmax(scores, dim=-1)
output = probabilities @ v
~~~

对一条长度为 T 的序列，每个 query 与 T 个 key 比较，scores 的每个头有 T×T 个数；probabilities 也有相同大小。

若这些步骤分别执行，就可能发生：把 scores 写入显存 → softmax 再读出来 → 写 probabilities → 乘 V 时再读出来。计算单元可能花很多时间等待搬运。这里描述的是朴素实现，具体框架是否融合应看实际执行。

**自拟容量例子**：B=1、H=32、T=4096，假设每个中间元素 2 bytes，仅一张 scores 就需要：

$$1\times32\times4096^2\times2=1\ \mathrm{GiB}.$$

再保留 probabilities 又约 1 GiB。若采用 FP32 中间结果，每张约 2 GiB。这里只算这两类张量，不是整个模型显存。

**“少存矩阵”与“少算匹配”是两件事。** FlashAttention 的关键是无需将完整 scores／probabilities 保存在显存中，但不会因此把一般全注意力的算术量变成线性。

## 3. HBM、片上存储、kernel 和融合各是什么

| 名词 | 具体含义 | 在这里的作用 |
|---|---|---|
| HBM | GPU 的高带宽显存，容量相对大 | 放模型张量、输入和输出；访问仍有代价 |
| 片上 SRAM／共享内存 | 计算芯片内的小而快的存储，容量有限 | 暂存正在计算的一小块数据 |
| 寄存器 | 执行线程使用的更小存储 | 保存累计量和局部数据 |
| kernel／内核 | 提交到 GPU 执行的一段设备程序 | 完成一个或多个计算步骤 |
| kernel fusion／算子融合 | 把原本分开的步骤合并组织执行 | 减少启动和中间结果的写回／重读 |

GPU 还有缓存等层次，这张表只是理解当前机制所需的简化。不能说“GPU 只有 HBM 和 SRAM”，也不能说“HBM 就是 CPU 内存”。

由于整张 T×T 矩阵通常放不进片上空间，不能简单“把整张矩阵搬到 SRAM”。可行做法是一次处理一块，让需要的临时数据尽量留在片上。

## 4. 分块以后怎样算

先固定一块 Q，它包含若干 query。然后逐块读取 K、V：

1. 计算这一块 Q 与当前 K 块的分数。
2. 应用对应位置的 mask。
3. 用在线 softmax 更新每个 query 的统计量和输出累计量。
4. 处理下一个 K／V 块，最终写出结果。

完整 kernel 会同时处理多条 query、复用载入的数据；各版本的循环顺序和工作分配可以不同。**不需要把整张分数或概率矩阵先放进 HBM，再让下一步重读。** 但 Q、K、V、输出与部分统计仍需存取，不是完全不访问显存。

这里最难的一点是 softmax：每块只看到部分 key，怎样知道全行的分母？不能每块独立 softmax 后直接相加。

<a id="online-softmax"></a>

## 5. 在线 softmax：用三个累计量合并全局结果

“在线”指数据分块到达时更新统计，不是联网。

对一个 query，记各 key 的分数为 s_j，内容为 v_j。最终想得到：

$$o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.$$

为避免指数溢出，减去同一个最大值，不改变比值。逐块处理时只维护：

- m：已处理分数的最大值。
- l：以 m 为基准的指数和，即 sum exp(s_j−m)。
- u：同样基准下的加权内容和，即 sum exp(s_j−m)·v_j；u 通常是向量。

新块分数为 s，令：

$$
m'=\max(m,\max s),\qquad \alpha=e^{m-m'}.
$$

更新：

$$
l'=\alpha l+\sum_{\text{新块}}e^{s_j-m'},\qquad
u'=\alpha u+\sum_{\text{新块}}e^{s_j-m'}v_j.
$$

最后输出 u/l。**alpha 的作用是把旧累计量换到新的最大值基准。** 旧数据不用重新全部读回，也不需要保存每一个概率。这个合并关系可直接由指数的乘法性质推导；它是计算同一个全局 softmax 的方式。

### 自拟手算例子

一个 query 对三个 key 的分数为 [0, ln2, ln4]，V 简化成三个数 [10,20,30]。指数是 [1,2,4]，完整答案为：

$$(1\times10+2\times20+4\times30)/(1+2+4)=170/7\approx24.286.$$

把前两个 key 作为第一块、最后一个作为第二块：

| 处理到哪 | m | l | u | u/l |
|---|---|---:|---:|---:|
| 第一块 | ln2 | 1/2+1=1.5 | 10/2+20=25 | 16.667 |
| 第二块结束 | ln4 | 1.5×0.5+1=1.75 | 25×0.5+30=42.5 | 24.286 |

第二块出现更大的最大值，旧累计量都乘 exp(ln2−ln4)=0.5，因此结果仍等于完整 softmax。

如果各块独立归一化，再把两个输出平均，会得到 (16.667+30)/2≈23.333，是错的：两块在全局中的概率质量并不相等。

在线归一化背景见 [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)；完整 FlashAttention 使用分块和这类归一化合并，而不只是分开调用几个普通矩阵算子。

## 6. 为什么能省显存，为什么反向还能算

**前向**：分块产生局部分数，更新输出与归一化统计，不保留完整 T×T 中间矩阵。在头宽固定的比较下，注意力中间存储不再需要随 T² 保存；Q／K／V 与输出本身仍占内存。

**反向**：训练需要概率等中间信息，但不一定要从前向一直保存它们。可以保留 Q／K／V、输出和逐行归一化统计等，在反向时按块重新计算所需部分。多一些重算，可能换来更少的 HBM 读写。

这不是免费消除计算，也不表示全部训练状态都变少。原论文 §3.1 和 Algorithm 1 说明分块与重计算；其 dense attention 算术量仍为 O(T²d)。[原论文](https://arxiv.org/pdf/2205.14135)

**exact 的意思**：实数运算下计算同一个注意力目标，不是靠删掉合法连接或低秩近似换精度。浮点运算顺序和累积方式改变后，结果不保证逐位相同，比较时要看合理数值容差。

## 7. 可运行的教学代码：一个 query 分块扫全部 key

本段用于理解 m／l／u，不要求明天手写 CUDA。它是普通 PyTorch 循环，**没有实现 GPU 的片上存储调度或融合内核，不能用于宣称 FlashAttention 加速**。

输入 q=[d]、k=[N,d]、v=[N,dv]，均为相同设备／dtype 的 FP32 或 FP64 小张量，并保证实际点积分数有限。allowed=[N]，为同设备布尔向量，True 表示允许；不允许任何 key 时显式报错。为了简单，示例只处理一个 query，完整方法会对 query 块并行。

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

最小调用：

~~~python
scores_demo = torch.log(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))
q_demo = torch.ones(1, dtype=torch.float64)
k_demo = scores_demo[:, None]
v_demo = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float64)
answer_demo = online_attention_one_query(q_demo, k_demo, v_demo, block_size=2)
assert torch.allclose(answer_demo, torch.tensor([170 / 7], dtype=torch.float64))
~~~

分母和最大值修正是本段重点。正式 kernel 还要安排多 query、GPU 线程、访存、精度、mask、dropout 与反向等；这里的 Python 循环和 .item() 反而可能引入启动／同步开销。启用 autograd 时这段代码还会保留其计算图，不能据它推断正式 FlashAttention 的训练内存。

想看真实内核结构时，再读 [Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，本次不要求临时实现一套内核。

## 8. PyTorch 怎样使用，调用名字不代表后端

等长、无 padding、完整序列因果注意力的最小示意：

~~~python
import torch.nn.functional as F

q_api = torch.randn(1, 2, 4, 8)  # [B,H,T,d]
k_api = torch.randn_like(q_api)
v_api = torch.randn_like(q_api)
out_api = F.scaled_dot_product_attention(
    q_api, k_api, v_api,
    dropout_p=0.0,
    is_causal=True,
)
~~~

该 API 可选择融合或数学实现，取决于版本、设备、dtype、shape 等。CPU 上算出结果，不代表运行了 GPU FlashAttention；实际应查后端选择与 profiler，而不是只看函数名。

这个例子 Q／K 等长。带缓存、Q 与 K 不等长时，要核对 mask 的位置对齐，不能直接复制 is_causal=True 就假定一定符合 decode 的绝对位置；回查 [KV cache 模板](05-model-system-coding.md#k06)。推理时显式传 dropout_p=0，不能仅依赖外层 eval() 改变这个函数的参数。

依据：[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。接口细节按实际版本核对。

<a id="serving-map"></a>

## 9. KV cache、PagedAttention、连续批处理、vLLM 分别是什么

**10/26 加强入口**：本节保留与 FlashAttention 的层次对照；缓存的逐步前向、显存推导、分页寻址、前缀共享、量化及追问集中在 [专项 02](../1/inference-service.md)。本次 KV／PagedAttention 已列为优先复习内容，量化基础随后补。

| 概念 | 具体在改什么 | 没有因此自动解决什么 |
|---|---|---|
| KV cache | 生成时保留历史 token 的 K／V，避免每步重做历史投影 | 不消除读取历史缓存；不是完整分数矩阵缓存 |
| FlashAttention | 分块、在线归一化和内核组织，减少 attention 的中间存储与搬运 | 不自动管理多请求的 KV 生命周期，也不压缩每个 token 的 KV 内容 |
| PagedAttention | 用逻辑块到物理块的映射，管理不连续存放的 KV，并让注意力能读取这些块 | 不等于把 KV 自动换到 CPU，也不等于数学上只需更少的历史内容 |
| continuous batching／连续批处理 | 在迭代边界更新活动请求集合，让已具备执行条件的新请求加入 | 不免除新请求的 prefill；不保证所有请求的延迟都降低 |
| vLLM | 大模型推理与服务系统，组织模型执行、请求调度和 KV 管理等 | 不等于某一个 attention kernel，也不等于改了模型全部结构 |

### PagedAttention 的具体例子

假设每块能放 4 个 token 的 KV，一条已有 10 个 token 的请求需要 3 个逻辑块，可以映射到物理块 [7,2,9]，无需占连续地址。有效内容有 10 个槽位，最后一块可能剩 2 个空位；这里没有把缓存内容压缩成更少 token。

相比为每条请求预留很长连续空间，按需分配块能减少预留与碎片浪费。内核按块表找到真实 K／V。符合条件的相同前缀还可以共享块，写入分歧时再按设计复制；共享需要模型、位置和前缀等上下文一致，不能只看文字相似。

依据：[PagedAttention／vLLM 论文](https://arxiv.org/abs/2309.06180)、[作者对块映射与共享的说明](https://vllm.ai/blog/2023-06-20-vllm)。

### 连续批处理的具体例子

请求 A 只剩 2 个输出 token，B 还剩 8 个。固定批次可能让 A 结束后留下空闲容量，直到 B 结束再换整批。

按迭代调度时，A 结束后，调度器可以在后续迭代纳入**已经完成所需 prefill**的新请求 C，不必等 B 结束。实际系统还需决定 prefill 与 decode 的预算和优先级，不能把加入新请求理解成无成本。

更多并发可以提高吞吐，也可能增加每轮计算、排队和 KV 占用，所以仍看 TTFT、TPOT 与尾延迟。这是调度层，与 attention 内核层可以同时优化。迭代级调度的原始工作可参考 [Orca，OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf)。

## 10. 面试还可能追问的两个名字

**FlashAttention-2**：在相同 attention 目标上继续改 GPU 工作分配，减少非矩阵乘开销、增加可用并行度，并减少 warp 间共享数据／通信成本。warp 是 NVIDIA GPU 的线程执行分组。具体收益依形状与设备，不背一个统一加速倍数。[作者论文](https://arxiv.org/abs/2307.08691)

**grouped GEMM**：GEMM 是通用矩阵乘。MoE 不同专家可能分别收到 2、7、0 个 token，形成多个不同大小的矩阵乘任务。Grouped GEMM 将这些任务一起调度，减少零散启动并利用更多计算资源；各专家权重仍然不同，不是把专家随便合成一个权重矩阵。它也不负责解决跨节点 All-to-All。[CUTLASS 官方说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)

## 11. 现在能用于面试的回答

FlashAttention 是计算标准注意力的一种高效实现。朴素实现会生成并读写较大的分数和概率矩阵；它将 Q、K、V 分块，在片上处理局部分数，使用在线 softmax 的统计量合并全局结果，避免完整中间矩阵反复写入显存。训练反向还可以重算局部量来省存储。它在数学上计算同一个注意力，浮点结果允许有数值差异；一般 dense attention 的算术复杂度仍是平方级。实际收益要看硬件、形状和后端。

## 12. 合上资料检查六件事

1. 写 scores 的 shape，并算出为什么示例单张矩阵需要 1 GiB。
2. 解释 HBM 与片上存储分别放什么，为什么要分块。
3. 复算 170/7，指出逐块 softmax 后平均错在哪里。
4. 说出 m、l、u，以及换最大值基准时为什么旧量要乘 alpha。
5. 区分算术量、注意力中间存储、KV 容量和端到端时延。
6. 分别用一句话解释 FlashAttention、PagedAttention 和连续批处理。

本页没有将学习模板写成本人既有 GPU 内核或服务部署经验。首先做到能解释机制，再谈自己做过哪些测量。

## 本次教学代码校验

2026-10-26，现有 PyTorch 2.13.0+cpu 环境：64 组 FP32／FP64、不同分块大小、尾块和 mask 对照通过；q／k／v 梯度与完整 softmax 对照、大分数稳定性、全屏蔽报错、等长因果 SDPA 示例及手算通过。没有安装新依赖，没有运行 GPU FlashAttention 内核或性能基准。
