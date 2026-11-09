# 模型与系统手写代码｜面试背诵版

更新：2026-10-25（K05 增加 Pre／Post-Norm）。按本人要求，先补 Transformer、KV cache、MoE、LRU 等模型／系统代码，暂不扩充算法题。模板以 Python／PyTorch 为主，每段给输入、步骤、关键解释与检查例子；这是教学实现，不是本人已有工程经历，也不是企业真题频率统计。

## 这次怎样背

| 顺序 | 模板 | 先能写到什么程度 |
|---|---|---|
| 1，核心 | [K03 注意力](#k03)、[K04 多头注意力](#k04) | 写出投影、分头、分数、mask、softmax、合头 |
| 2，核心 | [K08 LRU](#k08) | 不依赖 OrderedDict，哈希表＋双向链表实现 |
| 3，核心 | [K07 MoE](#k07) | router → top-k → dispatch → expert → 加权 combine |
| 4，进阶 | [K05 Transformer](#k05)、[K06 KV cache](#k06) | 组成 decoder block；解释缓存位置与因果 mask |
| 支撑 | [K01 softmax／交叉熵](#k01)、[K02 归一化](#k02) | 解释数值稳定、归一化维度与方差 |
| 有余力 | [K09 LoRA](#k09)、[K10 RoPE](#k10) | 低秩更新与旋转位置编码 |

先看注释复述 → 合上默写 → 用最小输入走一遍 → 回答两条追问。核心四项先过一轮，再加 block／cache；不把全部模板塞进面试前一晚。最小调用按本页从上到下执行，后段依赖前段已定义的类／函数。

## 共同约定

- 张量默认 FP32；B=batch，T=序列长度，D=模型宽度，H=头数，d=D/H，E=专家数，K=每 token 选的专家数。
- attention 的布尔 mask 在本页统一为 **True=允许关注**，不是所有框架接口都使用这个方向。
- 这些模板故意展示普通张量运算，便于解释；长序列服务应考虑融合算子、缓存管理与实际性能分析。

<a id="k01"></a>

## K01｜稳定 softmax 与交叉熵

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

<a id="k02"></a>

## K02｜LayerNorm 与 RMSNorm

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

<a id="k03"></a>

## K03｜缩放点积注意力，先写这一个函数

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

## K04｜完整多头因果自注意力 MHA

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

## K05｜Transformer：区分 Pre-Norm 与 Post-Norm

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

<a id="k06"></a>

## K06｜KV cache：最容易写错的是位置偏移

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

<a id="k07"></a>

## K07｜完整单设备 Top-K MoE

10/26：[负载均衡从零讲解与配套代码](07-moe-load-balancing-basics.md)补充了 assignment 计数、`EΣf_iP_i`、硬计数／可微概率、容量和 dropless。下方前向模板仍未自动接入辅助损失；选中后的归一化权重不能代替辅助项需要的完整路由概率。

输入输出 [B,T,D]。router 输出每 token 对 E 个专家的分数；每 token 取 K 个专家，本版将选中概率重新归一，再按 token 索引加回输出。

~~~python
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

**记忆口令**：先压平 token → router 概率 → top-k 索引和权重 → 按专家找 token → 算专家 → 按原索引加权累加。

**逐维解释**：BT 个 token 各选 K 个，共 BTK 个 assignment；torch.where 的两个返回值是 token 号、top-k 槽位。M 是这一专家收到的 assignment 数。不同专家输出不能直接 concat，必须恢复 token 对应关系。

**最小检查**：E=3、K=2，输出 shape 不变；所有专家设成恒等映射后，因选中权重和为 1，输出应近似等于输入。再让所有 token 路由到同一组专家，检查空专家与重复累加。

**常见追问与限制**：

- 选专家使用离散 top-k 索引，不代表索引本身可微；被选概率和专家参数仍可参与梯度计算。
- 这版 K=1 又对选中权重归一，会把权重变成 1，不能依靠输出损失给 router 提供通常期待的门控梯度。学习路由要明确门控约定、辅助目标和实际方法。
- 没有容量限制、丢 token、负载均衡损失、EP、All-to-All、grouped GEMM 或通算重叠。单设备 Python 循环不是高效大规模 MoE。
- 总共 E 个专家仍需存储。主要专家计算随 assignment 数增长，另有 router、重排和调度成本，不保证稀疏就一定更快。

依据：[Switch Transformer](https://arxiv.org/abs/2101.03961)、[Megatron MoE 实现说明](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)、[PyTorch index_add_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html)。不同 MoE 的门控／归一化并不统一，本页明确采用上述约定。

<a id="k08"></a>

## K08｜LRU：哈希表＋双向链表，完整手写

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

可用标准库时的短版见 [模拟三答案 C11](../../legacy/internship/1/archive/before-consolidation-2026-11-03/mock-answers.md#c11-lru-参考实现)，手写要求下优先准备本页双向链表版。

<a id="k09"></a>

## K09｜LoRA，P1 备用

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

<a id="k10"></a>

## K10｜RoPE，P1 备用

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

## 本次校验记录

2026-10-24，在现有 PyTorch 2.13.0+cpu 环境用小型合成输入运行校验，未安装新依赖或使用研究数据。已检查：softmax／交叉熵及梯度与框架对照；LayerNorm／RMSNorm；注意力数值与因果性；Transformer 前后向；不同 chunk 的 KV cache 一致性；MoE 对照、梯度与空专家；LRU 随机序列对照和链表一致性；LoRA 初始化与冻结；RoPE 范数和相对位置。

知识点内同步插入的同名函数／类与本页定义一致。这里验证的是教学代码正确性，不是 GPU、分布式或生产性能。官网 stable 页面可能显示更新版本，实际校验版本以本段为准。

10/25 补充校验：Pre／Post 两种 block 分别与同权重、同因果 mask 的 PyTorch TransformerEncoderLayer 对照，输出、输入梯度与注意力投影梯度在浮点容差内一致；零残差分支验证、两种小模型的 final norm 选择与前后向通过。默认行为保持 Pre-Norm，三个文档的同名代码定义一致。上述检查验证执行顺序，不作为训练稳定性或效果优劣的实验。

## 最后用这六个问题验收

1. 不看稿写注意力，并逐行标 shape；第一个 token 能否读到后面？
2. 写出分头／合头；为什么需要输出投影？
3. 分别写 Pre-Norm／Post-Norm block；归一化相对残差的位置、final norm、位置表示和因果 mask 各管什么？
4. 写缓存拼接，给 L=3、S=2 的正确 mask。
5. MoE 为什么用累加，哪些地方是选择、哪些地方是权重？
6. LRU 摘节点改哪两条连接，接尾改哪四条？更新旧 key 是否多占容量？

模板是可运行的学习参考，背完仍需独立默写和换例验证。这里不要求为准备面试训练模型或搭建集群。
