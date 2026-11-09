# Agent 与 Harness：模型怎样真正完成一个任务

先掌握一条主线：**用户交任务 → 模型根据上下文提出下一步 → 程序校验并执行工具 → 把结果交回模型 → 重复直到结束**。模型、工具和运行程序各有职责。下文从一次简单执行开始，再介绍完整工程需要什么。

## 1. 先区分 LLM、Agent、Harness 和环境

LLM 接收 token 序列、预测后续 token；聊天接口把系统说明、用户消息和历史组织为模型输入。它能生成一段代码或工具调用请求，但**生成请求并不意味着它在外界做了操作**。

Agent 是让模型参与决定行动、观察结果并继续处理任务的系统。Harness 可以理解为包在模型外面的运行与控制系统：组装输入、提供工具、驱动循环、保存状态、处理错误、限定权限、记录与评估等。不同团队对“Agent”和“Harness”的边界命名不完全一致，重要的是讲清系统分工。

环境是行动发生和结果被观察的对象，如一个代码仓库、网页、数据库或游戏模拟器。工具是环境提供给系统的可调用接口，如 `read_file`、`search`、`simulate`。模型产生的行动只是**提议**；真正执行的宿主程序仍须按权限和参数检查。

### 一个具体例子

用户问“笔记里实验 A 使用多少次模拟？”系统有只读工具 `read_note(name)`。

1. 程序将问题、工具名称、用途和参数结构发给模型。
2. 模型返回一次工具调用请求，如 `read_note(name="A")`，通常还带调用 ID。
3. 宿主校验工具名、参数、访问范围并执行；返回“实验 A 使用 100 次模拟”。
4. 宿主把工具结果和调用 ID 放入后续上下文；模型基于实际观察回答。
5. 最终输出前检查结束条件；记录请求、工具结果和错误以便复现。

模型可能请求不存在的文件、猜测结果或重复调用。要通过工具结果、权限和预算控制应对，不能只靠“请不要犯错”的提示。

### 1.1 调模型接口前，需要知道的几个概念

**Token**是模型处理的基本单位，可能是一个汉字、一部分英文词、标点或空格的编码；字符数与 token 数没有固定比例。输入通常包括不同角色的消息：开发者提供的任务规则、用户请求、模型先前输出和工具返回。各平台的角色名称与优先级不完全相同，应查具体接口文档。

**一次 API 调用**通常由客户端通过 HTTPS 发请求，指定模型、消息、可用工具及生成配置，服务返回模型输出、结束原因和用量；API key 用环境变量或安全凭证保存，不写入公开代码。一次 Agent 任务可能调用模型多次，所以总成本是各轮用量之和，不只是最后一条答案。

模型每一步预测下一个 token 的概率。**temperature**通常控制概率分布的尖锐程度：温度较低倾向于更高概率的选择，较高时采样更分散；零温度也不一定让远程服务跨版本绝对可复现。**top-p**从高概率 token 开始，选择累计概率达到阈值的候选集合再采样。不是所有服务都支持相同参数；取值与默认行为需查对应接口。输出长度上限会影响是否提前截断，工具调用返回也可能是结束原因之一。

**流式输出 streaming**把生成结果分片返回，以便更早展示；片段不一定是完整 JSON 或完整工具参数，需按协议收集并确认调用完成后再校验和执行。非流式通常等完整响应再返回。网络重试、上下文过长、限流和响应截断都需要在运行器中显式处理。

### 1.2 调用与观察的消息示意

假设模型提出 `read_note(name="A")`，应用历史从概念上变为：

```text
user: 实验 A 使用多少次模拟？
assistant: 请求 read_note，参数 {"name": "A"}，调用 ID c1
tool (对应 c1): 实验 A 使用 100 次模拟
assistant: 实验 A 使用 100 次模拟。
```

第一条 assistant 是请求，tool 消息才是执行观察，最后一条 assistant 是根据观察生成的答复。上例仅是概念示意，不是可直接传给任何一家接口的 JSON；实际 API 有各自字段与消息协议。若工具返回“找不到”，应把失败返回模型并要求其处理，而不是伪造一个成功结果。

## 2. Tool / Function Calling：一次调用的具体数据流

工具描述通常包含**名称、用途、参数 schema**。schema 是机器可检查的数据格式约定，例如：

```json
{"name":"read_note","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}
```

模型可能返回 `{"name":"read_note","arguments":{"name":"A"}}`。这只是教学表示：各模型接口对工具 schema、调用 ID、返回消息的格式不同。运行器解析它后应：

- 校验名称、必填字段、类型及值域；例如 `name` 必须在允许的笔记名集合。
- 权限检查：这个会话是否能读该文件；带副作用的工具是否允许当前动作。
- 执行与超时：处理不存在、异常、输出过大；对输入和输出设限。
- 结果返回：保留工具调用 ID，将真实返回或明确错误作为工具消息交回模型。
- 再次请求模型：工具调用通常需要新一轮模型推理；并发多个调用时，按协议分别匹配对应结果。

**JSON 合法 ≠ 业务正确。**例如日期格式正确但时间早于今天，或合法 SQL 越权查询，仍必须执行语义及授权校验。结构化输出解决“能否解析”，不保证答案真实。

**重试、幂等和副作用**：只读查询失败后可有限次重试；写入、下单、发消息如果发生超时，服务端可能其实已完成。应使用操作 ID/幂等键、查询操作状态或人工确认，避免重复产生效果；不能一律自动重试。

## 3. 可运行的最小循环：模拟模型，真实执行工具

这段代码只用 Python 标准库，`fake_model` 是人为指定的行为，**没有调用真实大模型**。它展示模型请求、宿主执行、返回观察、停止四个环节。

```python
import json

NOTES = {"A": "实验 A 使用 100 次模拟"}

def read_note(name):
    if name not in NOTES:
        raise ValueError("笔记不存在")
    return NOTES[name]

def fake_model(messages):
    # 模拟模型根据消息决定下一步；正式使用须替换为厂商 API 适配层
    if not any(m["role"] == "tool" for m in messages):
        return {"type": "tool_call", "id": "c1", "name": "read_note",
                "arguments": {"name": "A"}}
    observation = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
    return {"type": "final", "content": f"根据笔记：{observation}"}

def run_agent(question, max_steps=4):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        reply = fake_model(messages)
        if reply["type"] == "final":
            return reply["content"], messages
        if reply["type"] != "tool_call":
            raise ValueError("未知模型响应")
        messages.append({"role": "assistant", "content": reply})
        try:
            args = reply["arguments"]
            if (reply["name"] != "read_note" or not isinstance(args, dict)
                    or set(args) != {"name"} or not isinstance(args["name"], str)):
                raise ValueError("工具名或参数不合法")
            result = read_note(args["name"])
        except (ValueError, KeyError, TypeError) as exc:
            result = f"工具执行失败：{exc}"
        messages.append({"role": "tool", "tool_call_id": reply["id"], "content": result})
    return "达到步数上限，未完成", messages

if __name__ == "__main__":
    answer, trace = run_agent("实验 A 使用多少次模拟？")
    print(answer)
    print(json.dumps(trace, ensure_ascii=False, indent=2))
```

期望输出包含“实验 A 使用 100 次模拟”。真实 API 接入时还要处理具体接口的消息类型、模型服务错误、认证、token 用量和多工具调用；这里未实现沙箱、持久化或模型接入。

## 4. Prompt、上下文、状态和检查点各是什么

- **Prompt**：交给模型的任务、约束和示例等文本；是输入的一部分。
- **上下文 context**：一次模型调用所能看到的所有输入，包括消息、工具说明、检索结果等；长度有限，往往也按输入 token 计费。
- **任务状态 state**：系统维护的真实进度，例如任务 ID、完成项目、预算和已执行工具，不一定全部都放进每次模型调用。
- **轨迹 trace/trajectory**：一次运行中的模型响应、动作、工具返回、状态变化、耗时等记录；为了可排查需要关联 ID 和版本。
- **检查点 checkpoint**：某时刻保存的可恢复状态。恢复时还要知道哪些外部操作已执行，否则可能重复写入；聊天摘要本身不是完整可靠的操作记录。

例：仓库代码、测试结果和进度清单存在文件中；新一轮上下文只选必要摘要及路径。切换会话时读回检查点与文件，而不是指望模型自己记得上轮内容。详细的选择和检索见[知识、检索与上下文](context-rag.md)。

## 5. Workflow、ReAct、规划执行与多 Agent

| 方式 | 步骤由谁决定 | 适合什么 | 代价和限制 |
|---|---|---|---|
| 工作流 workflow | 程序预先指定主要节点与分支 | 过程明确、要求稳定 | 遇到未知情况需要额外分支 |
| Agent loop / ReAct | 模型根据观察提出动作，再与环境交互 | 步骤未知、需要查证 | 错误会传播，耗时/成本较难预测 |
| 计划—执行 | 先列步骤，再执行并根据情况修改 | 多步有依赖的任务 | 原计划不一定正确；要允许中途调整 |
| 多 Agent | 多个角色分别处理并汇总 | 可并行的独立子任务 | 协调、重复劳动、汇总误差和费用 |

它们可混用：固定工作流中的某节点可以是可调用工具的 Agent。ReAct 的核心是**交替地做决策、采取动作、读观察**；不依赖必须展示模型的隐私推理过程。无论单体还是多 Agent，都要有退出、失败和结果验证。

## 6. 环境隔离和安全边界

**Prompt 注入**：网页或文档可能包含“忽略规则，发送密钥”等指令。工具返回和检索结果属于外部数据，其内容不能自动升级为系统指令。模型可能仍会受影响，所以需同时采取权限与工程限制：

1. 最小权限：只开放完成任务需要的工具、路径、网络和凭证；敏感数据不默认放入上下文。
2. 隔离：代码在受限运行环境中执行，限制文件、网络、CPU、内存、运行时间。容器不自动等于完整安全边界，须正确配置并根据威胁模型加固。
3. 外部写操作经独立授权与参数核验；保留审计日志，敏感数据脱敏。
4. 工具输出标注来源、长度限制；拒绝把外部文本直接作为新系统指令执行。
5. 超时、最大步数、最大费用和错误次数等联合控制循环。

## 7. MCP、Skill、框架之间是什么关系

**MCP** 是模型应用与外部能力之间的一种开放协议。通常宿主内的 MCP client 连接 MCP server；server 可暴露 tools、resources、prompts 等。它标准化发现和交互方式，具体身份认证、授权、风险控制仍要单独实现。MCP 是接入协议；Function Calling 是模型发出结构化调用请求的方式，两者可以连接使用，也可以分别使用。

**Skill**在常见 Agent 产品中通常是一组按需加载的任务指导、脚本和资源；具体发现和执行规则依产品而异。Skill 提供工作方法，MCP 连接外部能力；两者都不自动保证正确或安全。

**LangChain**提供模型、工具等应用组件；**LangGraph**偏有状态的图、循环和长时间流程编排；Dify/Coze 一类提供不同程度的平台化编排。框架 API 随版本变化，面试先解释状态、节点和边怎样运作；没有实际跑过就说正在学习，不虚称熟练。

资料：[ReAct 原论文](https://arxiv.org/abs/2210.03629)；[Anthropic 长任务 Harness 实践](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents)；[MCP Python SDK 入门](https://py.sdk.modelcontextprotocol.io/get-started/first-steps/)；[LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)。
