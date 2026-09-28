# 原理篇（七）：从零推导一个 Agent（ReAct 原理）

> 对应课程：**第 10 课**
>
> 这是全套文档里最重要的一篇。**读完它，你再看任何 Agent 框架都能一眼看穿。**
> 建议先跑一遍 `python lessons/10_handmade_agent.py`，再回来读。

---

## 一、先把"Agent"这个词拆掉

市面上的说法经常把 Agent 讲得很玄：自主、规划、反思、记忆……

我们把它们全部剥掉，只留最核心的定义：

> **Agent = 一个大模型 + 一个循环 + 一组工具。**

用伪代码表示：

```python
def agent(question):
    messages = [question]
    while True:
        reply = llm(messages)              # ① 让模型说下一步
        if reply.has_no_tool_call:
            return reply.text              # ② 说完了，结束
        result = execute(reply.tool_call)  # ③ 它要动手，我们帮它动手
        messages.append(result)            # ④ 把结果告诉它，继续循环
```

**这就是全部。** 剩下的都是工程细节：

| 你以为的"高级能力" | 实际上是什么 |
|---|---|
| 记忆 | 把 `messages` 存进数据库（第 03 课） |
| 反思 | 循环里多加一个"批评家"节点（第 09 课） |
| 规划 | 让模型先输出一个任务列表，再逐个执行（第 07 课的 map-reduce） |
| 多智能体 | 循环里换不同的 SystemMessage（第 09 课） |
| 人工介入 | 循环里加 `if 危险: 问人`（第 04 课） |

---

## 二、为什么叫 ReAct

ReAct = **Rea**soning（推理）+ **Act**ing（行动）。

这个名字来自 2022 年的一篇论文，核心洞察是：

> **让模型"边想边做"，比让它"想完再做"效果好得多。**

```
    只推理（Reason Only）：
        问题 → [模型思考很久] → 答案
                 ↑
            只能靠内部知识，无法获取外部信息，也无法验证

    只行动（Act Only）：
        问题 → 调工具 → 调工具 → 调工具 → 答案
                 ↑
            没有推理指导，容易乱调

    ReAct（推理 + 行动交替）：
        问题 → 想：需要查天气 → 做：调天气工具 → 想：查到了，可以回答 → 答案
                 ↑___________________________↑
                        这个交替就是循环
```

**关键：每一轮"行动"的结果都会喂回给模型，作为下一轮"推理"的输入。**
这就是为什么 LangGraph 里那条从 `tools` 回到 `agent` 的边如此重要。

---

## 三、手写第一版：20 行搞定

```python
TOOL_REGISTRY = {"get_weather": get_weather, "calculator": calculator}

def bare_agent(question, max_rounds=5):
    messages = [HumanMessage(content=question)]

    for _ in range(max_rounds):
        # ① 问模型
        reply = llm_with_tools.invoke(messages)
        messages.append(reply)

        # ② 没有工具调用 → 它答完了
        if not reply.tool_calls:
            return reply.content

        # ③ 有工具调用 → 执行，把结果塞回去
        for call in reply.tool_calls:
            tool = TOOL_REGISTRY[call["name"]]
            result = tool.invoke(call["args"])
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=call["id"],      # ← 必须对应
                name=call["name"],
            ))

    return "达到最大轮数仍未结束"
```

**这段代码没有任何框架依赖，但它是完整的 Agent。**

现在回头看 LangGraph 提供的那些组件，你会发现它们都是这段代码的某个部分：

| 手写版里的代码 | LangGraph 的对应物 |
|---|---|
| `reply = llm_with_tools.invoke(messages)` | `agent` 节点 |
| `for call in reply.tool_calls: ...` | **`ToolNode`** |
| `if not reply.tool_calls: return` | **`tools_condition`** |
| `for _ in range(max_rounds)` | 图的循环边 + `recursion_limit` |
| `messages` 列表 | **`State`** 的 `messages` 字段 |
| `messages.append(...)` | **`add_messages`** reducer |

---

## 四、手写第二版：拆解每一块

### 4.1 `bind_tools` 到底发了什么

```python
llm_with_tools = llm.bind_tools([get_weather])
```

它做的事情是：**把工具的说明书附加到每次请求里。**

说明书的内容（可以在第 10 课里实际打印出来）：

```json
{
  "name": "get_weather",
  "description": "查询指定城市的天气。用户问天气、气温时使用。",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名称"}
    },
    "required": ["city"]
  }
}
```

**这些内容从哪来？** 从你的函数自动生成：

| 来源 | 变成 |
|---|---|
| 函数名 `get_weather` | `name` |
| docstring | `description` ← **最重要** |
| 参数名 `city` + 类型注解 `str` | `parameters.properties.city` |
| `Args:` 里对该参数的说明 | `parameters.properties.city.description` |

**所以 docstring 写得含糊，模型就选不对工具。** 这不是玄学，是信息质量问题。

### 4.2 模型返回的 `tool_calls` 是什么

```python
AIMessage(
    content="",                              # 通常是空的
    tool_calls=[{
        "name": "get_weather",
        "args": {"city": "北京"},             # 模型"猜"的参数
        "id": "call_abc123",                 # 本次调用编号
        "type": "tool_call",
    }],
)
```

**三个必须记住的点：**

1. **这是模型的输出，不是执行结果。** 模型只是"提出请求"。
2. **`args` 可能填错。** 它可能传 `{"city": "北京今天"}` 而不是 `{"city": "北京"}`——
   所以工具内部要做参数校验。
3. **模型可能一次要求调多个工具。**

### 4.3 `ToolNode` 的核心逻辑

```python
def my_tool_node(state):
    last = state["messages"][-1]
    outputs = []

    for call in last.tool_calls:
        tool = TOOL_REGISTRY.get(call["name"])

        if tool is None:
            content = "错误：不存在名为 %s 的工具" % call["name"]
        else:
            try:
                content = str(tool.invoke(call["args"]))
            except Exception as exc:
                content = "工具执行失败：%s" % exc    # ← 别让异常炸掉整个图

        outputs.append(ToolMessage(
            content=content,
            tool_call_id=call["id"],
            name=call["name"],
        ))

    return {"messages": outputs}
```

官方 `ToolNode` 额外做的：

| 额外能力 | 作用 |
|---|---|
| 并发执行多个工具 | 一次调 3 个工具时更快 |
| 完善的错误处理 | 返回错误信息而不是抛异常 |
| `InjectedState` 等注入 | 工具可以访问当前状态 |
| 大结果落盘 | 避免把超长结果塞进上下文 |

**核心逻辑没变。** 你完全有能力自己写一个。

### 4.4 `tools_condition` 的核心逻辑

```python
def my_condition(state):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"
```

**三行。** 官方实现也就这么多（只是把 `"__end__"` 换成了常量）。

### 4.5 消息协议的硬性要求

**规则**：模型发了一条带 N 个 `tool_calls` 的消息，后面必须跟齐 N 个 `ToolMessage`，
且 `tool_call_id` 一一对应。

违反会怎样？

```python
# 构造一段"缺了工具结果"的对话
bad_messages = [
    HumanMessage("北京天气"),
    AIMessage(tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "call_xyz"}]),
    HumanMessage("你刚才查到了什么？"),      # ← 中间缺了 ToolMessage
]

llm.invoke(bad_messages)
# 真实的 OpenAI / Anthropic 接口 → HTTP 400
# 错误大意："messages with role 'tool' must be a response to a preceding message with tool_calls"
```

**这条规则解释了第 04 课的一个设计**：用户拒绝执行工具后，我们**伪造**了一条内容为
"【人工审批未通过】" 的 `ToolMessage`。

不是为了骗模型，而是因为：
- 消息历史必须合法，否则下一轮请求发不出去
- 同时要告诉模型"别重试了"

---

## 五、从手写到框架：每一层的价值

```
    第 1 层：纯 while 循环（20 行）
        │
        │  缺什么？  进程一挂全丢；没法暂停；没法观察；
        │           没法并行；出错就从头来。
        ▼
    第 2 层：LangGraph 手写版
        │  把循环变成图：节点是可组合的单元，
        │  状态可以被持久化，每步可以被观察。
        │
        │  缺什么？  每次都要自己写工具节点和判断函数。
        ▼
    第 3 层：用 ToolNode + tools_condition
        │  把通用逻辑封装起来。图结构一模一样。
        │
        │  缺什么？  每次都要自己搭图。
        ▼
    第 4 层：create_agent(llm, tools=[...])
            一行搞定。
```

**每一层都在"减少重复代码"，但执行模型的本质一层都没变。**

这就是为什么"先手写一遍再学框架"是最好的学习路径：
你会知道每一层封装替你省掉了什么，出问题时也知道该去查哪里。

---

## 六、自己写 Agent 时的检查清单

如果你要脱离框架、自己实现一个 Agent（或者排查框架的问题），按这个清单过一遍：

**消息层面**
- [ ] `messages` 里的消息顺序对吗？（System → Human → AI → Tool → AI）
- [ ] 每个 `tool_call` 都有对应的 `ToolMessage` 吗？`tool_call_id` 对得上吗？
- [ ] 传进模型的 messages 有没有超出上下文窗口？

**工具层面**
- [ ] `TOOL_REGISTRY` 里的名字和模型输出的 `name` 一致吗？
- [ ] 工具有没有做参数校验？（模型会传奇怪的参数）
- [ ] 工具有没有 try/except？（一个工具报错不该炸掉整个循环）
- [ ] 危险工具的权限控制做了吗？

**循环层面**
- [ ] 有最大轮数限制吗？（否则模型一直调工具就死循环）
- [ ] 退出条件是什么？（通常是"没有 tool_calls"）
- [ ] 超时/重试策略有吗？

**状态层面**
- [ ] 每轮的状态需要持久化吗？（需要 → 用 checkpointer）
- [ ] 多用户/多会话隔离了吗？（用 `thread_id`）

---

## 七、小结

| 认知 | 说明 |
|---|---|
| Agent 的本质 | 模型 + 循环 + 工具 |
| ReAct | 推理与行动交替，行动结果喂回给推理 |
| 模型不执行代码 | 它只输出 `tool_calls`，执行权在你手里 |
| `tool_call_id` | 关联请求与结果，协议强制 |
| `ToolNode` | 查表 → 执行 → 包装，就这么点事 |
| `tools_condition` | 三行判断函数 |
| 框架的价值 | 把循环变成可持久化、可中断、可观察、可并行的工程结构 |

**你已经看穿了 Agent。**
剩下要做的，是用它去解决一个真实的问题——那才是真正长本事的阶段。

**下一章**：[`08-常见问题与排错.md`](./08-常见问题与排错.md)
