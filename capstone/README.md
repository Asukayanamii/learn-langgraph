# 综合实战 · 命令行智能助理

把前十课的能力拼成一个**真正能用**的东西。

---

## 它长什么样

```
$ python capstone/assistant.py --interactive

你 > 现在几点了？
AI > 现在是 2026-09-12 10:38:09，星期六。

你 > 你们会员卡怎么升级？
AI > 会员分为普通、银卡、金卡三级，消费每满 1000 元升一级，金卡享受 9 折。

你 > 记一下：周五下午三点开产品评审会
AI > 已添加待办：周五下午三点开产品评审会。当前共 1 条。

你 > 帮我发一封邮件给 client@example.com，主题是项目进度同步
AI >
⏸ 需要你确认：这个操作不可撤销，需要你确认：
    工具：send_email
      to = client@example.com
      subject = 项目进度同步
      body = 帮我发一封邮件给 client@example.com，主题是项目进度同步
  同意执行吗？(y/n) > y
AI > 邮件已发送…
```

---

## 它用到了哪些课的知识

| 能力 | 实现位置 | 对应课程 |
|---|---|---|
| ReAct 循环（会聊天、会用工具） | `build_assistant()` | 第 01、02 课 |
| 记忆（关掉程序还记得你） | `SqliteSaver` | 第 03 课 |
| 危险操作人工审批 | `approval_node()` | 第 04 课 |
| 多字段状态 + 自定义 reducer | `AssistantState` | 第 05 课 |
| 打字机式输出 | `ask()` 里的 `stream_mode=["messages","updates"]` | 第 06 课 |
| 工具清单可插拔 | `tools.py` 的 `ALL_TOOLS` | 第 02 课 |

**这六样东西组合起来，就是一个真实项目该有的样子。**

---

## 怎么运行

```bash
# 自动演示（推荐先看这个，30 秒看完所有能力）
python capstone/assistant.py

# 真正和它对话
python capstone/assistant.py --interactive
```

不需要 API Key——项目内置的离线模拟模型会接管全部模型调用。

---

## 交互模式命令

| 命令 | 作用 |
|---|---|
| `/help` | 查看帮助 |
| `/new` | 开一个新会话（旧会话记忆仍保存在数据库里） |
| `/sessions` | 列出所有历史会话 |
| `/history` | 查看当前会话的消息 |
| `/state` | 查看当前完整状态（含统计数据） |
| `/tools` | 查看所有工具 |
| `/quit` | 退出 |

**试试这些说法：**

```
现在几点了？
帮我算一下 (1280 - 320) * 0.85
你们退货政策是什么
记一下：明天上午十点开会
帮我看看我的待办
帮我发邮件给 a@b.com，主题是打招呼
```

---

## 架构图

```
                            START
                              │
                              ▼
                     ┌─────────────────┐
              ┌─────▶│      agent      │  大脑：决定回答还是调工具
              │      └────────┬────────┘
              │               │
              │      有工具调用吗？
              │        ┌──────┴──────┐
              │        没有          有
              │        │             │
              │        ▼             ▼
              │       END    ┌──────────────┐
              │              │   approval   │  闸门：危险操作先问人
              │              └──────┬───────┘
              │                     │
              │              批准了吗？
              │           ┌─────────┴─────────┐
              │         拒绝                 批准
              │           │                   │
              └───────────┤                   ▼
                          │          ┌──────────────┐
                          │          │    tools     │  执行工具
                          │          └──────┬───────┘
                          │                 ▼
                          │          ┌──────────────┐
                          │          │    stats     │  记录统计
                          │          └──────┬───────┘
                          │                 │
                          └─────────────────┘
```

三个值得注意的设计：

1. **`approval` 插在 `agent` 和 `tools` 之间**——这是人工审批能生效的关键位置
2. **`stats` 节点**——演示"在 ReAct 循环里插入辅助节点"，记录用过哪些工具
3. **拒绝后回到 `agent`**（而不是 END）——让模型有机会解释并询问用户下一步

---

## 改成你自己的项目

### 1. 换工具

编辑 `tools.py`，按同样的模式加函数：

```python
@tool
def query_order(order_id: str) -> str:
    """查询订单状态。当用户询问订单、物流、发货进度时使用。

    Args:
        order_id: 订单号。
    """
    return real_api_call(order_id)      # 接你自己的业务
```

然后把它加进 `ALL_TOOLS`。**就这么简单**——不需要改任何图代码。

### 2. 换人设

编辑 `assistant.py` 里的 `SYSTEM_PROMPT`。

### 3. 加审批

把工具名加进 `DANGEROUS_TOOLS`：

```python
DANGEROUS_TOOLS = {"send_email", "delete_account", "transfer_money"}
```

### 4. 换存储

```python
# 单机
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("data/assistant.sqlite") as cp:
    app = build_assistant(cp)

# 生产（多实例共享）
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string("postgresql://...") as cp:
    app = build_assistant(cp)
```

**业务代码一行都不用改**——因为 checkpointer 是可替换的接口。

### 5. 变成一个 Web 服务

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

@app.post("/chat")
async def chat(user_input: str, session_id: str):
    async def generate():
        async for chunk, meta in app.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            if chunk.content:
                yield chunk.content
    return EventSourceResponse(generate())
```

注意点：
- 用 `astream`（异步版本）
- `thread_id` 用前端传来的 `session_id`，这样多用户天然隔离
- 人工审批需要通过 WebSocket 或额外的接口往返（把 `interrupt` 的 payload 发给前端，
  用户点了"同意"再调 `Command(resume=True)`）

---

## 已知的简化处理

这是个教学项目，为了让代码保持可读，有几处做了简化：

| 简化 | 生产环境该怎么做 |
|---|---|
| 工具数据存在内存字典里 | 存数据库 |
| `send_email` 只返回字符串 | 真的调 SMTP / 邮件服务 |
| 没有做输入校验和限流 | 加 Pydantic 校验 + 限流 |
| 没有日志和监控 | 接 LangSmith + 结构化日志 |
| 审批只支持"同意/拒绝" | 支持"修改参数后再执行" |

**但架构是对的**——把这些简化补上，它就是一个可以上线的系统。
