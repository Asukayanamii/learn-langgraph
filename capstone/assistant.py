"""综合实战 · 命令行智能助理

===============================================================================
这是整套课程的"毕业设计"：把前九课的能力拼成一个真正能用的东西。
===============================================================================

它具备的能力，以及各自对应哪一课：

    ┌──────────────────────┬──────────────────────────────────────────┐
    │ 能力                  │ 用到的知识                                │
    ├──────────────────────┼──────────────────────────────────────────┤
    │ 会聊天、会用工具       │ 第 01、02 课  StateGraph + ReAct 循环     │
    │ 记得住你说过的话       │ 第 03 课      SqliteSaver 持久化          │
    │ 危险操作先问你         │ 第 04 课      interrupt 人工审批          │
    │ 有结构化的内部状态     │ 第 05 课      自定义 State + reducer      │
    │ 打字机式的输出         │ 第 06 课      stream_mode="messages"     │
    │ 会话隔离与切换         │ 第 03 课      thread_id                  │
    └──────────────────────┴──────────────────────────────────────────┘

运行方式：

    python capstone/assistant.py                  # 自动演示（推荐先看这个）
    python capstone/assistant.py --interactive    # 真正和它对话

交互模式里可用的命令：

    /help      查看帮助
    /new       开一个新会话（旧会话的记忆仍然保存在数据库里）
    /sessions  列出所有历史会话
    /history   查看当前会话的消息
    /state     查看当前完整状态（含统计数据）
    /tools     查看所有工具
    /quit      退出
"""

from __future__ import annotations

import operator
import sys
from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from capstone.tools import ALL_TOOLS, DANGEROUS_TOOLS, TOOL_REGISTRY, tool_catalog
from common import pretty
from common.config import DATA_DIR
from common.llm import get_model, model_status


# =============================================================================
# 一、状态设计（第 05 课）
# -----------------------------------------------------------------------------
# 除了消息历史，我们还记录一些"运行统计"，方便观察这个助理干了多少活。
# =============================================================================
class AssistantState(TypedDict):
    # 完整对话历史
    messages: Annotated[list[BaseMessage], add_messages]

    # 一共调用了几次工具（用 operator.add 累加）
    tool_call_count: Annotated[int, operator.add]

    # 已经执行过的工具名（去重追加，方便展示"这个助理用过哪些能力"）
    used_tools: Annotated[list, lambda old, new: list(dict.fromkeys((old or []) + (new or [])))]

    # 本次操作是否被批准（覆盖语义）
    approved: bool


# =============================================================================
# 二、节点实现
# =============================================================================
llm = get_model(quiet=True)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


SYSTEM_PROMPT = """你是一个中文智能助理，可以调用工具帮用户解决问题。

工作原则：
1. 需要实时信息（时间、天气）或精确计算时，必须调用工具，不要凭记忆回答。
2. 涉及公司政策的问题，先查知识库。
3. 用户让你记录、查看、完成待办时，使用 todo 工具。
4. 发邮件是不可撤销的操作，调用前系统会向用户确认。
5. 回答简洁、直接、中文。
"""


def agent_node(state: AssistantState) -> dict:
    """大脑：决定直接回答还是调用工具。

    ★ 注意 SystemMessage 是**临时拼上去**的，不会被存进状态。
      这样做的原因：系统提示词是"代码的一部分"，它不该随着对话历史
      一遍遍存进数据库（既浪费空间，也容易和旧记录冲突）。
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def approval_node(state: AssistantState) -> dict:
    """闸门：危险操作先问人（第 04 课）。

    放在 agent 和 tools 之间：
        - 模型只想普通回答        → 直接放行
        - 只调用安全工具（查天气）  → 直接放行，不打扰用户
        - 要调用危险工具（发邮件）  → interrupt 暂停，等用户拍板
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return {"approved": True}

    risky = [c for c in tool_calls if c["name"] in DANGEROUS_TOOLS]
    if not risky:
        return {"approved": True}

    # 暂停并把"要做什么"告诉人类
    decision = interrupt(
        {
            "message": "这个操作不可撤销，需要你确认：",
            "actions": [{"tool": c["name"], "args": c["args"], "id": c["id"]} for c in risky],
        }
    )

    if decision:
        return {"approved": True}

    # 被拒绝：补上 ToolMessage，让消息历史保持合法，同时告诉模型别重试
    rejected = [
        ToolMessage(
            content="【用户拒绝】该操作已被用户否决，请不要重试，改为询问用户下一步想怎么做。",
            tool_call_id=c["id"],
            name=c["name"],
        )
        for c in risky
    ]
    return {"messages": rejected, "approved": False}


def count_tools_node(state: AssistantState) -> dict:
    """统计节点：接在 tools 后面，记录这一轮用到了哪些工具。

    技巧：tools 节点刚刚执行完，所以 ToolMessage 一定在消息列表的**末尾**。
    从后往前数，遇到第一个非 ToolMessage 就停 —— 这样拿到的正好是"本轮的工具调用"。
    """
    names: list[str] = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            names.append(msg.name or "unknown")
        else:
            break
    names.reverse()

    # tool_call_count 用的是 operator.add，所以这里返回几，总数就加几
    return {"tool_call_count": len(names), "used_tools": names}


def route_after_approval(state: AssistantState) -> str:
    return "tools" if state.get("approved") else "agent"


def build_assistant(checkpointer):
    """组装完整图。

         START
           ↓
        [agent] ──有工具调用──▶ [approval] ──批准──▶ [tools] ──▶ 回到 agent
           │                        │
           │ 没有调用                │ 被拒绝
           ↓                        ↓
          END                    [agent]
    """
    return (
        StateGraph(AssistantState)
        .add_node("agent", agent_node)
        .add_node("approval", approval_node)
        .add_node("tools", ToolNode(ALL_TOOLS))
        .add_node("stats", count_tools_node)
        .add_edge(START, "agent")
        .add_conditional_edges("agent", tools_condition, {"tools": "approval", "__end__": END})
        .add_conditional_edges("approval", route_after_approval, {"tools": "tools", "agent": "agent"})
        .add_edge("tools", "stats")  # 工具执行完先记一笔统计
        .add_edge("stats", "agent")  # 再回到大脑继续思考
        .compile(checkpointer=checkpointer)
    )


# =============================================================================
# 三、会话交互层
# =============================================================================
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}


def ask(app, text: str, thread_id: str, approver=None, show_tokens: bool = True) -> str:
    """向助理提问，一边流式显示一边处理人工审批。

    参数：
        approver  一个函数：收到审批请求，返回 True/False。
                  传 None 表示"自动批准"（自动演示时用）。
    返回：助理最后的完整回答文本。
    """
    inputs: object = {"messages": [HumanMessage(content=text)]}
    config = _config(thread_id)
    final_text = ""

    while True:
        interrupt_payload = None
        buffer: list[str] = []

        # ★ stream_mode 传列表，可以同时拿到 token 流和节点级事件（第 06 课）
        for mode, chunk in app.stream(inputs, config=config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                message_chunk, metadata = chunk
                # 只显示 agent 节点产生的文本，避免把工具结果也混进来
                if metadata.get("langgraph_node") == "agent" and message_chunk.content:
                    text_piece = str(message_chunk.content)
                    buffer.append(text_piece)
                    if show_tokens:
                        print(text_piece, end="", flush=True)
            else:  # mode == "updates"
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    interrupt_payload = chunk["__interrupt__"]
                elif isinstance(chunk, dict):
                    for node in chunk:
                        if node == "tools" and show_tokens:
                            print(pretty.dim("\n  [工具执行完毕]"), end="", flush=True)

        if buffer:
            final_text = "".join(buffer)

        # ---- 没有中断 → 本轮结束 ----
        if interrupt_payload is None:
            break

        # ---- 有中断 → 问人类，然后带着答案继续跑 ----
        request = interrupt_payload[0].value
        if approver is None:
            approved = True  # 自动演示：默认同意
        else:
            approved = approver(request)

        if show_tokens:
            print(pretty.yellow("\n  [人工审批] 结果：%s" % ("同意" if approved else "拒绝")))
        inputs = Command(resume=approved)

    return final_text


# =============================================================================
# 四、自动演示（默认模式）
# =============================================================================
def run_demo(app, thread_id: str = "demo") -> None:
    """跑一段脚本化的对话，展示助理的全部能力。"""

    def auto_approve(request) -> bool:
        """自动审批：打印出请求内容，然后同意。"""
        print(pretty.yellow("\n  ⏸ 需要审批：%s" % request["message"]))
        for action in request["actions"]:
            print("      工具：%s" % pretty.bold(action["tool"]))
            for k, v in action["args"].items():
                print("        %s = %s" % (k, v))
        print(pretty.dim("      （演示模式：自动同意）"))
        return True

    def auto_reject(request) -> bool:
        print(pretty.yellow("\n  ⏸ 需要审批：%s" % request["message"]))
        for action in request["actions"]:
            print("      工具：%s" % pretty.bold(action["tool"]))
        print(pretty.dim("      （演示模式：自动拒绝）"))
        return False

    scenarios = [
        ("现在几点了？", None),
        ("帮我算一下 (1280 - 320) * 0.85 等于多少", None),
        ("你们会员卡怎么升级？", None),
        ("记一下：周五下午三点开产品评审会", None),
        ("再记一下：买牛奶", None),
        ("帮我看看我的待办", None),
        ("把买牛奶那条待办删掉", None),  # todo 的 delete 不在危险名单里，会直接执行
        ("帮我发一封邮件给 client@example.com，主题是项目进度同步，内容是本周已完成联调", auto_approve),
        ("再发一封邮件给 all@company.com，主题是紧急通知", auto_reject),
        ("我叫什么名字来着？", None),  # 演示记忆
    ]

    for question, approver in scenarios:
        print()
        print(pretty.cyan("你 > ") + question)
        print(pretty.green("AI > "), end="", flush=True)
        ask(app, question, thread_id, approver=approver)
        print()

    # 展示最终状态
    snapshot = app.get_state(_config(thread_id))
    pretty.section("本次会话结束时的状态")
    values = snapshot.values
    print("  消息总数        : %d" % len(values.get("messages", [])))
    print("  工具调用轮次    : %d" % values.get("tool_call_count", 0))
    print("  用过的工具      : %s" % "、".join(values.get("used_tools", [])) or "无")


# =============================================================================
# 五、交互模式
# =============================================================================
HELP_TEXT = """
  可用命令：
    /help      查看帮助
    /new       开始一个新会话
    /sessions  列出所有历史会话
    /history   查看当前会话的消息
    /state     查看当前完整状态
    /tools     查看所有工具
    /quit      退出

  或者直接说人话，比如：
    现在几点了？
    帮我算一下 128*37
    你们退货政策是什么
    记一下：明天上午十点开会
    帮我看看我的待办
    帮我发邮件给 a@b.com，主题是打招呼
"""


def print_tools() -> None:
    pretty.section("可用工具")
    for name, desc, dangerous in tool_catalog():
        flag = pretty.red(" [需要审批]") if dangerous else ""
        print("  %-18s %s%s" % (pretty.bold(name), desc, flag))


def interactive(app, thread_id: str) -> None:
    """交互模式：真的和这个助理聊天。"""
    pretty.section("交互模式")
    print(pretty.dim(HELP_TEXT))
    if not model_status().startswith("离线"):
        print(pretty.info("提示：可以试试让它查天气、算数、记待办、发邮件。"))
    print(pretty.info("提示：发邮件会让你确认 —— 这就是 Human-in-the-Loop。"))

    while True:
        try:
            raw = pretty.safe_input(pretty.cyan("\n你 > "))
        except KeyboardInterrupt:
            break
        if raw is None:
            break

        text = raw.strip()
        if not text:
            continue

        # ---- 内置命令 ----
        if text.startswith("/"):
            command = text.lower()
            if command in ("/quit", "/exit", "/q"):
                break
            if command == "/help":
                print(pretty.dim(HELP_TEXT))
                continue
            if command == "/tools":
                print_tools()
                continue
            if command == "/new":
                thread_id = "session-%d" % (len(list_sessions(app)) + 1)
                print(pretty.ok("已开启新会话：%s（旧会话的记忆仍保存在数据库里）" % thread_id))
                continue
            if command == "/sessions":
                print(pretty.section("历史会话"))
                for tid in list_sessions(app):
                    print("  · %s" % tid)
                continue
            if command == "/history":
                snapshot = app.get_state(_config(thread_id))
                pretty.show_messages(snapshot.values.get("messages", []), limit=10, title_text="会话 %s" % thread_id)
                continue
            if command == "/state":
                snapshot = app.get_state(_config(thread_id))
                for k, v in snapshot.values.items():
                    if k == "messages":
                        print("  messages = [%d 条]" % len(v))
                    else:
                        print("  %s = %s" % (k, v))
                continue
            print(pretty.warn("未知命令：%s（输入 /help 查看帮助）" % text))
            continue

        # ---- 正常对话 ----
        def approver(request, _self=None) -> bool:
            print()
            print(pretty.yellow("⏸ 需要你确认：%s" % request["message"]))
            for action in request["actions"]:
                print("    工具：%s" % pretty.bold(action["tool"]))
                for k, v in action["args"].items():
                    print("      %s = %s" % (k, v))
            answer = pretty.safe_input(pretty.cyan("  同意执行吗？(y/n) > "))
            return (answer or "").strip().lower() in ("y", "yes", "是", "同意", "1")

        print(pretty.green("\nAI > "), end="", flush=True)
        try:
            ask(app, text, thread_id, approver=approver)
        except Exception as exc:
            print(pretty.err("\n出错了：%s: %s" % (type(exc).__name__, exc)))
        print()


def list_sessions(app) -> list[str]:
    """从 checkpointer 里读出所有会话 ID。

    这里直接查 SQLite 的 checkpoints 表 —— 顺便让你看到
    "记忆"最终是怎么落盘的（第 03 课的知识）。
    """
    try:
        rows = app.checkpointer.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        return sorted(row[0] for row in rows)
    except Exception:
        return []


# =============================================================================
# 六、入口
# =============================================================================
def main() -> None:
    pretty.title("综合实战 · 命令行智能助理")

    db_path = DATA_DIR / "assistant.sqlite"
    print()
    pretty.kv("模型", model_status())
    pretty.kv("记忆库", str(db_path))
    pretty.kv("工具数", "%d 个（其中 %d 个需要审批）" % (len(ALL_TOOLS), len(DANGEROUS_TOOLS)))
    print()
    pretty.info("这个助理 = 第 01～06 课的能力组合：ReAct 循环 + SQLite 记忆 + 人工审批 + 流式输出。")

    # SqliteSaver 让记忆落在磁盘上：关掉程序再打开，它还记得你
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = build_assistant(checkpointer)

        if pretty.is_interactive():
            interactive(app, "session-1")
            print(pretty.dim("\n再见！下次运行我还会记得你。"))
        else:
            run_demo(app)
            print()
            pretty.try_hint()
            pretty.info("提示：上面这段演示里的所有能力，都能在 lessons/ 里找到对应的课程。")
            print()
            print(
                """
  想把它变成自己的项目？可以从这几个方向改：
      1. 换工具   → 改 capstone/tools.py，接你自己的业务 API
      2. 换人设   → 改 assistant.py 里的 SYSTEM_PROMPT
      3. 加审批   → 把更多工具名加进 DANGEROUS_TOOLS
      4. 换存储   → 把 SqliteSaver 换成 PostgresSaver，就能多实例部署
      5. 换成服务 → 把 ask() 包成一个 HTTP 接口，前端就能用（记得用 astream 异步版）
"""
            )


if __name__ == "__main__":
    main()
