"""第 04 课 · 让 Agent 动手前先问你（Human-in-the-Loop 人工介入）

对应官方教程：https://langgraph.com.cn/tutorials/get-started/4-human-in-the-loop/index.html

===============================================================================
为什么必须有这一课？
===============================================================================
第 02 课我们说过：ToolNode 会**无条件执行**模型要求的所有工具。
如果工具是"发邮件""删数据库""转账""下单"，那就是灾难：

    模型理解错一句话 → 邮件已经发出去了 → 无法撤回

这类场景的正确做法是：**在真正执行前，把决定权交回给人**。
这就是 Human-in-the-Loop（人类在环，简称 HITL）。

===============================================================================
interrupt() 到底做了什么？（原理）
===============================================================================
调用 interrupt(某个值) 时，LangGraph 会：

    1. 立刻**暂停**图的执行（抛出内部异常，节点函数不会继续往下跑）
    2. 把"某个值"作为提问内容抛给调用方
    3. 把当前完整状态**存进 checkpointer**（所以必须配 checkpointer！）
    4. 之后你调用 Command(resume=回答)，图会**从那个节点重新开始**执行，
       而 interrupt() 这一次会直接返回"你的回答"，不再暂停

    ★ 关键理解：不是"从断点继续"，而是"**重放该节点**"。
      所以中断节点里 interrupt() 之前的代码会被再执行一次 —— 有副作用的代码
      （写文件、发请求）千万别放在 interrupt() 之前。

===============================================================================
两种暂停方式
===============================================================================
    interrupt_before=["tools"]   静态断点：编译时写死，执行到该节点前必停
    interrupt({...})             动态中断：写在节点函数里，可以带上下文、可以条件触发
                                  —— 实际项目里更常用，本课重点讲它
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from common import graph_viz, pretty
from common.llm import get_model

# =============================================================================
# 第 1 步：定义工具，标出哪些是"危险操作"
# =============================================================================


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送一封电子邮件。当用户要求发邮件、发送通知给某人时使用。

    Args:
        to: 收件人邮箱地址。
        subject: 邮件主题。
        body: 邮件正文。
    """
    # 真实项目里这里会调用 SMTP 服务，一旦发出就收不回来了 —— 典型的危险操作
    return "邮件已发送至 %s，主题：%s" % (to, subject)


@tool
def query_order(order_id: str) -> str:
    """查询订单状态。当用户询问订单、物流、发货进度时使用。

    Args:
        order_id: 订单号。
    """
    return "订单 %s：已发货，预计明天送达。" % order_id


TOOLS = [send_email, query_order]

# ★ 需要人工审批的工具名单。不在这个名单里的工具会直接执行（比如查询类）。
DANGEROUS_TOOLS = {"send_email"}


# =============================================================================
# 第 2 步：状态
# -----------------------------------------------------------------------------
# 多加了一个 approved 字段，记录"人类是否批准了本次操作"。
# 它没有 reducer，所以是覆盖语义 —— 每次审批都会把上一次的结果盖掉，正合适。
# =============================================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool


llm = get_model(quiet=True).bind_tools(TOOLS)


def agent_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


# =============================================================================
# 第 3 步：审批节点 —— 本课的核心
# =============================================================================
def approval_node(state: State) -> dict:
    """检查模型想调用哪些工具；如果是危险操作，就暂停下来问人。

    节点放在 agent 和 tools 之间：
        agent → approval → tools
    相当于在"决定要做"和"真的做了"之间加了一道闸门。
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    # 没有要求调用工具 → 说明模型只是普通回复，直接放行
    if not tool_calls:
        return {"approved": True}

    # 找出其中需要审批的调用
    risky = [call for call in tool_calls if call["name"] in DANGEROUS_TOOLS]

    # 没有危险操作（比如只是查订单）→ 放行，不打扰用户
    if not risky:
        return {"approved": True}

    # ---- 有危险操作：调用 interrupt() 把图暂停，向人类提问 ----
    #
    # interrupt() 的参数可以是任何可序列化的 Python 对象，
    # 它会原样出现在调用方拿到的结果里，用来展示"到底要做什么"。
    #
    # ★★ 注意：图恢复执行时，这个节点会**从头重跑**，
    #    所以 interrupt() 之前不能有副作用代码（这里就没写）。
    decision = interrupt(
        {
            "type": "approval_request",
            "question": "Agent 想执行以下操作，是否允许？",
            "actions": [
                {"tool": call["name"], "args": call["args"], "id": call["id"]} for call in risky
            ],
        }
    )

    # ---- 图被恢复后，interrupt() 直接返回人类给的答案，继续往下走 ----
    approved = bool(decision)

    if approved:
        return {"approved": True}

    # 人类拒绝了：我们要伪造 ToolMessage 塞回消息列表。
    # 为什么必须塞？因为模型发了一条"带 tool_calls 的消息"，
    # 按照 OpenAI 的协议，后面必须跟齐每个 tool_call_id 对应的工具结果，
    # 否则下一轮请求会直接报错（消息历史不合法）。
    reject_messages = [
        ToolMessage(
            content="【人工审批未通过】用户拒绝执行 %s。请不要重试，改为向用户说明情况并询问下一步。" % call["name"],
            tool_call_id=call["id"],
            name=call["name"],
        )
        for call in risky
    ]
    return {"messages": reject_messages, "approved": False}


def route_after_approval(state: State) -> str:
    """审批之后走哪条路：批准了就去执行工具，被拒绝就回去让模型解释。"""
    return "tools" if state.get("approved") else "agent"


def build_graph(checkpointer, static_breakpoint: bool = False):
    """建图。

    参数 static_breakpoint=True 时，演示第二种暂停方式：编译期静态断点。
    """
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)

    if not static_breakpoint:
        # 动态中断方案：加一个审批节点
        builder.add_node("approval", approval_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "approval", "__end__": END},  # 注意：要去 tools 的，先经过 approval
        )
        builder.add_conditional_edges("approval", route_after_approval, {"tools": "tools", "agent": "agent"})
    else:
        # 静态断点方案：不加审批节点，靠 interrupt_before 无条件暂停
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})

    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge("tools", "agent")

    return builder.compile(
        checkpointer=checkpointer,
        # 静态断点：每次要执行 tools 节点之前都停下来（不管工具危不危险）
        interrupt_before=["tools"] if static_breakpoint else None,
    )


# =============================================================================
# 第 4 步：运行 —— 观察"暂停 → 提问 → 恢复"的完整过程
# =============================================================================


def run_with_approval(app, user_input: str, thread_id: str, human_says_yes: bool) -> None:
    """跑一轮完整的"请求 → 审批 → 执行/拒绝"流程。"""
    config = {"configurable": {"thread_id": thread_id}}

    pretty.section("用户：%s" % user_input)
    print("  %s 模拟人类审批结果：%s\n" % (pretty.dim("[演示]"), pretty.bold("同意" if human_says_yes else "拒绝")))

    # ---- 第一个阶段：跑到 interrupt 处停下 ----
    result = app.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)

    interrupt_payload = result.get("__interrupt__") if isinstance(result, dict) else None

    if not interrupt_payload:
        # 没有触发中断（比如只是普通聊天或查询类工具）
        print("  %s 本次不需要审批，图直接跑完了。" % pretty.green("✔"))
        pretty.show_messages(result["messages"], limit=4, title_text="消息列表")
        return

    # ---- 展示"人类看到了什么" ----
    request = interrupt_payload[0].value
    print("  %s 图已暂停！人类看到了这样的审批请求：" % pretty.yellow("⏸ "))
    print("     问题：%s" % pretty.bold(request["question"]))
    for action in request["actions"]:
        print("     · 工具：%s" % pretty.cyan(action["tool"]))
        print("       参数：%s" % action["args"])

    # ---- 顺便看看暂停时状态被存成什么样了 ----
    snapshot = app.get_state(config)
    print("     暂停时图的 next = %s（说明恢复后会从这里继续）" % pretty.bold(str(snapshot.next)))

    # ---- 第二个阶段：恢复执行 ----
    print()
    print("  %s 人类做出决定，图继续运行……" % pretty.yellow("▶ "))
    resumed = app.invoke(Command(resume=human_says_yes), config=config)

    pretty.show_messages(resumed["messages"], limit=6, title_text="恢复后的完整消息列表")


def demo_static_breakpoint() -> None:
    """演示静态断点 interrupt_before=['tools']。"""
    pretty.section("另一种方式：静态断点 interrupt_before=[\"tools\"]")

    app = build_graph(InMemorySaver(), static_breakpoint=True)
    config = {"configurable": {"thread_id": "static-demo"}}

    print("  编译时声明了 interrupt_before=['tools']，所以只要走到 tools 节点就会停。")
    app.invoke({"messages": [{"role": "user", "content": "帮我发一封邮件给 a@b.com，主题是周会"}]}, config=config)

    snapshot = app.get_state(config)
    print("  暂停了！next = %s" % pretty.bold(str(snapshot.next)))
    print("  这时你可以检查状态、甚至手动改状态，然后决定要不要继续。")

    # 传 None 表示"我不改任何东西，继续跑"
    result = app.invoke(None, config=config)
    print("  继续执行完毕，消息数 = %d" % len(result["messages"]))

    print()
    pretty.info("两者区别：静态断点无条件停；动态 interrupt() 可以按条件停、还能携带上下文。")
    pretty.info("生产环境通常用动态 interrupt()，因为你需要告诉审批人「到底要做什么」。")

    # 清理演示用的数据，避免影响下次运行
    snapshot.values["messages"].clear()


def interactive_loop(app) -> None:
    pretty.section("交互模式（真的由你来审批）")
    print(pretty.dim("  试试：帮我发一封邮件给 boss@example.com，主题是请假申请"))

    config = {"configurable": {"thread_id": "interactive-hitl"}}
    while True:
        user_input = pretty.safe_input(pretty.cyan("\n你 > "))
        if user_input is None:
            print()
            break
        if user_input.strip().lower() in ("quit", "exit", "q", ""):
            print(pretty.dim("再见！"))
            break

        result = app.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)

        payload = result.get("__interrupt__") if isinstance(result, dict) else None
        if payload:
            request = payload[0].value
            print()
            print(pretty.yellow("⏸ 需要你审批："))
            print("  %s" % request["question"])
            for action in request["actions"]:
                print("  · %s(%s)" % (pretty.bold(action["tool"]), action["args"]))
            answer = pretty.safe_input(pretty.cyan("  同意吗？(y/n) > "))
            if answer is None:
                break
            approved = answer.strip().lower() in ("y", "yes", "是", "同意", "1")
            result = app.invoke(Command(resume=approved), config=config)

        pretty.show_messages(result["messages"], limit=2, title_text="最新消息")


def main() -> None:
    pretty.lesson_header(
        "04",
        "让 Agent 动手前先问你（Human-in-the-Loop）",
        goals=[
            "理解 interrupt() 的暂停原理：抛异常 + 存状态 + 重放节点",
            "掌握 Command(resume=...) 恢复执行的用法",
            "学会给危险操作加审批闸门，以及拒绝后如何伪造 ToolMessage 收尾",
            "了解静态断点 interrupt_before 与动态 interrupt 的区别",
        ],
        prerequisites="第 02 课（工具调用）、第 03 课（checkpointer，本课必须依赖它）",
    )

    from common.llm import model_status

    pretty.kv("当前模型", model_status())
    pretty.kv("需要审批的工具", "、".join(sorted(DANGEROUS_TOOLS)) or "无")

    app = build_graph(InMemorySaver())
    graph_viz.show_structure(app, title="带审批闸门的图（注意 agent → approval）")

    pretty.section("开始演示")
    # 场景一：危险操作，人类同意
    run_with_approval(app, "帮我发一封邮件给 boss@example.com，主题是项目周报", "hitl-approve", True)
    # 场景二：危险操作，人类拒绝
    run_with_approval(app, "帮我发一封邮件给 all@company.com，主题是全员通知", "hitl-reject", False)
    # 场景三：安全操作，不打扰人类
    run_with_approval(app, "帮我查一下订单 A12345 的状态", "hitl-safe", True)

    demo_static_breakpoint()

    if pretty.is_interactive():
        interactive_loop(app)
    else:
        pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. Human-in-the-Loop 解决的是"AI 会做错事且无法撤销"的问题。
     凡是不可逆的操作（发消息、删数据、付款），都应该加审批。

  2. interrupt() 的三步原理：
         ① 抛出内部异常让图停下      ② 把状态存进 checkpointer
         ③ 恢复时**重放该节点**，interrupt() 直接返回人类给的答案

     ★ 推论：interrupt() 之前千万别写有副作用的代码，否则会被重复执行。

  3. 恢复执行用 Command(resume=值)，这个值就是 interrupt() 的返回值。

  4. 拒绝之后必须伪造 ToolMessage 补上 tool_call_id，
     否则消息历史不合法，下一轮请求会报错。

  5. 两种暂停方式：
         interrupt_before=["节点"]   静态、无条件，适合"每步都要人确认"的调试场景
         interrupt({...})            动态、可带上下文、可条件触发，生产环境首选

  6. 前提条件：必须配 checkpointer，否则没法保存暂停时的状态。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 给审批加一个"修改参数"的能力：中断返回的不是布尔值，而是一个新参数字典，
     然后用 Command(resume={"to": "新邮箱"}) 恢复，让工具用修改后的参数执行。
     （提示：在 approval_node 里把人类返回的值覆盖到 tool_calls 上）

  ② 再加一个危险工具 delete_account(user_id)，观察它会不会也被拦下来。

  ③ 思考题：如果审批要等三个小时，程序能一直挂在内存里等吗？
     想清楚这一点，你就明白为什么 LangGraph 要把状态存进数据库了。
"""
    )


if __name__ == "__main__":
    main()
