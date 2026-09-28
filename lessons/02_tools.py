"""第 02 课 · 让机器人学会使用工具（ReAct 循环）

对应官方教程：https://langgraph.com.cn/tutorials/get-started/2-add-tools/index.html

===============================================================================
为什么需要工具？
===============================================================================
大模型只会"说话"，它的知识是训练时冻结进去的：
    - 不知道今天的天气
    - 算不准大数字乘法
    - 不知道你公司的内部资料
工具（Tool）就是给模型装上的"手"：让它能去查、去算、去调用外部系统。

===============================================================================
本课最核心的一张图：ReAct 循环
===============================================================================
                    ┌─────────────────────────────┐
                    ↓                             │
    START ──▶  [agent 节点]  ──有工具调用吗？──▶ [tools 节点]
                    │                             │
                    │ 没有工具调用（说明答完了）       │
                    ↓                             │
                   END  ◀─────────────────────────┘

    这个"想 → 做 → 看结果 → 再想"的循环，就是 Agent 的心脏。
    ReAct = Reasoning（推理）+ Acting（行动）。

===============================================================================
三个关键机制（代码里会详细标注）
===============================================================================
1. bind_tools  —— 把工具的"说明书"告诉模型，模型才知道有哪些工具可用。
2. tool_calls  —— 模型不是直接执行工具，而是输出一段"我要调用哪个工具、参数是什么"的 JSON。
3. ToolNode    —— LangGraph 提供的节点：负责真正执行这些工具，并把结果包成 ToolMessage。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from common import graph_viz, pretty
from common.llm import get_model

# =============================================================================
# 第 1 步：写工具
# -----------------------------------------------------------------------------
# @tool 装饰器做的事：把普通函数包装成 LangChain 的"工具对象"，它会自动
#   1. 用函数名当工具名（get_weather）
#   2. 用 **docstring** 当工具说明 —— 这段文字会被发给模型，模型靠它决定用不用、
#      什么时候用。所以 docstring 要写得像给同事看的接口说明，别偷懒！
#   3. 用函数签名 + 类型注解生成参数 schema（city: str）
# =============================================================================


@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的天气情况。当用户问天气、气温、是否下雨时使用。

    Args:
        city: 城市名称，例如"北京"、"上海"。
    """
    # 真实项目里这里会去调用天气 API；教学项目里返回假数据，
    # 这样你可以把注意力放在 LangGraph 的编排逻辑上。
    fake_db = {
        "北京": "晴，气温 25℃，微风，空气质量良",
        "上海": "多云转阴，气温 22℃，有阵雨，记得带伞",
        "深圳": "雷阵雨，气温 30℃，湿度 85%",
        "杭州": "阴，气温 20℃，适合出门",
    }
    return fake_db.get(city, "%s：晴，气温 24℃，数据来源于教学模拟" % city)


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式。当用户需要做算术、算数、计算数值时使用。

    Args:
        expression: 数学表达式，例如 "12*(3+4)"。
    """
    # 只允许数字和运算符，防止执行任意代码（安全习惯要从小处养成）
    import re

    if not re.fullmatch(r"[\d\.\+\-\*/\(\)\s]+", expression):
        return "表达式里含有不允许的字符，只能做基础四则运算。"
    try:
        # eval 在这里相对安全，因为上面已经严格限制了字符集
        return "%s = %s" % (expression, eval(expression))  # noqa: S307
    except Exception as exc:
        return "算式解析失败：%s" % exc


@tool
def search_knowledge(query: str) -> str:
    """查询公司内部知识库。当用户询问公司规定、退换货政策、产品资料、员工手册时使用。

    Args:
        query: 要查询的问题或关键词。
    """
    kb = {
        "退货": "七天无理由退货：商品需保持完好，附带全部配件与包装，联系客服提交申请即可。",
        "保修": "整机保修一年，主要部件保修两年。人为损坏不在保修范围内。",
        "发票": "下单时勾选「开具发票」，电子发票会在发货后 24 小时内发到你的邮箱。",
        "会员": "会员等级分为普通、银卡、金卡，消费每满 1000 元升一级，金卡享 9 折。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    keys = "、".join(kb.keys())
    return "知识库里没有找到相关资料。目前收录的主题有：%s" % keys


# 所有工具放进一个列表，方便统一绑定和图节点使用
TOOLS = [get_weather, calculator, search_knowledge]


# =============================================================================
# 第 2 步：定义状态
# -----------------------------------------------------------------------------
# 和第一课完全一样。工具调用过程中产生的信息（模型的 tool_calls、工具返回的
# ToolMessage）全部存在 messages 里，所以状态结构不用改。
# =============================================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =============================================================================
# 第 3 步：把工具绑定到模型上
# -----------------------------------------------------------------------------
# ★ 这是最容易误解的一步：
#   bind_tools 并不是"让模型去执行工具"。
#   它只是把工具的说明书（名字、用途、参数格式）随请求一起发给模型。
#   模型"看到"这些工具后，如果觉得该用，就会在回复里输出 tool_calls。
#   真正的执行发生在后面的 ToolNode 里。
# =============================================================================
llm = get_model(quiet=True)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: State) -> dict:
    """思考节点：把对话历史丢给模型，让它决定"直接回答"还是"调用工具"。"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    """组装 ReAct 循环图。"""
    builder = StateGraph(State)

    # --- 节点 1：agent（大脑）---
    builder.add_node("agent", agent_node)

    # --- 节点 2：tools（手脚）---
    # ToolNode 是 LangGraph 内置的节点，它做三件事：
    #   1. 从上一条 AI 消息里读出 tool_calls
    #   2. 按名字找到对应的工具函数并执行，把参数传进去
    #   3. 把每个工具的执行结果包成 ToolMessage，返回 {"messages": [...]}
    #
    # ★ 安全提醒：ToolNode 会**无条件执行**模型要求的工具。
    #   如果工具有破坏性（删数据、发邮件、转账），绝不能直接这样接！
    #   第 04 课会用 interrupt 加上"人工审批"。
    builder.add_node("tools", ToolNode(TOOLS))

    # --- 入口边 ---
    builder.add_edge(START, "agent")

    # --- 条件边：agent 之后该去哪？---
    # tools_condition 是内置的判断函数，逻辑非常朴素：
    #     最后一条消息里有 tool_calls  →  返回 "tools"
    #     否则                        →  返回 "__end__"
    # 因为它的返回值是字符串，所以要用 add_conditional_edges 注册"路由表"。
    builder.add_conditional_edges(
        "agent",
        tools_condition,  # 判断函数：读取当前状态，返回一个标签
        {
            # 标签 -> 目标节点 的映射表
            "tools": "tools",
            "__end__": END,
        },
    )

    # --- 回环边：工具执行完，回到 agent 继续思考 ---
    # ★ 这一条边就是"循环"的来源。正是因为有了它，
    #   图才能"思考 → 行动 → 再思考"，直到模型认为可以收工为止。
    builder.add_edge("tools", "agent")

    return builder.compile()


# =============================================================================
# 第 4 步：运行与观察
# =============================================================================
def run_once(app, user_input: str) -> None:
    """跑一轮，并把每次工具调用的细节打印出来。"""
    pretty.section("用户：%s" % user_input)

    for event in app.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="updates"):
        for node, update in event.items():
            if node == "agent":
                msg = update["messages"][-1]
                if getattr(msg, "tool_calls", None):
                    # 模型没有直接回答，而是要求调用工具
                    for call in msg.tool_calls:
                        print("  %s 模型决定调用工具：%s" % (pretty.green("① 思考"), pretty.bold(call["name"])))
                        print("     参数：%s" % pretty.yellow(str(call["args"])))
                else:
                    print("  %s %s" % (pretty.green("③ 回答"), msg.content))
            elif node == "tools":
                # 工具执行完，结果是 ToolMessage
                for msg in update["messages"]:
                    print("  %s 工具 %s 返回：%s" % (pretty.yellow("② 执行"), msg.name, msg.content))


def demo_manual_routing(app) -> None:
    """进阶：不用内置的 tools_condition，自己写一个判断函数。

    理解了这个，你就知道"条件边"到底是怎么工作的 —— 它只是一个普通函数。
    """
    pretty.section("拆解：tools_condition 内部到底做了什么？")

    def my_condition(state: State) -> str:
        """自己实现一个和 tools_condition 等价的判断函数。"""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"  # 有工具调用 → 去执行工具
        return "__end__"  # 没有 → 结束

    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", my_condition, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")
    my_app = builder.compile()

    print("  用自定义判断函数建的图，跑起来和内置版完全一样：")
    for event in my_app.stream({"messages": [{"role": "user", "content": "北京天气"}]}, stream_mode="updates"):
        print("     经过节点：%s" % list(event.keys()))

    pretty.info("结论：所谓「条件边」，就是「一个函数返回字符串，框架按映射表跳转」。")
    pretty.info("tools_condition 只是官方替你写好了这个函数而已，没有任何魔法。")


def interactive_loop(app) -> None:
    pretty.section("交互模式（输入 quit / exit / q 退出）")
    print(pretty.dim("  试试这些问题：北京天气怎么样 / 帮我算一下 12*(3+4) / 你们退货政策是什么"))

    # ★ 注意：这里每一轮都是独立的，历史不会累积（没有 checkpointer）。
    #   所以想连续对话，请等第 03 课。
    while True:
        user_input = pretty.safe_input(pretty.cyan("你 > "))
        if user_input is None:
            print()
            break
        if user_input.strip().lower() in ("quit", "exit", "q", ""):
            print(pretty.dim("再见！"))
            break
        run_once(app, user_input)


def main() -> None:
    pretty.lesson_header(
        "02",
        "让机器人学会使用工具（ReAct 循环）",
        goals=[
            "掌握 @tool 装饰器：docstring 就是给模型看的工具说明书",
            "理解 bind_tools 只是「告知」，真正执行工具的是 ToolNode",
            "看懂 ReAct 循环：agent ⇄ tools 的回环边是怎么转起来的",
            "学会自己写条件边判断函数（不再依赖内置 tools_condition）",
        ],
        prerequisites="第 01 课（StateGraph 三要素、reducer）",
    )

    from common.llm import model_status

    pretty.kv("当前模型", model_status())
    pretty.kv("已注册工具", "、".join(t.name for t in TOOLS))

    app = build_graph()
    graph_viz.show_structure(app, title="ReAct 智能体的图结构")

    pretty.section("开始演示")
    run_once(app, "北京今天天气怎么样？")
    run_once(app, "帮我算一下 128 * 37 + 56 等于多少")
    run_once(app, "你们公司的退货政策是什么？")
    run_once(app, "你好呀")  # 不需要工具的情况，走另一条分支

    demo_manual_routing(app)

    if pretty.is_interactive():
        interactive_loop(app)
    else:
        pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. 工具三件套：
         @tool 装饰函数      → 把函数变成模型能"看懂"的工具
         model.bind_tools()  → 把工具说明书发给模型（只是告知，不执行）
         ToolNode([...])     → 真正执行工具，结果包装成 ToolMessage

  2. ReAct 循环靠"一条回环边"实现：
         builder.add_edge("tools", "agent")

     没有这条边，图是单向的直线：模型要求调工具，执行完就结束了，
     模型永远看不到工具的返回结果。

  3. 条件边就是普通函数：
         输入 state → 返回一个字符串标签 → 框架按映射表找下一个节点
         tools_condition 只是官方写好的一个现成实现。

  4. 消息在循环里的流动顺序：
         HumanMessage → AIMessage(带 tool_calls) → ToolMessage → AIMessage(最终回答)

  5. 【安全提醒】ToolNode 会无条件执行模型要求的工具。
     涉及删除、转账、发邮件这类危险操作，必须加人工审批（第 04 课）。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 加一个新工具 get_time(城市)，返回该城市的当前时间，并在交互模式里测试它。

  ② 故意把 get_weather 的 docstring 改得含糊（比如只写"获取信息"），
     再问"北京天气"，观察模型还能不能正确挑中这个工具。
     → 结论：docstring 的质量直接决定模型选工具准不准。

  ③ 把回环边 builder.add_edge("tools", "agent") 注释掉，再问一次天气问题，
     看看图会怎么走、结果有什么不同。想清楚"为什么模型说不出答案"。
"""
    )


if __name__ == "__main__":
    main()
