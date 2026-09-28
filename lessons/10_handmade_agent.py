"""第 10 课 · 手写一个 Agent（把封装全部拆开）

===============================================================================
这是整套课程的"毕业课"。
===============================================================================
前九课我们用过不少现成的轮子：
    ToolNode          —— 自动执行工具
    tools_condition   —— 自动判断"还要不要调工具"
    create_agent      —— 一行代码造一个 Agent（LangChain 1.x 提供）

它们很好用，但用久了容易产生一种错觉："Agent 是个很玄的东西"。

这一课我们把它们**全部拆掉**，只用手动写的 Python 代码从头搭一遍。
搭完之后你会发现：Agent 一点都不神秘，它就是

    一个 while 循环 + 一个消息列表 + 几个工具函数

仅此而已。剩下的都是工程细节。

===============================================================================
手写之后，你会彻底搞懂这四件事
===============================================================================
1. bind_tools 到到底给模型发了什么？（答案是：一份 JSON Schema 说明书）
2. 模型怎么"要求"调用工具？（答案是：不调用，只是输出一段结构化 JSON）
3. 谁真正执行了工具？（答案是：我们的代码，模型碰不到你的函数）
4. 循环什么时候停？（答案是：模型不再输出 tool_calls 的时候）
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common import graph_viz, pretty
from common.llm import get_model

llm = get_model(quiet=True, force_offline=True)  # 这一课强制用离线模型，保证结果可复现


# =============================================================================
# 第 1 步：定义工具
# -----------------------------------------------------------------------------
# 这里故意用一个最简单的字典来当"工具注册表"。
# 框架里的 ToolNode 内部，做的事和下面这个字典一模一样：
#     按名字找到函数 → 把参数传进去 → 拿到返回值
# =============================================================================


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。用户问天气、气温时使用。

    Args:
        city: 城市名称。
    """
    return "%s：晴，25℃" % city


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。用户需要算数时使用。

    Args:
        expression: 数学表达式，例如 1+2*3。
    """
    import re

    if not re.fullmatch(r"[\d\.\+\-\*/\(\)\s]+", expression):
        return "表达式含有非法字符"
    try:
        return "%s = %s" % (expression, eval(expression))  # noqa: S307
    except Exception as exc:
        return "计算失败：%s" % exc


TOOLS = [get_weather, calculator]

# ★ 工具注册表：名字 → 工具对象。这就是 ToolNode 内部最重要的一张表。
TOOL_REGISTRY = {t.name: t for t in TOOLS}


def show_what_model_receives() -> None:
    """打印"工具的说明书"长什么样 —— 这就是 bind_tools 真正发给模型的东西。"""
    pretty.section("① bind_tools 到底给模型发了什么？")

    for t in TOOLS:
        print("  工具：%s" % pretty.bold(t.name))
        print("      描述：%s" % t.description.split("\n")[0])
        # args_schema 是由函数签名 + 类型注解 + docstring 自动生成的
        schema = t.args_schema.model_json_schema()
        print("      参数结构（发给模型的 JSON Schema）：")
        print("      %s" % json.dumps(schema, ensure_ascii=False, indent=8)[:400])
        print()

    pretty.info("模型看到的就是这三样：名字、描述、参数结构。它靠这些判断该不该用、怎么填参数。")
    pretty.info("所以 docstring 写得清不清楚，直接决定模型会不会用错工具。")


# =============================================================================
# 第 2 步：手工写"思考节点"
# =============================================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm_with_tools = llm.bind_tools(TOOLS)


def agent_think(state: State) -> dict:
    """思考：把历史交给模型，让它产出"要么回答、要么要求调工具"。"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# =============================================================================
# 第 3 步：手工写"工具节点"（ToolNode 的等价实现）
# =============================================================================
def my_tool_node(state: State) -> dict:
    """执行模型要求的工具。**这个函数就是 ToolNode 的核心逻辑。**

    ToolNode 内部大概也就这么多代码，只是多了并发、错误处理、缓存等细节。
    """
    last_message = state["messages"][-1]

    # 模型把"想调用的工具"写在 tool_calls 里，结构是：
    #     [{"name": "get_weather", "args": {"city": "北京"}, "id": "call_abc"}, ...]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    print("      %s 模型要求调用 %d 个工具" % (pretty.dim("[手工工具节点]"), len(tool_calls)))

    outputs: list[ToolMessage] = []

    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        call_id = call["id"]
        print("        → %s(%s)" % (pretty.bold(name), args))

        # ① 查注册表
        tool_obj = TOOL_REGISTRY.get(name)
        if tool_obj is None:
            content = "错误：不存在名为 %s 的工具" % name
        else:
            # ② 真正执行（★ 记住：是你的代码在跑，模型只能"请求"）
            try:
                content = str(tool_obj.invoke(args))
            except Exception as exc:
                content = "工具执行失败：%s" % exc

        # ③ 把结果包成 ToolMessage。
        #    tool_call_id 必须和请求里的 id 对上 —— 这是协议的硬性要求，
        #    因为模型可能一次要求调用多个工具，它要靠 id 分清哪个结果对应哪个请求。
        outputs.append(ToolMessage(content=content, tool_call_id=call_id, name=name))

    return {"messages": outputs}


# =============================================================================
# 第 4 步：手工写"判断函数"（tools_condition 的等价实现）
# =============================================================================
def my_condition(state: State) -> str:
    """判断下一步去哪：还有工具要调用就继续循环，否则结束。"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


def build_manual_graph():
    """用手写组件组装出来的 Agent —— 和 ToolNode/tools_condition 版本完全等价。"""
    return (
        StateGraph(State)
        .add_node("agent", agent_think)
        .add_node("tools", my_tool_node)  # ★ 手工版
        .add_edge(START, "agent")
        .add_conditional_edges("agent", my_condition, {"tools": "tools", "end": END})  # ★ 手工版
        .add_edge("tools", "agent")
        .compile()
    )


# =============================================================================
# 第 5 步：干脆连 LangGraph 都不要了 —— 纯 while 循环版
# =============================================================================
def bare_python_agent(question: str, max_rounds: int = 5) -> str:
    """不用 LangGraph、不用任何封装，一个 while 循环搞定。

    ★★ 这是本课最重要的一段代码。看完它你就彻底明白：
       所谓 Agent，就是一个"模型说话 → 我们执行工具 → 把结果告诉模型"的循环。
       LangGraph 帮你把这套循环做成了可持久化、可中断、可观测、可并行的工程结构。
    """
    pretty.section("③ 极致简化版：纯 Python while 循环实现的 Agent")

    messages: list[BaseMessage] = [HumanMessage(content=question)]
    print("  用户：%s" % question)

    for round_no in range(1, max_rounds + 1):
        # ---- 1) 问模型 ----
        reply = llm_with_tools.invoke(messages)
        messages.append(reply)

        tool_calls = getattr(reply, "tool_calls", None) or []

        # ---- 2) 没有工具调用 → 说明它答完了，跳出循环 ----
        if not tool_calls:
            print("  第 %d 轮：模型直接回答，循环结束" % round_no)
            return str(reply.content)

        # ---- 3) 有工具调用 → 执行，把结果塞回消息列表，继续下一轮 ----
        print("  第 %d 轮：模型要求调用 %d 个工具" % (round_no, len(tool_calls)))
        for call in tool_calls:
            tool_obj = TOOL_REGISTRY.get(call["name"])
            result = str(tool_obj.invoke(call["args"])) if tool_obj else "工具不存在"
            print("        %s(%s) → %s" % (call["name"], call["args"], result))
            messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))

    return "达到最大轮数仍未结束"


# =============================================================================
# 第 6 步：验证"消息协议"确实是硬性要求
# =============================================================================
def demo_invalid_message_protocol() -> None:
    """演示：如果不给 ToolMessage 补上 tool_call_id，会怎样。

    很多人第一次遇到这个报错都会懵，这里主动把它复现一遍。
    """
    pretty.section("④ 为什么 ToolMessage 必须带 tool_call_id？")

    print("  规则：模型发了一条带 tool_calls 的消息后，后面必须跟着")
    print("        每个 tool_call_id 对应的 ToolMessage，一个都不能少。")
    print()

    # 构造一段"缺了工具结果"的对话
    bad_messages = [
        HumanMessage(content="北京天气"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "call_xyz", "type": "tool_call"}],
        ),
        # 故意不加 ToolMessage
        HumanMessage(content="你刚才查到了什么？"),
    ]

    try:
        llm.invoke(bad_messages)
        pretty.warn("离线模拟模型没校验协议，所以没报错。")
        pretty.info("换成真实的 OpenAI / Anthropic 接口，这里会直接返回 400 错误。")
    except Exception as exc:
        print("  报错了：%s" % type(exc).__name__)

    print()
    print("  正确的写法是补上：")
    print("      ToolMessage(content='北京：晴，25℃', tool_call_id='call_xyz', name='get_weather')")
    print()
    pretty.info("这就是为什么第 04 课「人工拒绝」之后，也要伪造一条 ToolMessage —— 否则协议不完整。")


# =============================================================================
# 第 7 步：对比三种写法
# =============================================================================
def demo_comparison() -> None:
    pretty.section("⑤ 三种写法对比：你现在能看懂封装里发生了什么")

    print(pretty.bold("  写法一：纯 while 循环（本课第 5 步）"))
    print("      约 20 行，零依赖。缺点：无法持久化、无法中断、无法并行、出错就全丢。")
    print()
    print(pretty.bold("  写法二：LangGraph 手写版（本课第 3、4 步）"))
    print("      自己写工具节点和判断函数，但用图来组织。")
    print("      优点：能加 checkpointer、能中断、能并行、能观察每一步。")
    print()
    print(pretty.bold("  写法三：用官方封装（第 02 课的写法）"))
    print("      ToolNode(TOOLS) + tools_condition —— 就是写法二里那两个函数的官方实现。")
    print("      再往上还有一行搞定的 create_agent：")
    print()

    # 真正跑一个 create_agent（LangChain 1.x 的官方 Agent 构造器）
    try:
        from langchain.agents import create_agent

        agent = create_agent(llm, tools=TOOLS)
        result = agent.invoke({"messages": [HumanMessage(content="上海天气怎么样")]})
        print("      agent = create_agent(llm, tools=TOOLS)")
        print("      agent.invoke({...})")
        print("      → 内部结构：%s" % pretty.green("和本课手写的图几乎一模一样"))
        print("      → 最终消息数：%d" % len(result["messages"]))
        for m in result["messages"]:
            print("         %s | %s" % (type(m).__name__, str(m.content)[:60]))
    except Exception as exc:
        pretty.warn("create_agent 演示失败（不影响本课其它内容）：%s" % exc)

    print()
    pretty.info("结论：create_agent 不是魔法，它内部就是本课第 3、4 步那两个函数的加强版。")
    pretty.info("现在你可以放心用封装了 —— 因为你已经知道里面在干什么，出问题也知道去哪查。")


def main() -> None:
    pretty.lesson_header(
        "10",
        "手写一个 Agent（把封装全部拆开）",
        goals=[
            "看清 bind_tools 真正发给模型的东西（JSON Schema 说明书）",
            "亲手实现 ToolNode 和 tools_condition，理解它们到底做了什么",
            "用纯 while 循环写出一个能跑的最小 Agent",
            "理解 tool_call_id 消息协议，知道报错时该往哪查",
        ],
        prerequisites="第 02 课（ReAct 循环），建议前九课都已学过",
    )

    show_what_model_receives()

    manual_app = build_manual_graph()
    graph_viz.show_structure(manual_app, title="完全用手写组件搭出的 Agent 图")

    pretty.section("② 运行手写版 Agent")
    for question in ["北京天气怎么样", "帮我算一下 99*88"]:
        print()
        print("  用户：%s" % pretty.cyan(question))
        result = manual_app.invoke({"messages": [HumanMessage(content=question)]})
        print("  最终回答：%s" % pretty.green(str(result["messages"][-1].content).replace("\n", " ")[:120]))

    answer = bare_python_agent("北京天气怎么样")
    print("  回答：%s" % pretty.green(answer))

    print()
    answer2 = bare_python_agent("帮我算一下 123*456 等于多少")
    print("  回答：%s" % pretty.green(answer2))

    demo_invalid_message_protocol()
    demo_comparison()

    pretty.section("毕业寄语")
    print(
        """
  恭喜你学完了全部十课！现在回头看，一条 Agent 的完整技术链路是：

     第 01 课  StateGraph        —— 把程序写成图
     第 02 课  工具调用           —— 让模型能动手
     第 03 课  Checkpointer      —— 让状态能持久
     第 04 课  interrupt          —— 让人能管住它
     第 05 课  自定义状态          —— 让数据有结构
     第 06 课  流式               —— 让体验能接受
     第 07 课  分支/并行/Send     —— 让流程能复杂
     第 08 课  子图/时间旅行       —— 让代码能复用、能回滚
     第 09 课  多智能体           —— 让分工能落地
     第 10 课  手写循环           —— 让这一切不再神秘

  接下来该做什么？

     1. 打开 capstone/ 目录，看一个把这些能力全部串起来的完整项目。
     2. 回去把每节课的「动手练习」做完 —— 那才是真正长本事的地方。
     3. 挑一个你自己的真实需求（比如自动整理会议纪要、批量处理表格），
        用这套框架做出来。做出来一个，比看十个教程都有用。
"""
    )


if __name__ == "__main__":
    main()
