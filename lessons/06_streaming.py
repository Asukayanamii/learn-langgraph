"""第 06 课 · 流式输出（把"等 10 秒"变成"立刻开始打字"）

对应官方教程中的 streaming 部分。

===============================================================================
为什么要流式？
===============================================================================
不流式：
    用户按下回车 → 界面卡住 → 10 秒后突然蹦出一大段字
    （用户会以为程序死了）

流式：
    用户按下回车 → 立刻开始一个字一个字往外冒
    （哪怕总耗时一样，体感完全不同）

===============================================================================
LangGraph 的 4 种流模式（本课全部演示一遍）
===============================================================================
    stream_mode="values"     每一步之后，吐出**完整状态**（看得出状态怎么长大的）
    stream_mode="updates"    每一步之后，只吐**这一步的更新**（最常用，看得出走过哪些节点）
    stream_mode="messages"   大模型吐出的**每个 token**（做打字机效果就用它）
    stream_mode="custom"     节点内部自己`发射`的数据（进度条、中间日志等）

还可以传一个列表同时拿多种流：stream_mode=["updates", "messages"]

===============================================================================
原理：流式的数据是从哪来的？
===============================================================================
图不是"一段代码从头跑到尾"，而是"一个节点一个节点地跑"。
每跑完一个节点，引擎就有机会把中间结果交出来 —— 这就是 values / updates。

而 messages 模式更进一步：它给大模型挂了一个回调，
模型每吐出一个 token，回调就把它推出来 —— 所以在你还没拿到完整回答时，
就能一个字一个字地显示。

★ 注意：能否真正逐 token 输出，取决于模型是否支持流式。
  本项目的离线模拟模型也实现了流式接口，所以一样能看到打字机效果。
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common import pretty
from common.llm import get_model

llm = get_model(quiet=True)


# =============================================================================
# 第 1 步：准备一张带"多步骤"的图，这样流式才有东西可看
# =============================================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    draft: str
    score: int


def node_research(state: State) -> dict:
    """第一步：假装做了些调研工作，并自己发射一条进度信息给前端。"""
    # ★ get_stream_writer() 拿到一个"发射器"，
    #   往里写的东西，会在 stream_mode="custom" 时被推送给调用方。
    #   这跟 yield 很像，但不需要把节点改写成生成器。
    writer = get_stream_writer()
    for i, step in enumerate(["检索资料", "筛选要点", "整理结论"], 1):
        writer({"progress": i / 3, "step": step})
        time.sleep(0.05)  # 假装在干活
    return {"draft": "关于 LangGraph 的调研草稿：它是一个有状态的多步骤编排框架。"}


def node_polish(state: State) -> dict:
    """第二步：调用大模型润色（这一步会产生 token 流）。"""
    response = llm.invoke([HumanMessage(content="请润色这段话：%s" % state["draft"])])
    return {"messages": [response], "score": 90}


def build_graph():
    return (
        StateGraph(State)
        .add_node("research", node_research)
        .add_node("polish", node_polish)
        .add_edge(START, "research")
        .add_edge("research", "polish")
        .add_edge("polish", END)
        .compile()
    )


# =============================================================================
# 第 2 步：逐个演示四种流模式
# =============================================================================
def demo_values(app) -> None:
    pretty.section('模式一：stream_mode="values" —— 每步之后的完整状态')

    for i, snapshot in enumerate(app.stream({"messages": []}, stream_mode="values")):
        print("  快照 %d：" % i)
        for k, v in snapshot.items():
            if k == "messages":
                print("      messages = [%d 条]" % len(v))
            else:
                print("      %-10s = %s" % (k, str(v)[:60]))

    pretty.info("特点：看到的是「全量」。适合调试状态演化，但数据量大。")


def demo_updates(app) -> None:
    pretty.section('模式二：stream_mode="updates" —— 每步的增量（最常用）')

    for event in app.stream({"messages": []}, stream_mode="updates"):
        for node, update in event.items():
            print("  节点 %s 刚刚执行完，它更新了：" % pretty.bold(node))
            for k, v in update.items():
                print("      %-10s" % k)

    pretty.info("特点：只告诉你「谁跑完了、改了什么」。做进度提示、日志记录最合适。")


def demo_messages(app) -> None:
    pretty.section('模式三：stream_mode="messages" —— 逐 token 输出（打字机效果）')

    print("  AI 正在思考…… ", end="", flush=True)
    for chunk, metadata in app.stream({"messages": []}, stream_mode="messages"):
        # chunk 是一个 AIMessageChunk（消息碎片），metadata 里有它来自哪个节点
        text = chunk.content
        if text:
            print(text, end="", flush=True)
    print()

    pretty.info("特点：token 级粒度。这就是 ChatGPT 那种「一个字一个字冒出来」的实现方式。")
    pretty.info("metadata 里的 langgraph_node 字段告诉你「这段文字是哪个节点产生的」。")


def demo_custom(app) -> None:
    pretty.section('模式四：stream_mode="custom" —— 节点内部自己发射数据')

    for chunk in app.stream({"messages": []}, stream_mode="custom"):
        # chunk 就是节点里 writer(...) 传出来的对象，原样到达
        bar = "█" * int(chunk["progress"] * 20)
        print("  [%-20s] %3d%%  %s" % (bar, chunk["progress"] * 100, chunk["step"]))

    pretty.info("特点：完全自定义。适合做进度条、阶段性提示、把中间结果推给前端。")


def demo_multi_mode(app) -> None:
    pretty.section("进阶：同时订阅多种流")

    print("  stream(..., stream_mode=['updates', 'custom']) 会返回 (模式名, 数据) 元组：")
    print()
    for mode, chunk in app.stream({"messages": []}, stream_mode=["updates", "custom"]):
        if mode == "custom":
            print("      %s → %s" % (pretty.yellow(mode), chunk))
        else:
            print("      %s → 节点 %s" % (pretty.cyan(mode), list(chunk.keys())))

    pretty.info("这个能力在做 Web 服务时特别有用：一个接口同时推「进度」和「token」。")

    # 别忘了演示 values + updates 的组合
    print()
    print("  再试试 ['values', 'updates']：")
    for mode, chunk in app.stream({"messages": []}, stream_mode=["values", "updates"]):
        label = "完整状态" if mode == "values" else "本次增量"
        print("      %-6s (%s) → %s" % (pretty.cyan(mode), label, str(chunk)[:60]))


# =============================================================================
# 第 3 步：实战 —— 做一个真正好用的流式聊天函数
# =============================================================================
def streaming_chat(app, user_input: str, config: dict | None = None) -> str:
    """一个可以直接抄到你自己项目里的流式对话函数。

    返回完整的回答文本（方便调用方做后续处理，比如存库）。
    """
    full_text = []
    print("  %s " % pretty.green("AI  "), end="", flush=True)

    for chunk, metadata in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        # 只显示 AI 自己产生的文本，跳过工具节点产生的中间内容
        if metadata.get("langgraph_node") == "polish" or chunk.content:
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_text.append(str(chunk.content))

    print()
    return "".join(full_text)


def demo_typewriter() -> None:
    pretty.section("实战：一个可以直接复用的流式聊天机器人")

    # 用第 01 课那种最简图来演示（真正的聊天场景）
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chatbot(state: ChatState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    app = StateGraph(ChatState).add_node("chatbot", chatbot).add_edge(START, "chatbot").compile()

    for question in ["用一句话解释什么是 LangGraph", "再举一个它适合的场景"]:
        print()
        print("  %s %s" % (pretty.cyan("用户"), question))
        streaming_chat(app, question)


def main() -> None:
    pretty.lesson_header(
        "06",
        "流式输出（Streaming）",
        goals=[
            "分清 4 种 stream_mode 各自的用途和返回值形状",
            "用 messages 模式实现逐 token 的打字机效果",
            "用 custom 模式在节点里主动推送进度信息",
            "掌握同时订阅多种流的写法，以及可复用的流式对话函数",
        ],
        prerequisites="第 01 课（stream 基础）、第 05 课（自定义状态）",
    )

    app = build_graph()

    demo_values(app)
    demo_updates(app)
    demo_messages(app)
    demo_custom(app)
    demo_multi_mode(app)
    demo_typewriter()

    pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. 四种流模式，各司其职：
         values    每步之后的完整状态      调试状态演化
         updates   每步的增量              进度提示、日志（最常用）
         messages  逐 token                打字机效果
         custom    节点自己发射的数据       进度条、自定义事件

  2. 可以传列表同时订阅多种：
         for mode, chunk in app.stream(..., stream_mode=["updates", "messages"]):
         （传列表时，返回值一定变成 (模式名, 数据) 元组）

  3. 节点里用 get_stream_writer() 就能往 custom 流里推数据，
     不需要把节点函数改写成生成器。

  4. 流式的本质：图是"一个节点一个节点跑"的，
     每跑完一步引擎就有机会把中间结果交出来。
     messages 模式则是给模型挂了回调，token 一产生就推送。

  5. 异步版本：把 stream 换成 astream，invoke 换成 ainvoke，
     配合 async for 使用（在 Web 服务里几乎必用）。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 用 stream_mode="messages" 的 metadata，统计"polish 节点一共吐了多少个 token"。

  ② 改造 streaming_chat：把每次拿到的 chunk 累加起来，遇到句号就把这一句
     用不同颜色打印（模拟"一句一句显示"的效果）。

  ③ 用 custom 模式做一个真正有用的东西：在一个循环节点里，每轮都
     writer({"round": n}) 推一次，前端就能显示"正在进行第 N 轮思考"。

  ④ 思考题：为什么 messages 模式返回的是 (chunk, metadata) 二元组，
     而不是直接返回 chunk？（提示：一张图里可能有好几个模型调用）
"""
    )


if __name__ == "__main__":
    main()
