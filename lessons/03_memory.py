"""第 03 课 · 让机器人记住你说过的话（Checkpointer 持久化记忆）

对应官方教程：https://langgraph.com.cn/tutorials/get-started/3-add-memory/index.html

===============================================================================
先搞清楚一个常见误解
===============================================================================
很多人以为"给模型传 messages 列表"就等于有记忆了。其实不是：

    传给模型的 messages 只是"这一轮请求里带了什么"。
    程序一重启、或者换一个用户，这个列表就没了。

真正的"记忆"要解决两件事：
    1. 存哪  —— 把每一轮的状态存下来（内存、SQLite、Postgres...）
    2. 怎么取 —— 下一个请求来的时候，把对应的历史捞出来拼回去

LangGraph 的答案就是 Checkpointer（检查点保存器）。

===============================================================================
Checkpointer 的工作原理（本课重点）
===============================================================================
    图每执行完一个"超级步"(superstep)，就把当前完整状态存一份快照：

        checkpoint 1: {messages: [用户:你好]}
        checkpoint 2: {messages: [用户:你好, AI:你好呀]}
        checkpoint 3: {messages: [用户:你好, AI:你好呀, 用户:我叫小明]}

    每份快照都挂在一个 thread_id（会话 ID）下面，相当于"文件夹"。
    下次同一个 thread_id 再来请求，引擎就把最后一份快照的状态作为起点。

    ★ 所以"记忆"本质上不是模型的能力，而是**状态持久化**的能力。
      这是 LangGraph 相对"手写一个 while 循环调 API"最值钱的地方之一。

===============================================================================
两种 Checkpointer（本课都会演示）
===============================================================================
    InMemorySaver  存在内存里，程序一关就没了 —— 适合测试
    SqliteSaver    存在 SQLite 文件里，关掉程序再打开还在 —— 适合单机应用
    （生产环境常用 PostgresSaver，用法完全一样，换一个对象即可）
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from common import graph_viz, pretty
from common.config import DATA_DIR
from common.llm import get_model

# =============================================================================
# 第 1 步：还是熟悉的配方 —— 状态 + 工具 + 模型
# =============================================================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def save_note(content: str) -> str:
    """把一段内容记录到笔记本里。当用户说"记住"、"记下来"、"保存"某事时使用。

    Args:
        content: 要记录的内容。
    """
    return "已记入笔记本：%s" % content


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。当用户问天气、气温时使用。

    Args:
        city: 城市名称。
    """
    return "%s 今天晴，气温 25℃" % city


TOOLS = [save_note, get_weather]
llm = get_model(quiet=True).bind_tools(TOOLS)


def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def build_graph(checkpointer=None):
    """建图。

    ★ 关键点：checkpointer 不是在 add_node 时指定的，
      而是在 compile() 的时候"挂"到整张图上的。
      这意味着 —— 同一份图的定义，加不加记忆只是一行代码的差别。
    """
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "chatbot")

    # 挂了 checkpointer 之后，LangGraph 就要求每次调用必须提供 thread_id，
    # 否则它不知道该把这轮对话存到哪个"会话文件夹"里，会直接报错。
    return builder.compile(checkpointer=checkpointer)


def say(app, text: str, thread_id: str) -> str:
    """往指定的会话里说一句话，返回 AI 的回复文本。

    ★ 注意第二个参数 config：通过它把 thread_id 传进去。
      这个 config 会一路传递到 checkpointer，告诉它"这一轮属于哪个会话"。
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [{"role": "user", "content": text}]}, config=config)

    last = result["messages"][-1]
    return str(last.content)


# =============================================================================
# 第 2 步：实验一 —— 没有 checkpointer 会怎样
# =============================================================================
def experiment_no_memory() -> None:
    pretty.section("实验一：不加 checkpointer（对照组的失败现场）")

    app = build_graph(checkpointer=None)

    print("  %s 第 1 轮：" % pretty.cyan("用户"))
    print("     我叫小明")
    print("  %s 第 2 轮：" % pretty.cyan("用户"))
    print("     我叫什么？")

    # 每一轮都是全新的状态，模型只看得到当前这一句
    r1 = app.invoke({"messages": [{"role": "user", "content": "我叫小明"}]})
    print("  %s %s" % (pretty.green("AI  "), r1["messages"][-1].content))

    r2 = app.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]})
    print("  %s %s" % (pretty.green("AI  "), r2["messages"][-1].content))

    print()
    pretty.warn("每轮调用都只有 2 条消息（本次提问 + 本次回答），历史根本没传进去。")
    pretty.info("这说明：不持久化状态，机器人就是个「每次失忆」的家伙。")


# =============================================================================
# 第 3 步：实验二 —— 加上 InMemorySaver
# =============================================================================
def experiment_memory_saver() -> None:
    pretty.section("实验二：加上 InMemorySaver（同一进程内记住）")

    checkpointer = InMemorySaver()
    app = build_graph(checkpointer=checkpointer)

    print("  【会话 A：thread_id = 小明】")
    print("     用户：我叫小明")
    print("     AI  ：%s" % say(app, "我叫小明", "小明"))

    print("     用户：我叫什么？")
    print("     AI  ：%s" % say(app, "我叫什么？", "小明"))

    print()
    print("  【会话 B：thread_id = 小红】—— 另一个完全独立的会话")
    print("     用户：北京天气")
    print("     AI  ：%s" % say(app, "北京天气怎么样", "小红"))

    print("     用户：我叫什么？")
    print("     AI  ：%s" % say(app, "我叫什么？", "小红"))

    print()
    pretty.ok("会话 A 记得「小明」，会话 B 完全不知道小明是谁 —— 说明会话之间是隔离的。")
    pretty.info("隔离的依据就是 thread_id：它相当于每个用户的独立档案袋。")


# =============================================================================
# 第 4 步：实验三 —— 查看状态历史（这是 Checkpointer 最酷的能力）
# =============================================================================
def experiment_history() -> None:
    pretty.section("实验三：查看一个会话的完整状态演化史")

    app = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "history-demo"}}

    say(app, "你好", "history-demo")
    say(app, "我叫小明", "history-demo")
    say(app, "北京天气怎么样", "history-demo")

    # get_state_history 会把这条会话的每一份快照按时间顺序列出来。
    # 注意：LangGraph 默认从最新的开始倒序返回。
    history = list(app.get_state_history(config))

    print("  这个会话一共产生了 %d 份状态快照（每一步一份）：" % len(history))
    for i, snapshot in enumerate(reversed(history)):
        # snapshot.next 表示"这份快照之后，接下来要执行哪些节点"
        nxt = "、".join(snapshot.next) if snapshot.next else "(执行完毕)"
        step = snapshot.metadata.get("step", "?") if snapshot.metadata else "?"
        count = len(snapshot.values.get("messages", []))
        print("     步骤 %-2s │ 消息数 %-2d │ 下一步: %s" % (step, count, nxt))

    print()
    pretty.info("用途：出问题时可以回放到任意一步，看看当时状态到底是什么样（第 08 课会用它做「时间旅行」）。")
    pretty.info("这也是为什么 LangGraph 适合做需要审计、需要人工介入的严肃业务系统。")


# =============================================================================
# 第 5 步：实验四 —— SqliteSaver，真正的"关机也记得"
# =============================================================================
def experiment_sqlite() -> None:
    pretty.section("实验四：SqliteSaver（把记忆写进磁盘，重启程序也不丢）")

    db_path = DATA_DIR / "memory_demo.sqlite"
    print("  数据库文件：%s" % pretty.bold(str(db_path)))

    # from_conn_string 是一个上下文管理器，负责开关数据库连接。
    # 注意：SqliteSaver 的连接对象不能被多个线程共用，
    # 所以在实际项目里通常每个请求开一个，或者用连接池。
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        print()
        print("  %s 写入一句话（thread_id = 持久化演示）" % pretty.cyan("第一次运行："))
        print("     AI：%s" % say(app, "记住一件事：我下周三要去杭州出差", "持久化演示"))

        # 直接读数据库确认它真的落盘了
        snapshot = app.get_state({"configurable": {"thread_id": "持久化演示"}})
        print("     当前会话已保存 %d 条消息，它们已经写进 SQLite 文件了。" % len(snapshot.values["messages"]))

    print()
    pretty.ok("现在可以关掉程序，再重新运行本课 —— 记忆还在（见下方「跨进程验证」）。")
    pretty.info("这就是「持久化记忆」：状态不在进程内存里，而在数据库里。")


def verify_across_restart() -> None:
    """模拟"重启程序再读一次"：重新打开数据库，看历史还在不在。"""
    pretty.section("跨进程验证：重新打开数据库，看记忆是否还在")

    db_path = DATA_DIR / "memory_demo.sqlite"
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "持久化演示"}}

        snapshot = app.get_state(config)
        messages = snapshot.values.get("messages", [])

        if not messages:
            pretty.warn("数据库里还没有内容，请先运行「实验四」。")
            return

        print("  从磁盘读回了 %d 条历史消息：" % len(messages))
        pretty.show_messages(messages, limit=6)

        print()
        print("  %s 继续这个会话问一句：我刚才说我下周三要干嘛？" % pretty.cyan("用户"))
        print("     AI：%s" % say(app, "我刚才说我下周三要干嘛？", "持久化演示"))

    print()
    pretty.ok("新进程、新连接，历史依然被完整读了出来 —— 这就是持久化的意义。")


def interactive_loop(app) -> None:
    pretty.section("交互模式（输入 quit / exit / q 退出）")
    print(pretty.dim("  当前 thread_id = interactive-demo，你可以连续对话，它会记得你。"))

    while True:
        user_input = pretty.safe_input(pretty.cyan("你 > "))
        if user_input is None:
            print()
            break
        if user_input.strip().lower() in ("quit", "exit", "q", ""):
            print(pretty.dim("再见！下次运行本课，记忆还在（如果你用的是 SqliteSaver）。"))
            break
        print("  %s %s" % (pretty.green("AI  "), say(app, user_input, "interactive-demo")))


def main() -> None:
    pretty.lesson_header(
        "03",
        "让机器人记住你说过的话（Checkpointer）",
        goals=[
            "理解「记忆」的本质是状态持久化，不是模型的能力",
            "掌握 thread_id：会话隔离的依据",
            "学会 InMemorySaver 与 SqliteSaver 两种存储后端",
            "会用 get_state / get_state_history 查看和回溯历史状态",
        ],
        prerequisites="第 01 课（State）、第 02 课（工具）",
    )

    from common.llm import model_status

    pretty.kv("当前模型", model_status())
    print()

    experiment_no_memory()
    experiment_memory_saver()
    experiment_history()
    experiment_sqlite()
    verify_across_restart()

    if pretty.is_interactive():
        with SqliteSaver.from_conn_string(str(DATA_DIR / "memory_demo.sqlite")) as checkpointer:
            interactive_loop(build_graph(checkpointer=checkpointer))
    else:
        pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. 记忆 = 状态持久化。模型本身没有记忆，是 LangGraph 把历史状态存下来、
     下次再拼回去，才产生了「记忆」的效果。

  2. 开启记忆只需要一行：compile(checkpointer=...)

  3. thread_id 是会话隔离的依据：
         同一个 thread_id  → 共享同一份历史（同一个用户/同一个对话窗口）
         不同 thread_id    → 完全独立（不同用户互不干扰）

  4. 三种 Checkpointer 用法完全一样，按场景挑：
         InMemorySaver   内存，进程结束就没了      测试用
         SqliteSaver     本地文件，单机持久化       桌面/小工具
         PostgresSaver   数据库，可多实例共享       生产环境

  5. append 之外的能力：get_state(读当前) / get_state_history(读全部快照) /
     update_state(手动改状态) —— 这是"时间旅行"和"人工修正"的基础。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 把 interactive_loop 里的 thread_id 换成一个变量，改成可以输入用户名，
     体会"不同用户不同记忆"的效果。

  ② 用 get_state_history 找到"用户说第一句话"之前的那份快照，
     看看它的 values 和 next 分别是什么（提示：空会话的快照长什么样）。

  ③ 思考题：为什么 SqliteSaver 适合单机、而生产环境要用 PostgresSaver？
     （提示：想想如果你的服务开了 3 个进程，每个进程各写各的本地文件会怎样）
"""
    )


if __name__ == "__main__":
    main()
