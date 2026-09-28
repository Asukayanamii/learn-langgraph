"""第 01 课 · 构建一个最基础的聊天机器人

对应官方教程：https://langgraph.com.cn/tutorials/get-started/1-build-basic-chatbot/index.html

===============================================================================
本课要解决的三个问题
===============================================================================
1. 状态（State）是什么？—— 它是整张图共享的"记忆黑板"。
2. 节点（Node）是什么？—— 一个函数：读状态 → 干活 → 返回状态的更新。
3. 边（Edge）是什么？   —— 决定"这个节点跑完后，下一个该跑谁"。

===============================================================================
一张图的最小骨架（记住这 5 行就够了）
===============================================================================
    class State(TypedDict): ...            # 1. 定状态：图里有哪些数据
    builder = StateGraph(State)            # 2. 建图：把状态结构告诉引擎
    builder.add_node("名字", 函数)          # 3. 加节点：一个干活的地方
    builder.add_edge(START, "名字")         # 4. 加边：从入口指向节点
    app = builder.compile()                # 5. 编译：得到可运行的图

补充说明：compile() 不是"编译成机器码"，而是
    「校验图的合法性（有没有孤儿节点、有没有出口）→ 把状态结构装进引擎 → 返回一个可执行对象」。
以后加的 checkpointer（记忆）、断点（人工介入）也都是在 compile() 这一步挂上去的。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 把项目根目录加入模块搜索路径，使 common 可以被导入
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from common import graph_viz, pretty
from common.llm import get_model

# =============================================================================
# 第 1 步：定义状态（State）
# -----------------------------------------------------------------------------
# State 就是这张图的"数据契约"，规定了图运行过程中要携带哪些数据。
#
# 用 TypedDict 是因为：它既能被类型检查工具理解，运行时又只是个普通字典，
# 开销极小。LangGraph 会用它的"键名"来校验节点返回的更新是否合法。
#
# ★ 关键理解：Annotated[list, add_messages] 里的 add_messages 叫 reducer（合并函数）。
#   它规定了"当多个节点都要更新 messages 这个字段时，新值怎么和旧值合并"。
#   - 有 reducer：新返回的消息会被 **追加** 到已有列表后面（对话历史越滚越长）
#   - 没 reducer：新值直接 **覆盖** 旧值（最后一句话是谁说的就只剩谁）
#   我们写的是聊天机器人，显然需要"追加"，所以必须带上 add_messages。
# =============================================================================


class State(TypedDict):
    """图的共享状态：这里只有一个字段 messages，装整段对话历史。"""

    messages: Annotated[list, add_messages]


# =============================================================================
# 第 2 步：准备大模型
# -----------------------------------------------------------------------------
# get_model() 是项目封装的统一入口（见 common/llm.py）：
# 配了 API Key 就用真模型，没配就用离线模拟模型——两者接口完全一样。
# =============================================================================
llm = get_model(quiet=True)


# =============================================================================
# 第 3 步：写节点函数（Node）
# -----------------------------------------------------------------------------
# 节点函数的固定套路，请务必背下来：
#
#     输入：当前完整状态 state（一个字典）
#     输出：一个字典，表示"我要更新哪些字段"
#
# ★★ 最容易犯的错：直接改 state 里的内容然后 return state。
#    这是错的！节点不应该修改传进来的状态，而应该"返回增量"。
#    LangGraph 会拿你的返回值，按 reducer 的规则合并进全局状态。
# =============================================================================
def chatbot(state: State) -> dict:
    """把整段对话交给大模型，让它生成下一条回复。"""
    # state["messages"] 里是完整的历史（用户说的 + AI 说的 + 工具结果），
    # 一起发给模型，它才能"记得"上面聊了什么。
    response = llm.invoke(state["messages"])

    # 返回 {"messages": [...]}，add_messages 会把这条新消息追加到历史末尾。
    return {"messages": [response]}


def build_graph():
    """组装并编译图。把构建过程单独封装，方便复用和测试。"""
    # --- 3.1 建图：告诉引擎状态长什么样 ---
    builder = StateGraph(State)

    # --- 3.2 加节点 ---
    # 第一个参数是节点的唯一名字（后面画图和调试都会显示它）；
    # 第二个参数是"被调用时执行的函数"。名字你可以随便起，但见名知意最好。
    builder.add_node("chatbot", chatbot)

    # --- 3.3 加入口边 ---
    # START 是 LangGraph 内置的虚拟节点，代表"图开始执行的地方"。
    # 这条边的意思是：一运行图，就先去执行 chatbot 节点。
    builder.add_edge(START, "chatbot")

    # --- 3.4 编译 ---
    # 这里没有写"chatbot 之后去哪"，因为一个节点没有出边时，
    # LangGraph 会认为它执行完就到终点了（相当于自动连到 END）。
    # 如果图有多个终点分支，就需要显式 add_edge("chatbot", END)。
    return builder.compile()


# =============================================================================
# 第 4 步：跑起来
# =============================================================================
def run_once(app, user_input: str) -> None:
    """跑一轮对话，并把"图是怎么走的"打印出来。"""
    pretty.section("用户：%s" % user_input)

    # graph.stream(...) 会按"每一步"返回结果，而不是等全部跑完才返回。
    # 参数 stream_mode="updates" 的含义是：
    #   每执行完一个节点，就吐出 {"节点名": 该节点返回的状态更新}。
    # 用它最能看清图的执行顺序。
    print(pretty.dim("  ── 图的执行过程 ──"))
    for event in app.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="updates"):
        pretty.show_events(event, node_names={"chatbot"})

    # invoke 和 stream 是同一件事的两种调用方式：
    # invoke 等全部跑完返回最终状态，stream 边跑边返回中间过程。


def demo_invoke(app) -> None:
    """演示 invoke：一次性拿到最终状态。"""
    pretty.section("用 invoke() 拿最终状态（看看 State 里到底存了什么）")
    final_state = app.invoke({"messages": [{"role": "user", "content": "你好，介绍一下你自己"}]})
    pretty.show_messages(final_state["messages"], title_text="最终状态里的 messages")


def demo_reducer() -> None:
    """对比实验：有 reducer 和没 reducer 到底差在哪。

    这是本课最重要的实验。我们真的建一张小图跑一遍，让差别肉眼可见。

    注意：operator 必须 import 在文件顶部。因为 TypedDict 里的类型注解是"字符串"，
    要等 LangGraph 建图时才去解析，那时候函数内部的局部导入早就失效了。
    """
    pretty.section("对比实验：reducer 到底做了什么？")

    # 这张小图的状态有两个字段，故意一个带 reducer、一个不带：
    class DemoState(TypedDict):
        # 带 reducer：用 operator.add 把新旧列表"拼接"
        log: Annotated[list, operator.add]
        # 不带 reducer：后来者直接覆盖
        latest: str

    def node_one(state: DemoState) -> dict:
        return {"log": ["节点1 干的活"], "latest": "节点1"}

    def node_two(state: DemoState) -> dict:
        return {"log": ["节点2 干的活"], "latest": "节点2"}

    # 两张小图串起来跑：START -> node_one -> node_two
    demo = (
        StateGraph(DemoState)
        .add_node("node_one", node_one)
        .add_node("node_two", node_two)
        .add_edge(START, "node_one")
        .add_edge("node_one", "node_two")
        .compile()
    )

    result = demo.invoke({"log": ["初始记录"], "latest": "初始值"})

    print("  执行顺序：START → node_one → node_two")
    print("  节点1 返回：{'log': ['节点1 干的活'], 'latest': '节点1'}")
    print("  节点2 返回：{'log': ['节点2 干的活'], 'latest': '节点2'}")
    print()
    print(pretty.bold("  最终结果："))
    print("      log    = %s" % pretty.green(str(result["log"])))
    print("               ↑ 三条记录都在！因为 operator.add 把它们拼接了起来")
    print("      latest = %s" % pretty.red(repr(result["latest"])))
    print("               ↑ 只剩最后一次的值，前面的被覆盖了")

    print()
    pretty.info("结论：想让数据「累积」就必须加 reducer；不加 reducer 就是「覆盖」。")
    pretty.info("messages 字段官方给了现成的 add_messages，它会追加消息、并按 id 自动去重更新。")


def interactive_loop(app) -> None:
    """交互模式：和你一问一答（对应官方文档第 7 步的 while 循环）。"""
    pretty.section("交互模式（输入 quit / exit / q 退出）")

    # ★ 注意这里的 config：本课没有加记忆功能，所以每一轮都是"全新的一轮对话"，
    #   上一次说的话不会被记住。第 03 课我们会用 checkpointer 解决它。
    while True:
        user_input = pretty.safe_input(pretty.cyan("你 > "))
        if user_input is None:
            print()
            break
        if user_input.strip().lower() in ("quit", "exit", "q", ""):
            print(pretty.dim("再见！"))
            break
        run_once(app, user_input)


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    pretty.lesson_header(
        "01",
        "构建一个基础的聊天机器人",
        goals=[
            "理解 StateGraph 的五个必备要素（State / 节点 / 边 / START / compile）",
            "理解 reducer（add_messages）如何决定状态更新的合并方式",
            "掌握 stream(stream_mode=\"updates\") 观察图执行过程的方法",
        ],
        prerequisites="Python 基础语法、字典、类型注解；不需要 LangGraph 基础",
    )

    # 打印当前用的是真模型还是离线模拟模型
    from common.llm import model_status

    pretty.kv("当前模型", model_status())
    print()

    # --- 建图 ---
    app = build_graph()

    # --- 画出来看看：LangGraph 的图结构是可以直接观察的 ---
    graph_viz.show_structure(app, title="我们刚构建的图")

    # --- 自动演示 ---
    pretty.section("开始演示")
    run_once(app, "你好，介绍一下你自己")
    run_once(app, "LangGraph 是什么？")

    demo_invoke(app)
    demo_reducer()

    # --- 进入交互模式（可选） ---
    if pretty.is_interactive():
        interactive_loop(app)
    else:
        pretty.try_hint()

    # --- 本课小结 ---
    pretty.section("本课小结")
    print(
        """
  1. StateGraph 的核心就三样东西：

        状态 State ── 数据放在哪
        节点 Node  ── 谁在干活（一个函数，进 State 出更新）
        边   Edge  ── 干完活下一步去哪

  2. 节点函数不要修改传进来的 state，而是"返回一个字典表示更新"。

  3. reducer 决定"新值如何合并进旧状态"：
         messages 用 add_messages → 追加（对话历史越来越长）
         其他字段没写 reducer     → 覆盖（只保留最新值）

  4. 观察图跑法的两个视角：
         stream(..., stream_mode="updates")  看"哪个节点执行了、改了什么"
         invoke(...)                         直接拿最终完整状态

  5. compile() 是"组装完成"的标志：后面加记忆、断点都在这一步挂载。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 给图加一个新的 system 提示（人设）：
       在 run_once 里这样传：
           {"messages": [{"role": "system", "content": "你是一个只讲冷笑话的机器人"},
                         {"role": "user", "content": user_input}]}
       观察模型回复风格的变化。

  ② 把 stream_mode 改成 "values" 再跑一次，看看和 "updates" 有什么不同
       （提示：values 返回的是"每一步之后的完整状态"，updates 只返回"这一步的增量"）。

  ③ 思考题：为什么说"本课的机器人没有记忆"？
       你可以连续问两轮"我叫小明"、"我叫什么"，看看它能不能答对。
       想清楚问题出在哪，第 03 课就顺理成章了。
"""
    )


if __name__ == "__main__":
    main()
