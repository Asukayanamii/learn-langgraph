"""第 07 课 · 控制流：分支、并行、动态分发

===============================================================================
到此为止，我们的图都是"一条道走到黑"。真实业务远没有这么直：
===============================================================================
    用户问技术问题 → 转技术 Agent
    用户问价格     → 转销售 Agent
    用户就是闲聊   → 直接回复

再比如，一个调研任务要同时查 10 个来源，串行要 10 秒，并行只要 1 秒。

本课讲三种控制流：
    1. 多路分支     一个判断函数返回不同标签，走不同的路
    2. 并行执行     一个节点连出多条边，多个节点"同时"跑
    3. 动态分发     用 Send 把一批任务"发射"到同一个节点上（map-reduce）

===============================================================================
最重要的概念：超级步（Superstep）
===============================================================================
LangGraph 不是随便什么时候都能并行。它的执行模型是"一轮一轮"的：

    第 1 轮（超级步）：执行所有被激活的节点 → 收集它们返回的状态更新
    第 2 轮：用合并后的状态，执行下一批被激活的节点
    ...

    ★ 同一轮里的节点是**并行**执行的，彼此看不到对方的更新
      —— 它们读到的都是"这一轮开始时的状态"。
      这一批跑完、状态合并之后，下一轮才能看到。

理解这一点，你就知道为什么：
    - 并行分支之间不能互相依赖（它们同时跑，看不见对方）
    - 合并必须靠 reducer（两个分支都写同一个字段时，需要合并规则）
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from common import graph_viz, pretty
from common.llm import get_model

llm = get_model(quiet=True)


# =============================================================================
# 第一部分：多路分支
# -----------------------------------------------------------------------------
# 之前我们用过 tools_condition（两个分支）。条件边其实可以有任意多个分支，
# 判断函数返回什么标签，就走对应的那条边。
# =============================================================================
class RouteState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def node_router(state: RouteState) -> dict:
    """入口节点：先原样返回，把"分诊"交给后面的条件边来做。"""
    return {}


def classify_question(state: RouteState) -> str:
    """分诊函数：看用户问了什么，返回对应的科室标签。

    ★ 生产环境这里通常会让大模型来做意图识别，返回结构化的类别。
      本课为了让你看清控制流，用关键词判断，效果更直观可预测。
    """
    text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = str(msg.content)
            break

    if any(w in text for w in ("价格", "多少钱", "报价", "优惠", "便宜")):
        return "sales"
    if any(w in text for w in ("报错", "崩溃", "接口", "代码", "部署", "bug")):
        return "tech"
    return "smalltalk"


def node_sales(state: RouteState) -> dict:
    reply = "【销售专员】您好！我们的专业版 199 元/月，年付享 8 折。需要我安排顾问给您详细报价吗？"
    return {"messages": [AIMessage(content=reply)]}


def node_tech(state: RouteState) -> dict:
    reply = "【技术支持】收到！麻烦提供一下报错堆栈和复现步骤，我先帮您定位是不是接口版本不兼容导致的。"
    return {"messages": [AIMessage(content=reply)]}


def node_smalltalk(state: RouteState) -> dict:
    reply = "【机器人】你好呀～有什么我可以帮你的吗？"
    return {"messages": [AIMessage(content=reply)]}


def build_router_graph():
    return (
        StateGraph(RouteState)
        .add_node("router", node_router)
        .add_node("sales", node_sales)
        .add_node("tech", node_tech)
        .add_node("smalltalk", node_smalltalk)
        .add_edge(START, "router")
        # 一个判断函数 + 一张标签到节点的映射表 = 多路分支
        .add_conditional_edges(
            "router",
            classify_question,
            {"sales": "sales", "tech": "tech", "smalltalk": "smalltalk"},
        )
        .add_edge("sales", END)
        .add_edge("tech", END)
        .add_edge("smalltalk", END)
        .compile()
    )


def demo_routing(app) -> None:
    pretty.section('第一部分：多路分支（按问题类型分诊）')

    for question in ["你们的专业版多少钱？", "我这边接口报错了怎么办", "嗨，在吗"]:
        result = app.invoke({"messages": [HumanMessage(content=question)]})
        print("  用户：%s" % pretty.cyan(question))
        print("  → %s" % result["messages"][-1].content)
        print()


# =============================================================================
# 第二部分：并行执行
# -----------------------------------------------------------------------------
# 一个节点连出 N 条边，这 N 个目标节点会在**同一个超级步**里并行跑。
# 最后需要有一个"汇聚节点"把它们的结果合并起来。
#
#         ┌──▶ check_weather ──┐
#   START ┤                    ├──▶ summarize ──▶ END
#         └──▶ check_stock   ──┘
# =============================================================================
class ParallelState(TypedDict):
    # ★ 注意这两个字段都用了 operator.add：
    #   两个并行节点各写各的结果，reducer 负责把它们拼起来。
    #   如果没写 reducer，后跑完的那个会把先跑完的覆盖掉 —— 这是并行场景的经典 bug！
    findings: Annotated[list, operator.add]
    timings: Annotated[list, operator.add]


def node_weather(state: ParallelState) -> dict:
    time.sleep(0.3)  # 假装这是个慢接口
    return {"findings": ["天气：明天多云转晴，22℃"], "timings": ["weather 耗时 0.3s"]}


def node_stock(state: ParallelState) -> dict:
    time.sleep(0.3)
    return {"findings": ["行情：上证指数收涨 0.8%"], "timings": ["stock 耗时 0.3s"]}


def node_summarize(state: ParallelState) -> dict:
    joined = "；".join(state["findings"])
    return {"findings": ["汇总报告：" + joined]}


def build_parallel_graph():
    return (
        StateGraph(ParallelState)
        .add_node("weather", node_weather)
        .add_node("stock", node_stock)
        .add_node("summarize", node_summarize)
        # ★ 从 START 同时连出两条边 —— 这就是并行的写法
        .add_edge(START, "weather")
        .add_edge(START, "stock")
        # 两条边都汇聚到同一个节点
        .add_edge("weather", "summarize")
        .add_edge("stock", "summarize")
        .add_edge("summarize", END)
        .compile()
    )


def demo_parallel(app) -> None:
    pretty.section("第二部分：并行执行（同时查天气和行情）")

    print("  ── 先看它到底是不是真的并行 ──")
    start = time.time()
    events = []
    for event in app.stream({"findings": [], "timings": []}, stream_mode="updates"):
        for node in event:
            events.append((round(time.time() - start, 2), node))
    total = time.time() - start

    for elapsed, node in events:
        print("      +%.2fs  节点 %s 执行完毕" % (elapsed, pretty.bold(node)))

    print()
    print("  weather 和 stock 各睡 0.3 秒。如果是串行，总耗时应 ≥0.6s；")
    print("  实际总耗时 = %s —— 说明它们确实在同一轮里并行跑了。" % pretty.green("%.2fs" % total))

    result = app.invoke({"findings": [], "timings": []})
    print()
    print("  ── 并行结果被 reducer 合并成了 ──")
    for item in result["findings"]:
        print("      · %s" % item)

    print()
    pretty.info("关键点：并行分支都会写 findings 字段，所以它必须配 reducer —— 否则图会直接报错。")


# =============================================================================
# 第三部分：用 Send 做动态分发（map-reduce）
# -----------------------------------------------------------------------------
# 前面两种是"编译时就定好"的控制流。但有些场景任务是运行时才知道的：
#
#     用户上传了 8 个文件 → 每个文件都要处理一遍
#
# 这时候用 Send：判断函数返回一个 Send 列表，
# 每个 Send 表示"往某个节点发一份输入"（相当于 map），
# 这些节点并行跑完后，结果靠 reducer 汇总（相当于 reduce）。
# =============================================================================
class MapReduceState(TypedDict):
    topic: str
    # 要处理的子任务列表（由 planner 节点生成）
    subtasks: list
    # 每个子任务的处理结果，用 operator.add 汇总
    results: Annotated[list, operator.add]


def node_plan(state: MapReduceState) -> dict:
    """规划节点：把一个调研任务拆成多个子问题。"""
    topic = state["topic"]
    return {
        "subtasks": [
            "%s 是什么" % topic,
            "%s 有什么用" % topic,
            "%s 有什么坑" % topic,
        ]
    }


def dispatch(state: MapReduceState):
    """★ 动态分发：为每个子任务生成一个 Send。

    返回 [Send(节点名, 传给该节点的状态片段), ...]
    这些节点会在同一个超级步里并行执行。
    """
    return [Send("worker", {"subtask": task}) for task in state["subtasks"]]



def node_worker(state: dict) -> dict:
    """工作节点：处理一个子任务。

    ★ 注意它的输入状态只有 subtask 一个字段 —— 因为 Send 只传了我们指定的那一小块，
      这正是 Send 的好处：可以给每个并行任务喂不同的、最小的输入。
    """
    task = state["subtask"]
    time.sleep(0.2)  # 假装在调用模型或查资料
    return {"results": ["【%s】→ 已调研完成，要点：……" % task]}

def node_reduce(state: MapReduceState) -> dict:
    """汇总节点：把所有子任务的结果合并成最终产出。"""
    return {"results": ["=== 调研报告（共 %d 个子问题）===\n" % len(state["results"]) + "\n".join(state["results"])]}


def build_map_reduce_graph():
    return (
        StateGraph(MapReduceState)
        .add_node("planner", node_plan)
        .add_node("worker", node_worker)
        .add_node("reduce", node_reduce)
        .add_edge(START, "planner")
        # Send 的分发目标写在第三个参数里；判断函数只负责"发射"，不负责"选边"
        .add_conditional_edges("planner", dispatch, ["worker"])
        .add_edge("worker", "reduce")
        .add_edge("reduce", END)
        .compile()
    )


def demo_map_reduce(app) -> None:
    pretty.section("第三部分：Send 动态分发（把 3 个子任务并行处理）")

    start = time.time()
    events = []
    for event in app.stream({"topic": "LangGraph", "subtasks": [], "results": []}, stream_mode="updates"):
        for node in event:
            events.append((round(time.time() - start, 2), node))

    for elapsed, node in events:
        print("      +%.2fs  节点 %s" % (elapsed, pretty.bold(node)))

    result = app.invoke({"topic": "LangGraph", "subtasks": [], "results": []})
    print()
    print("  最终产出：")
    print("      %s" % result["results"][-1].replace("\n", "\n      "))
    print()
    pretty.info("三个 worker 各处理一个子问题，并行跑完由 reduce 汇总 —— 这就是 map-reduce。")


def demo_why_reducer_matters() -> None:
    """反例：并行分支写同一个字段，但没有 reducer，会发生什么。

    结论可能和你想的不一样 —— LangGraph 会**主动报错**，而不是默默丢数据。
    这正是 reducer 存在的意义。
    """
    pretty.section("反例：并行分支写同一个字段却不加 reducer")

    class BadState(TypedDict):
        findings: list  # ★ 故意不写 reducer

    def a(state) -> dict:
        time.sleep(0.2)
        return {"findings": ["A 的结果"]}

    def b(state) -> dict:
        return {"findings": ["B 的结果"]}

    bad = (
        StateGraph(BadState)
        .add_node("a", a)
        .add_node("b", b)
        .add_edge(START, "a")
        .add_edge(START, "b")
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
    )

    try:
        result = bad.invoke({"findings": []})
        print("  居然跑通了，结果 = %s" % str(result["findings"]))
    except Exception as exc:
        print("  %s 运行时报错了：" % pretty.red("✘"))
        print("      %s" % pretty.red(type(exc).__name__))
        print("      %s" % str(exc).split("\n")[0][:150])

    print()
    pretty.info("LangGraph 发现「同一个超级步里有两个节点要写同一个字段」，直接拒绝执行。")
    pretty.info("所以并行场景下，被多个分支写入的字段**必须**配 reducer，否则图根本跑不起来。")

    # ---- 对比：如果是"先后执行"而不是"同时执行"呢？----
    print()
    print("  对比一下：把两个节点改成先后执行（不是并行），会怎样？")

    seq = (
        StateGraph(BadState)
        .add_node("a", a)
        .add_node("b", b)
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
    )
    result = seq.invoke({"findings": []})
    print("      结果 = %s" % pretty.yellow(str(result["findings"])))
    print()
    pretty.warn("顺序执行时不会报错，而是后者覆盖前者 —— 数据被悄悄丢掉了。")
    pretty.info("所以：并行没 reducer 会报错（好，至少你知道出事了）；")
    pretty.info("      串行没 reducer 会静默覆盖（危险，可能上线了才发现不对）。")


def main() -> None:
    pretty.lesson_header(
        "07",
        "控制流：分支、并行、动态分发",
        goals=[
            "用条件边实现多路分支（超过两条路的情况）",
            "理解超级步模型，知道同一轮里的节点是并行且互相看不见的",
            "掌握并行分支的写法，以及为什么必须配 reducer",
            "用 Send 实现 map-reduce 式的动态分发",
        ],
        prerequisites="第 01 课（状态与 reducer）、第 02 课（条件边）",
    )

    router_app = build_router_graph()
    parallel_app = build_parallel_graph()
    mr_app = build_map_reduce_graph()

    graph_viz.show_structure(router_app, title="① 多路分支图")
    graph_viz.show_structure(parallel_app, title="② 并行执行图")
    graph_viz.show_structure(mr_app, title="③ Send 动态分发图")

    demo_routing(router_app)
    demo_parallel(parallel_app)
    demo_map_reduce(mr_app)
    demo_why_reducer_matters()

    pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. 多路分支 = 判断函数返回标签 + 映射表里注册多个目标：
         add_conditional_edges("router", classify, {"sales": "sales", "tech": "tech", ...})

  2. 执行模型是「超级步」：
         同一轮里的节点并行执行，读到的都是本轮开始时的状态
         → 所以并行分支之间不能互相依赖
         → 所以并行分支写同一个字段时，必须靠 reducer 合并

  3. 并行 = 从一个节点连出多条边，再用汇聚节点收口。
     别忘了给被多个分支写入的字段加 reducer（operator.add 是默认选择）。

  4. Send 用于"运行时才知道有多少任务"的场景：
         判断函数返回 [Send("worker", {...}), ...]
         每个 Send 可以只传一小块状态给目标节点
         结果靠 reducer 汇总

  5. 选择建议：
         分支数量固定、逻辑简单     → 条件边
         任务数不固定、要并行处理   → Send
         需要多个节点协作出结果     → 并行边 + 汇聚节点
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 给分诊图加上第四类「退款」，观察路由是否正确。
     再想想：如果关键词同时命中了两类（比如"价格报错"），结果会怎样？

  ② 把并行图里的 summarize 改成"只保留两条 finding 里较长的那条"，
     练习在汇聚节点里做筛选逻辑。

  ③ 用 Send 改造 map-reduce 例子：让 planner 根据用户输入的数量动态决定
     拆成几个子任务（比如"拆成 5 个"就发 5 个 Send）。

  ④ 思考题：并行分支里，A 节点改了 user_name，B 节点能读到 A 改后的值吗？
     为什么？（提示：回到"超级步"的定义）
"""
    )


if __name__ == "__main__":
    main()
