"""第 08 课 · 子图与时间旅行（把图当积木用）

===============================================================================
第一部分：子图（Subgraph）
===============================================================================
图可以嵌套图 —— 一张编译好的图，可以直接当作另一张图的节点来用。

为什么要这样做？
    1. 复用      "订单处理"流程被三个地方用到，写一次就够了
    2. 封装      每个团队维护自己的子图，对外只暴露一个节点名
    3. 清晰      父图看大局（几步走），子图看细节（每步怎么做）

子图有两种用法，区别只在**状态怎么传**：

    用法 A：共享状态
        父子图用同一套状态字段（比如都有 messages），直接 add_node("子图名", 子图)
        父图的 state 原样传进去，子图的更新也原样传回来。最简单。

    用法 B：独立状态（需要写一个转换函数）
        子图有自己的私有字段，父图不认识。
        这时要包一层函数：把父图状态"翻译"成子图输入，再把子图输出"翻译"回去。

===============================================================================
第二部分：时间旅行（Time Travel）
===============================================================================
配了 checkpointer 之后，每一步的状态都快照都存着。
于是你可以：

    1. 回看    get_state_history()    把整个会话的历史列出来
    2. 修改    update_state()         手动改某一时刻的状态（比如纠正模型的错误输出）
    3. 重放    指定 checkpoint_id 再执行一次 —— 相当于"从那个时间点开一条新分支"

这在真实业务里非常有用：
    - 模型答错了 → 人工改一下状态 → 让它接着往下跑，不用从头再来
    - 排查线上问题 → 回到出问题的那一步，看当时到底看到了什么
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common import graph_viz, pretty
from common.llm import get_model

llm = get_model(quiet=True)


# =============================================================================
# 第一部分：造一个子图 —— "订单处理流水线"
# =============================================================================
# 子图有自己独立的、更贴近业务的状态结构。
# 父图不需要关心"库存怎么扣的"，它只关心"订单处理完了没有"。
# =============================================================================
class OrderState(TypedDict):
    """子图的私有状态：只关心订单相关的数据。"""

    order_id: str
    amount: float
    steps: Annotated[list, lambda old, new: (old or []) + (new or [])]


def sub_check_stock(state: OrderState) -> dict:
    return {"steps": ["校验库存：充足"]}


def sub_calc_discount(state: OrderState) -> dict:
    """按金额算折扣。"""
    amount = state.get("amount", 0)
    if amount >= 1000:
        discount = "9 折（大额订单）"
    elif amount >= 300:
        discount = "95 折"
    else:
        discount = "无折扣"
    return {"steps": ["计算折扣：%s" % discount]}


def sub_create_shipment(state: OrderState) -> dict:
    return {"steps": ["生成发货单：单号 SF%s" % state["order_id"]]}


def build_order_subgraph():
    """构建并编译子图。注意：子图本身也是一张完整的、可以独立运行的图。"""
    return (
        StateGraph(OrderState)
        .add_node("check_stock", sub_check_stock)
        .add_node("calc_discount", sub_calc_discount)
        .add_node("create_shipment", sub_create_shipment)
        .add_edge(START, "check_stock")
        .add_edge("check_stock", "calc_discount")
        .add_edge("calc_discount", "create_shipment")
        .add_edge("create_shipment", END)
        .compile()
    )


# =============================================================================
# 父图：客服总流程，中间把订单处理"外包"给子图
# =============================================================================
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    order_id: str
    amount: float
    steps: Annotated[list, lambda old, new: (old or []) + (new or [])]


def node_parse(state: SupportState) -> dict:
    """父图节点：从用户的话里解析出订单号和金额。"""
    text = str(state["messages"][-1].content)

    import re

    order_id = "1001"
    m = re.search(r"([A-Za-z]?\d{4,})", text)
    if m:
        order_id = m.group(1)

    amount = 500.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*元", text)
    if m:
        amount = float(m.group(1))
    elif m := re.search(r"(\d+(?:\.\d+)?)", text):
        amount = float(m.group(1))

    return {"order_id": order_id, "amount": amount, "steps": ["解析订单：%s，金额 %s 元" % (order_id, amount)]}


def node_reply(state: SupportState) -> dict:
    """父图节点：根据子图的处理结果回复用户。"""
    detail = "\n".join("  · " + s for s in state["steps"])
    reply = "已为你处理订单，流程如下：\n%s\n请留意发货短信。" % detail
    return {"messages": [AIMessage(content=reply)]}


def build_parent_graph_shared() -> object:
    """用法 A：子图与父图**共享部分状态**。

    直接把编译好的子图当节点加进去。LangGraph 会把父图的状态传给子图，
    子图只取它认识的字段（order_id / amount / steps），返回的更新也会合并回父图。

    ★ 前提：子图的状态字段必须是父图状态字段的**子集**。
      这里 OrderState 的字段都在 SupportState 里，所以可以直接挂。
    """
    subgraph = build_order_subgraph()

    return (
        StateGraph(SupportState)
        .add_node("parse", node_parse)
        .add_node("order_pipeline", subgraph)  # ★ 子图作为一个节点
        .add_node("reply", node_reply)
        .add_edge(START, "parse")
        .add_edge("parse", "order_pipeline")
        .add_edge("order_pipeline", "reply")
        .add_edge("reply", END)
        .compile()
    )


# -----------------------------------------------------------------------------
# 用法 B：子图状态与父图完全不同，需要写转换函数
# -----------------------------------------------------------------------------
class PureOrderState(TypedDict):
    """一个"纯粹独立"的子图状态：字段名和父图完全对不上。"""

    oid: str
    total: float
    log: Annotated[list, lambda old, new: (old or []) + (new or [])]


def pure_check(state: PureOrderState) -> dict:
    return {"log": ["独立子图：校验通过 oid=%s" % state["oid"]]}


def pure_settle(state: PureOrderState) -> dict:
    return {"log": ["独立子图：结算完成，金额 %.2f" % state["total"]]}


def build_pure_subgraph():
    return (
        StateGraph(PureOrderState)
        .add_node("check", pure_check)
        .add_node("settle", pure_settle)
        .add_edge(START, "check")
        .add_edge("check", "settle")
        .add_edge("settle", END)
        .compile()
    )


def build_parent_graph_isolated():
    """用法 B：用一层包装函数来"翻译"状态。

    父图状态 SupportState  ──转换──▶  子图状态 PureOrderState
    父图状态 SupportState  ◀──转换──  子图状态 PureOrderState
    """
    subgraph = build_pure_subgraph()

    def call_subgraph(state: SupportState) -> dict:
        """包装函数：它本身是父图的一个普通节点，内部去调子图。"""
        # ① 父图 → 子图：把字段名翻译过去
        sub_input = {"oid": state.get("order_id", "unknown"), "total": state.get("amount", 0.0), "log": []}

        # ② 调用子图（子图可以独立编译、独立测试，这里还能单独加 config）
        sub_output = subgraph.invoke(sub_input)

        # ③ 子图 → 父图：把结果翻译回来
        return {"steps": ["（经独立子图处理）"] + sub_output["log"]}

    return (
        StateGraph(SupportState)
        .add_node("parse", node_parse)
        .add_node("isolated_pipeline", call_subgraph)
        .add_node("reply", node_reply)
        .add_edge(START, "parse")
        .add_edge("parse", "isolated_pipeline")
        .add_edge("isolated_pipeline", "reply")
        .add_edge("reply", END)
        .compile()
    )


# =============================================================================
# 第二部分：时间旅行
# =============================================================================
class CounterState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    total: int
    log: Annotated[list, lambda old, new: (old or []) + (new or [])]


def node_add(state: CounterState) -> dict:
    return {"total": state.get("total", 0) + 10, "log": ["加 10，当前 %d" % (state.get("total", 0) + 10)]}


def node_double(state: CounterState) -> dict:
    return {"total": state.get("total", 0) * 2, "log": ["翻倍，当前 %d" % (state.get("total", 0) * 2)]}


def build_counter_graph(checkpointer):
    return (
        StateGraph(CounterState)
        .add_node("add", node_add)
        .add_node("double", node_double)
        .add_edge(START, "add")
        .add_edge("add", "double")
        .add_edge("double", END)
        .compile(checkpointer=checkpointer)
    )


def demo_time_travel() -> None:
    pretty.section("第二部分：时间旅行（回看 / 修改 / 重放）")

    checkpointer = InMemorySaver()
    app = build_counter_graph(checkpointer)
    config = {"configurable": {"thread_id": "travel"}}

    # ---- ① 先正常跑两轮 ----
    app.invoke({"total": 0, "log": []}, config=config)
    app.invoke({"total": 0, "log": []}, config=config)

    print("  ① 跑了两轮，现在状态是：%s" % pretty.bold(str(app.get_state(config).values["total"])))

    # ---- ② 回看所有历史快照 ----
    history = list(app.get_state_history(config))
    print()
    print("  ② 一共 %d 份快照（从新到旧）：" % len(history))
    for snap in reversed(history):
        step = snap.metadata.get("step", "?") if snap.metadata else "?"
        total = snap.values.get("total", "-")
        nxt = "、".join(snap.next) if snap.next else "结束"
        # checkpoint_id 是每份快照的唯一编号，重放时要靠它。
        # 它形如 UUID，前缀按时间递增，所以看后 8 位更能区分不同快照。
        ckpt = snap.config["configurable"]["checkpoint_id"]
        print("      步骤 %-2s │ total=%-4s │ 下一步: %-16s │ id=…%s" % (step, total, nxt, ckpt[-8:]))

    # ---- ③ 修改状态：人工纠正 ----
    print()
    print("  ③ 人工把 total 改成 999（模拟「人工纠正模型输出」）")

    # update_state 会生成一份新的快照，不会破坏历史
    app.update_state(config, {"total": 999, "log": ["【人工修改】把 total 设为 999"]})
    print("      改完之后 total = %s" % pretty.bold(str(app.get_state(config).values["total"])))

    # ---- ④ 从历史某一点重放（开一条新分支） ----
    print()
    print("  ④ 从最早的那份快照重新跑一遍（相当于「重开一局」）")

    # 找到"第一轮还在起点"的那份快照：next 里包含 'add'
    candidates = [s for s in history if s.next and "add" in s.next]
    if candidates:
        earliest = candidates[-1]  # 越靠后越早
        fork_config = earliest.config  # 这份 config 里带着 checkpoint_id
        print("      从步骤 %s 重新执行……" % earliest.metadata.get("step"))

        # ★ 直接拿历史快照的 config 去 invoke，就会从那个时间点往后跑，
        #   而且会写成一条新的分支（新的 checkpoint_id），原来的历史不受影响。
        result = app.invoke({"total": 0, "log": []}, config=fork_config)
        print("      重放后的 total = %s" % pretty.bold(str(result["total"])))

    print()
    pretty.info("时间旅行的三个用途：回看（get_state_history）、修改（update_state）、重放（用旧 config 再 invoke）。")
    pretty.info("这就是为什么 LangGraph 敢用在严肃业务里 —— 每一步都可追溯、可回滚。")


def main() -> None:
    pretty.lesson_header(
        "08",
        "子图与时间旅行（把图当积木用）",
        goals=[
            "掌握把编译好的图当作节点挂进父图的两种方式",
            "理解共享状态与独立状态（转换函数）的取舍",
            "学会用 get_state_history / update_state / 旧 config 做时间旅行",
            "理解「状态快照」为什么是生产级 Agent 的必备能力",
        ],
        prerequisites="第 03 课（checkpointer）、第 05 课（自定义状态）",
    )

    # ---- 子图 ----
    shared_app = build_parent_graph_shared()
    isolated_app = build_parent_graph_isolated()

    graph_viz.show_structure(shared_app, title="父图（其中 order_pipeline 是一个子图）")

    pretty.section("用法 A：父子图共享状态（直接挂上去）")
    result = shared_app.invoke(
        {"messages": [HumanMessage(content="我的订单 1001 花了 1200 元，帮我处理一下")]}
    )
    print("  用户：我的订单 1001 花了 1200 元，帮我处理一下")
    print("  %s" % result["messages"][-1].content)

    pretty.section("用法 B：子图状态完全独立（写转换函数）")
    result2 = isolated_app.invoke({"messages": [HumanMessage(content="订单 2002，金额 800 元")]})
    print("  用户：订单 2002，金额 800 元")
    print("  %s" % result2["messages"][-1].content)

    print()
    pretty.info("两种用法的区别只在状态怎么传：字段能对上就直接挂，对不上就包一层转换函数。")

    # ---- 时间旅行 ----
    demo_time_travel()

    pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. 子图 = 把一张编译好的图当成节点用：
         builder.add_node("节点名", 子图)

     好处：复用、封装、分工。父图看大局，子图管细节。

  2. 状态传递有两种情况：
         字段能对上（子图状态是父图状态的子集）→ 直接挂，最省事
         字段对不上                              → 包一个转换函数，
                                                   把父图状态翻译成子图输入，
                                                   再把子图输出翻译回父图状态

  3. 子图可以单独编译、单独测试、单独运行 —— 这对大型项目非常重要。

  4. 时间旅行三件套（需要 checkpointer）：
         get_state_history(config)   列出所有历史快照
         update_state(config, {...}) 修改当前状态（生成新快照，不改旧历史）
         用旧快照的 config 再 invoke  从那一时刻重新执行（开新分支）

  5. 为什么这很值钱：Agent 会犯错，业务要求可追溯。
     有了快照，你就能"回到犯错之前，人工修正，再往下跑"，而不是从头再来。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 给子图加一个"库存不足"的分支：如果金额超过 10000，子图直接跳到 END
     并返回一条失败信息，父图据此回复用户"订单被驳回"。

  ② 练习用 update_state 修改 messages 字段（把模型上一句错误回答改掉），
     再 invoke(None, config) 让图接着往下跑。

  ③ 思考题：子图能不能也有自己的 checkpointer？如果能，父子两级快照会怎么配合？
     （提示：LangGraph 支持给子图传 checkpoint_ns 做命名空间隔离）
"""
    )


if __name__ == "__main__":
    main()
