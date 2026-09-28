"""第 05 课 · 设计你自己的状态结构（State Schema 与自定义 Reducer）

对应官方教程：https://langgraph.com.cn/tutorials/get-started/5-customize-state/index.html

===============================================================================
为什么需要这一课？
===============================================================================
前四课的状态都只有一个 messages 字段。但真实项目里，图要携带的数据远不止对话：

    - 用户画像（姓名、等级、偏好）
    - 中间结果（检索到的文档、生成的草稿）
    - 流程控制（重试次数、当前阶段、是否需要人工介入）
    - 统计信息（调用了多少次工具、花了多少 token）

State 就是这张图的**数据库表结构**。设计得好，节点之间传数据就很顺；
设计得差，你会到处塞"临时变量"，图很快就乱了。

===============================================================================
本课要建立的核心认知：State 是"带合并规则的字典"
===============================================================================
    字段 = 数据
    类型注解 = 数据长什么样
    reducer = 当两个节点都想写这个字段时，怎么合并

    没有 reducer → 覆盖（后来者赢）
    有 reducer   → 按你定义的规则合并（追加 / 累加 / 求并集 / 取最大值...）

    reducer 的本质是一个函数：(旧值, 新值) -> 合并后的值
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages

from common import graph_viz, pretty

# =============================================================================
# 第 1 步：自定义 reducer
# -----------------------------------------------------------------------------
# 规则很简单：写一个接收「旧值、新值」两个参数、返回「合并结果」的函数。
# 为什么要用集合？因为每次"打标签"都返回同一个标签时，append 会重复，
# 而求并集天然去重。
# =============================================================================


def merge_unique(old: list, new: list) -> list:
    """合并两个列表并去重，保持原有顺序。

    ★ 注意 old 可能是 None：图刚启动、该字段还没被任何节点写过时，
      LangGraph 有时会传入 None 作为初始值，所以 reducer 要能容忍它。
    """
    result = list(old or [])
    for item in new or []:
        if item not in result:
            result.append(item)
    return result


def keep_max(old: int, new: int) -> int:
    """只保留最大值 —— 用来演示 reducer 可以任意定制，不一定非得是"合并"。"""
    return max(old or 0, new or 0)


# =============================================================================
# 第 2 步：定义一个信息量丰富的状态
# =============================================================================
class SupportState(TypedDict):
    """一个客服机器人可能用到的状态结构。"""

    # ---- 对话历史：用官方 reducer，追加 + 按 id 去重 ----
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- 用户画像：覆盖语义（每次都是最新值） ----
    user_name: str
    user_level: str

    # ---- 轮次计数：累加。operator.add 会让 1 + 1 = 2 ----
    turn: Annotated[int, operator.add]

    # ---- 标签：求并集去重（自定义 reducer） ----
    tags: Annotated[list, merge_unique]

    # ---- 处理过的"最严重级别"：取最大值（展示 reducer 可以任意定制） ----
    max_severity: Annotated[int, keep_max]

    # ---- 每个节点干了什么：追加，方便最后复盘 ----
    trace: Annotated[list, operator.add]


# =============================================================================
# 第 3 步：写几个协作的节点
# -----------------------------------------------------------------------------
# ★ 关键技巧：节点只返回"自己改动的字段"，不必返回完整状态。
#   没返回的字段会保持原样（或者说，用恒等函数"合并"了一下原值）。
#   这让节点之间的职责非常清晰。
# =============================================================================


def node_intake(state: SupportState) -> dict:
    """接待节点：识别用户身份，记录第一笔追踪信息。"""
    text = state["messages"][-1].content if state["messages"] else ""

    # 从消息里粗略识别用户等级（演示用）
    level = "VIP" if "VIP" in text or "会员" in text else "普通"

    return {
        "user_name": "小明",  # 覆盖语义：直接写入
        "user_level": level,  # 覆盖语义
        "turn": 1,  # 累加语义：reducer 会做 0 + 1
        "tags": ["接待"],  # 并集语义
        "max_severity": 1,  # 取最大语义
        "trace": ["intake: 完成身份识别，等级=%s" % level],
    }


def node_classify(state: SupportState) -> dict:
    """分类节点：判断问题类型，打上标签。"""
    text = state["messages"][-1].content if state["messages"] else ""

    tags = []
    severity = 1
    if any(w in text for w in ("退款", "投诉", "坏了", "故障")):
        tags.append("售后")
        severity = 3
    if any(w in text for w in ("急", "马上", "立刻")):
        tags.append("紧急")
        severity = 5
    if not tags:
        tags.append("咨询")

    return {
        "turn": 1,  # 又 +1
        "tags": tags,  # merge_unique 会把它们并入已有标签
        "max_severity": severity,  # keep_max 会保留较大的那个
        "trace": ["classify: 打标签 %s，严重度 %d" % (tags, severity)],
    }


def node_reply(state: SupportState) -> dict:
    """回话节点：根据前面的分析结果生成回复。

    ★ 注意这里读了 state 里的多个字段 —— 这正是"共享状态"的价值：
      节点之间不需要互相传参，都从 state 里拿。
    """
    name = state.get("user_name", "用户")
    level = state.get("user_level", "普通")
    tags = state.get("tags", [])
    severity = state.get("max_severity", 0)

    if severity >= 5:
        action = "已为你升级为紧急工单，客服会在 5 分钟内联系你。"
    elif severity >= 3:
        action = "已记录你的问题，售后专员会在 2 小时内跟进。"
    else:
        action = "你的问题我已记录，你可以继续补充细节。"

    reply = "你好 %s（%s 用户）！识别到问题标签：%s。%s" % (name, level, "、".join(tags), action)

    return {
        "messages": [AIMessage(content=reply)],
        "turn": 1,
        "trace": ["reply: 生成回复，严重度=%d" % severity],
    }


def build_graph():
    return (
        StateGraph(SupportState)
        .add_node("intake", node_intake)
        .add_node("classify", node_classify)
        .add_node("reply", node_reply)
        .add_edge(START, "intake")
        .add_edge("intake", "classify")
        .add_edge("classify", "reply")
        .add_edge("reply", END)
        .compile()
    )


# =============================================================================
# 第 4 步：观察状态是怎么被"一步步攒起来"的
# =============================================================================
def demo_state_evolution(app, user_input: str) -> None:
    pretty.section("用户：%s" % user_input)
    print(pretty.dim("  ── 每一步之后，状态各字段变成了什么 ──"))

    for event in app.stream({"messages": [HumanMessage(content=user_input)]}, stream_mode="updates"):
        for node, update in event.items():
            print("  %s" % pretty.bold("节点 " + node))
            for k, v in update.items():
                print("      %-14s ← %s" % (k, pretty.yellow(str(v)[:70])))

    final = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print()
    print(pretty.bold("  最终状态："))
    for k, v in final.items():
        if k == "messages":
            print("      %-14s = [%d 条消息] 最后一条：%s" % (k, len(v), str(v[-1].content)[:50]))
        else:
            print("      %-14s = %s" % (k, v))
    return final


def demo_no_reducer_vs_reducer() -> None:
    """再强调一次覆盖 vs 合并的区别，用真实的图来跑。"""
    pretty.section("对比：同一个字段，加不加 reducer 差别有多大")

    class WithReducer(TypedDict):
        total: Annotated[int, operator.add]

    class WithoutReducer(TypedDict):
        total: int

    def bump(state) -> dict:
        return {"total": 1}

    for name, schema in (("加 operator.add", WithReducer), ("不加 reducer", WithoutReducer)):
        g = StateGraph(schema).add_node("bump", bump).add_edge(START, "bump").add_edge("bump", END).compile()
        # 连续跑三次，每次都从 {} 开始，看看到底累计了没有
        result = g.invoke({"total": 0})
        print("  %s：三个节点各返回 total=1 → 结果 total = %s" % (name, pretty.bold(str(result["total"]))))

    print()
    pretty.info("等等，上面两边结果看起来一样？因为我们只跑了一个节点。")
    print()

    # 这回真的串三个节点
    def make_chain(schema):
        return (
            StateGraph(schema)
            .add_node("a", bump)
            .add_node("b", bump)
            .add_node("c", bump)
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .add_edge("c", END)
            .compile()
        )

    a = make_chain(WithReducer).invoke({"total": 0})
    b = make_chain(WithoutReducer).invoke({"total": 0})
    print("  三个节点串联，每个都返回 total=1：")
    print("      加 operator.add  → total = %s  %s" % (pretty.green(str(a["total"])), "（累加了）"))
    print("      不加 reducer      → total = %s  %s" % (pretty.red(str(b["total"])), "（只剩最后一个）"))


def demo_invalid_key() -> None:
    """演示两种常见的书写错误会造成什么后果。

    ★ 这一段是"血泪教训"，请认真看 —— 尤其是第一种，它不报错！
    """
    pretty.section("两个必踩的坑：写错字段名 / 节点返回了非字典")

    # ---- 坑一：返回了状态里没定义的字段 ----
    def typo_node(state: SupportState) -> dict:
        # 假设你本来想写 user_level，手快打成了 user_lvl
        return {"user_lvl": "VIP", "turn": 1}

    g1 = (
        StateGraph(SupportState)
        .add_node("typo", typo_node)
        .add_edge(START, "typo")
        .add_edge("typo", END)
        .compile()
    )
    out = g1.invoke({"messages": []})

    print("  坑一：节点返回了拼错的字段名 user_lvl（正确是 user_level）")
    print("      结果 state 里有哪些键：%s" % pretty.bold(str(list(out.keys()))))
    print("      你写的 user_lvl 还在吗？%s" % pretty.red(str("user_lvl" in out)))
    print()
    pretty.warn("它被【静默丢弃】了 —— 不报错、不警告，数据就这么凭空消失了。")
    pretty.warn("这是 LangGraph 里最容易 debug 到崩溃的一类问题，务必记住这一点。")

    # ---- 坑二：节点返回的不是字典 ----
    def not_a_dict(state: SupportState):
        return "我以为我返回了状态更新"

    g2 = (
        StateGraph(SupportState)
        .add_node("bad", not_a_dict)
        .add_edge(START, "bad")
        .add_edge("bad", END)
        .compile()
    )
    print()
    print("  坑二：节点函数忘了返回字典")
    try:
        g2.invoke({"messages": []})
    except Exception as exc:
        print("      这一次会明确报错（算是好事）：")
        print("      %s: %s" % (pretty.red(type(exc).__name__), str(exc)[:110]))

    print()
    pretty.info("结论：State 的字段定义就是「数据契约」，往里面加字段必须同步改 TypedDict。")
    pretty.info("好处是节点之间传什么数据一目了然；代价是写错字段名不会报错，要自己小心。")


def demo_messages_state() -> None:
    """MessagesState：官方给的最常用状态简写。"""
    pretty.section("偷懒写法：MessagesState")

    def bot(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="收到：%s" % state["messages"][-1].content)]}

    g = StateGraph(MessagesState).add_node("bot", bot).add_edge(START, "bot").add_edge("bot", END).compile()
    out = g.invoke({"messages": [HumanMessage(content="你好")]})
    print("  MessagesState 等价于：")
    print("      class MessagesState(TypedDict):")
    print("          messages: Annotated[list[AnyMessage], add_messages]")
    print()
    print("  运行结果：%s" % [str(m.content) for m in out["messages"]])


def main() -> None:
    pretty.lesson_header(
        "05",
        "设计你自己的状态结构（State Schema 与 Reducer）",
        goals=[
            "掌握多字段 State 的设计方法，节点只返回自己改动的字段",
            "学会编写自定义 reducer（去重合并、取最大值等）",
            "理解覆盖 / 追加 / 累加 / 并集 四种常见合并语义",
            "知道节点返回未定义字段会报错，以及 MessagesState 的简写方式",
        ],
        prerequisites="第 01 课（reducer 基础）",
    )

    app = build_graph()
    graph_viz.show_structure(app, title="客服流程的图结构")

    demo_state_evolution(app, "你好，我是 VIP 会员，我的设备坏了，很急！")
    demo_no_reducer_vs_reducer()
    demo_invalid_key()
    demo_messages_state()

    pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. State 是"带合并规则的字典"：
         字段名 + 类型注解 + reducer（可选）

  2. reducer 就是一个函数：(旧值, 新值) -> 合并后的值
     写法：
         Annotated[list, operator.add]        追加
         Annotated[int, operator.add]         累加
         Annotated[list, 自定义函数]           任意规则
     不写 reducer 就是覆盖。

  3. 节点只返回自己改动的字段，不要返回完整状态 —— 职责清晰，也避免误覆盖。

  4. reducer 要能处理 None（字段第一次被写入时可能是 None）。

  5. 状态里没定义的字段不能返回，会直接报错。
     这是 LangGraph 帮你防笔误的机制，也逼你把"数据契约"想清楚。

  6. 只用 messages 一个字段时，可以用官方的 MessagesState 简写。

  7. 设计建议：
         该追加的（历史、日志）用 add_messages / operator.add
         该覆盖的（当前状态、标志位）不加 reducer
         该去重汇总的（标签、命中的规则）写自定义 reducer
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 给 SupportState 加一个 dict 类型的字段 `metrics`，用自定义 reducer
     把两次写入的字典"浅合并"（提示：{**old, **new}）。

  ② 加一个"重试次数"字段，写一个只增不减的 reducer，并让某个节点在
     次数超过 3 时返回一条提示消息（模拟"最多重试 3 次"）。

  ③ 思考题：为什么 reducer 要做成"纯函数"，而不能依赖外部变量？
     （提示：想想如果图被恢复重放，外部状态会怎样）
"""
    )


if __name__ == "__main__":
    main()
