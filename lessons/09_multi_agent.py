"""第 09 课 · 多个专家协作（Supervisor 多智能体架构）

===============================================================================
为什么一个 Agent 不够用？
===============================================================================
当你把所有工具、所有职责塞给一个 Agent 时，会出问题：

    - 工具太多（20+），模型选错工具的几率飙升
    - 系统提示词太长，什么都要交代，反而什么都不精
    - 一个环节出错，整条链路都要重来

解决思路和人类组织一样：**分工**。
把大任务拆给几个各有所长的专家，再安排一个"主管"来调度。

===============================================================================
Supervisor（主管）模式
===============================================================================
                     ┌──────────────┐
                     │  supervisor  │◀─────────┐
                     │ （主管/调度） │          │
                     └──────┬───────┘          │
              决定"下一步谁干活" │               │
          ┌─────────────┬─────┴─────┐          │
          ▼             ▼           ▼          │
    ┌──────────┐  ┌──────────┐  ┌────────┐     │
    │researcher│  │  writer  │  │ FINISH │     │
    │ 研究员    │  │  写作员   │  │  收工  │     │
    └────┬─────┘  └────┬─────┘  └───┬────┘     │
         └─────────────┴────────────┘          │
                       │                        │
                       └────────────────────────┘

    主管自己不干活，只做两件事：
        1. 看当前进展，决定下一个该谁上
        2. 判断活干完了没有（FINISH）

===============================================================================
本课的关键技术点
===============================================================================
1. 结构化输出（with_structured_output）
   让模型返回一个**符合指定结构**的对象，而不是一段自由文本。
   调度这种场景最怕模型"自由发挥"，结构化输出能把它的选择限制在枚举值里。

2. 把子图当专家
   每个专家其实就是一个带工具的小 ReAct 图（第 08 课的子图用法）。
   —— 你会发现多智能体并不神秘，就是把第 02 课和第 08 课拼起来。

3. 共享消息列表
   专家们往同一个 messages 里写东西，主管就能看到"大家都干了什么"。
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field as PydanticField

import _bootstrap  # noqa: F401  —— 让 common 可以被导入
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from common import graph_viz, pretty
from common.llm import get_model

llm = get_model(quiet=True)


# =============================================================================
# 第 1 步：给每个专家配工具
# =============================================================================
@tool
def search_knowledge(query: str) -> str:
    """查询内部知识库，获取关于 LangGraph、Agent、框架的资料。用于调研和技术查阅。

    Args:
        query: 要查询的问题。
    """
    return (
        "知识库命中：LangGraph 是 LangChain 团队推出的有状态编排框架，"
        "核心抽象是 StateGraph（状态图）、Checkpointer（持久化）与 interrupt（人工介入）。"
    )


@tool
def web_search(query: str) -> str:
    """联网搜索最新资料。当需要了解最新动态、外部信息时使用。

    Args:
        query: 搜索关键词。
    """
    return "网络检索：2026 年多智能体（Multi-Agent）已成为复杂 AI 应用的主流架构之一。"


@tool
def format_report(content: str, style: str = "正式") -> str:
    """把材料整理成规范的书面报告。写作、总结、润色时使用。

    Args:
        content: 要整理的原始材料。
        style: 文风，可选"正式"或"轻松"。
    """
    return "【%s报告】\n%s\n（以上内容已按 %s 文风整理）" % (style, content, style)


# =============================================================================
# 第 2 步：造一个"专家工厂"
# -----------------------------------------------------------------------------
# 每个专家都是一个完整的 ReAct 子图（第 02 课那套东西）。
# 把它们封装成函数，是为了避免复制粘贴 —— 三个专家的结构完全一样，只有工具和人设不同。
# =============================================================================
class WorkerState(TypedDict):
    """专家的状态：只需要消息列表。"""

    messages: Annotated[list[BaseMessage], add_messages]


def make_worker(name: str, system_prompt: str, tools: list):
    """创建一个"专家子图"。

    返回的是一张编译好的图，可以直接当节点挂到父图上（第 08 课的用法 A）。
    """
    worker_llm = get_model(quiet=True).bind_tools(tools)

    def agent(state: WorkerState) -> dict:
        # 把系统提示词放在最前面，给这个专家设定角色
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        return {"messages": [worker_llm.invoke(messages)]}

    return (
        StateGraph(WorkerState)
        .add_node(name, agent)
        .add_node("tools", ToolNode(tools))
        .add_edge(START, name)
        .add_conditional_edges(name, tools_condition, {"tools": "tools", "__end__": END})
        .add_edge("tools", name)
        .compile(name=name)  # 给子图起个名字，方便调试时辨认
    )


RESEARCHER = make_worker(
    "researcher",
    "你是研究员，擅长查资料。请调用搜索工具核实信息，然后用简洁的要点说明你查到了什么。",
    [search_knowledge, web_search],
)

WRITER = make_worker(
    "writer",
    "你是写作员，擅长把材料整理成通顺的报告。请调用排版工具，输出一份结构清晰的报告。",
    [format_report],
)


# =============================================================================
# 第 3 步：主管（Supervisor）
# =============================================================================
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str  # 主管决定的"下一个执行者"
    done: Annotated[list, lambda old, new: (old or []) + (new or [])]  # 已完成的步骤


class Route(BaseModel):
    """主管的决策结构。

    ★ 用 Literal 把可选值**限定死**，模型就算想乱来也只能在这几个里挑。
      这是"让 LLM 输出可控"的最常用手段之一。
    """

    next: Literal["researcher", "writer", "FINISH"] = PydanticField(
        description="下一个应该执行的专家，如果任务已完成就填 FINISH"
    )
    reason: str = PydanticField(description="为什么这么决定，一句话说明")


SUPERVISOR_PROMPT = """你是一个调度主管，负责把任务分配给合适的专家，自己不要动手做事。

你的团队：
  - researcher：负责查资料、调研、核实信息
  - writer     ：负责把材料整理成报告

规则：
  1. 先看用户的要求，再看已经完成了哪些步骤。
  2. 如果资料还没查过，先派 researcher。
  3. 资料查完后，派 writer 整理成文。
  4. 都完成了，返回 FINISH。

已完成：{done}
"""


def supervisor_node(state: SupervisorState) -> dict:
    """主管节点：决定下一步派谁上。"""
    # 用结构化输出，让模型返回一个 Route 对象（而不是一段文字）
    router = llm.with_structured_output(Route)

    done = state.get("done", [])
    prompt = SUPERVISOR_PROMPT.format(done="、".join(done) if done else "（还没有人干活）")

    # 把提示词 + 完整对话历史一起给模型
    decision = router.invoke([SystemMessage(content=prompt)] + list(state["messages"]))

    pretty_decision = "主管决定：下一位 → %s（%s）" % (decision.next, decision.reason)
    print("  %s %s" % (pretty.yellow("🧭"), pretty_decision))

    return {
        "next": decision.next,
        "done": [decision.next] if decision.next != "FINISH" else [],
        "messages": [AIMessage(content="【主管】%s" % pretty_decision)],
    }


def route_from_supervisor(state: SupervisorState) -> str:
    """根据主管的决定，返回下一个节点名。"""
    return "FINISH" if state.get("next") == "FINISH" else state["next"]


def node_finish(state: SupervisorState) -> dict:
    """收尾节点：把最终结果整理出来。"""
    # 找出最后一个"专家产出"作为最终结果。
    # ★ 要跳过主管自己的调度记录（以【主管】开头），否则会把过程消息当成结果。
    final = ""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, AIMessage):
            continue
        text = str(msg.content)
        if text.startswith("【主管】"):
            continue
        final = text
        break

    if not final:
        final = "任务已完成，但专家没有留下文字产出。"

    return {"messages": [AIMessage(content="\n=== 最终交付 ===\n%s" % final)]}


def build_graph():
    return (
        StateGraph(SupervisorState)
        .add_node("supervisor", supervisor_node)
        .add_node("researcher", RESEARCHER)  # ★ 子图直接当节点
        .add_node("writer", WRITER)
        .add_node("finish", node_finish)
        .add_edge(START, "supervisor")
        .add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {"researcher": "researcher", "writer": "writer", "FINISH": "finish"},
        )
        # 专家干完活，回到主管那里继续调度 —— 这就是多智能体的"循环"
        .add_edge("researcher", "supervisor")
        .add_edge("writer", "supervisor")
        .add_edge("finish", END)
        .compile()
    )


# =============================================================================
# 第 4 步：跑起来
# =============================================================================
def run_task(app, task: str) -> None:
    pretty.section("任务：%s" % task)

    final_state = None
    for event in app.stream(
        {"messages": [HumanMessage(content=task)], "done": []},
        stream_mode="updates",
        config={"recursion_limit": 25},  # 多智能体容易转圈，设个上限防止死循环
    ):
        for node in event:
            if node not in ("supervisor",):
                print("      %s 节点 %s 执行完毕" % (pretty.green("✔"), pretty.bold(node)))
        final_state = event

    # 单独跑一次拿完整状态，方便展示结果
    result = app.invoke(
        {"messages": [HumanMessage(content=task)], "done": []},
        config={"recursion_limit": 25},
    )
    print()
    print("  %s" % pretty.bold("最终结果："))
    print("      " + str(result["messages"][-1].content).replace("\n", "\n      "))


def demo_recursion_limit() -> None:
    """演示：多智能体为什么会「转圈」，以及怎么防。"""
    pretty.section("常见坑：无限循环与 recursion_limit")

    print("  多智能体最经典的 bug 就是「转圈」：主管一直派活，专家一直干，永远不 FINISH。")
    print()
    print("  LangGraph 的保护机制：recursion_limit（默认 25 个超级步）")
    print("      超过上限会抛 GraphRecursionError，而不是把 CPU 跑满。")
    print()
    print("  常见原因与对策：")
    print("      · 主管的提示词没说清楚「什么算完成」  → 把终止条件写死")
    print("      · 专家干完活没有任何产出            → 让专家必须写一条总结消息")
    print("      · 状态里没有记录「谁已经干过」      → 像本课一样加 done 字段")

    # 故意跑一个不会终止的图，看报错
    from langgraph.errors import GraphRecursionError

    class LoopState(TypedDict):
        n: Annotated[int, lambda old, new: (old or 0) + 1]

    loop = (
        StateGraph(LoopState)
        .add_node("a", lambda s: {"n": 1})
        .add_node("b", lambda s: {"n": 1})
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", "a")  # 故意造一个环，且永不退出
        .compile()
    )

    print()
    print("  测试一个死循环的图（recursion_limit=6）：")
    try:
        loop.invoke({"n": 0}, config={"recursion_limit": 6})
    except GraphRecursionError as exc:
        print("      %s 被成功拦截：%s" % (pretty.green("✔"), str(exc).split("\n")[0][:90]))
    except Exception as exc:
        print("      抛出了 %s：%s" % (type(exc).__name__, str(exc)[:90]))


def main() -> None:
    pretty.lesson_header(
        "09",
        "多个专家协作（Supervisor 多智能体）",
        goals=[
            "理解 Supervisor 架构：主管调度 + 专家干活",
            "掌握 with_structured_output 让模型输出受控的结构化决策",
            "学会用「专家工厂函数」+ 子图复用出多个 Agent",
            "知道多智能体为什么会转圈，以及 recursion_limit 怎么防",
        ],
        prerequisites="第 02 课（ReAct）、第 08 课（子图）、第 05 课（自定义状态）",
    )

    from common.llm import model_status

    pretty.kv("当前模型", model_status())
    pretty.kv("专家团队", "researcher（研究员）、writer（写作员）")

    app = build_graph()
    graph_viz.show_structure(app, title="Supervisor 多智能体图")

    run_task(app, "帮我调研一下 LangGraph 是什么，然后写一份正式报告")
    demo_recursion_limit()
    print()

    pretty.try_hint()

    pretty.section("本课小结")
    print(
        """
  1. 多智能体的本质：**分工 + 调度**。
     一个 Agent 什么都会，往往什么都做不好；拆成专家反而更稳。

  2. Supervisor 模式的三要素：
         主管节点   只做决策，不干活（看进展 → 派活 / 收工）
         专家节点   各自带工具，本质是第 02 课的 ReAct 子图
         循环边     专家干完回到主管，主管再决定下一步
         终止条件   主管返回 FINISH → 走向 END

  3. 让 LLM 做决策时必须用结构化输出：
         class Route(BaseModel):
             next: Literal["researcher", "writer", "FINISH"]
         router = llm.with_structured_output(Route)
     用 Literal 把可选项限死，模型就不能"自由发挥"了。

  4. 专家 = make_worker(名字, 人设, 工具列表) 造出来的子图，
     直接 add_node 挂上去 —— 复用第二次体现了子图的价值。

  5. 必须防死循环：
         config={"recursion_limit": 25}
         同时在状态里记录 done，让主管知道"谁已经干过了"。

  6. 多智能体的其他常见拓扑：
         层级式   主管下面还有主管（团队套团队）
         网络式   专家之间可以直接交接（handoff）
         流水线   固定顺序，不用主管（其实就是第 07 课的并行/串行）
     用哪种取决于任务需不需要"动态决定下一步"。
"""
    )

    pretty.section("动手练习")
    print(
        """
  ① 加第三个专家 critic（评审员），负责挑报告的问题；把它加入 Route 的 Literal，
     并让主管在 writer 之后派 critic，critic 之后再派 writer 修订。

  ② 给主管加一个"最多派几轮"的限制：在状态里加 round 字段，超过 3 轮直接 FINISH。

  ③ 把主管的两次决策打印出来对比（第一次派 researcher，第二次派 writer），
     观察 messages 里累积的信息是如何影响决策的。

  ④ 思考题：如果把三个专家的工具全部塞给一个 Agent，会发生什么？
     为什么"工具太多"会让效果变差？（提示：想想上下文长度和干扰项）
"""
    )


if __name__ == "__main__":
    main()
