"""把编译好的图"画"出来。

LangGraph 最大的好处之一：程序结构就是一张图，可以画出来肉眼检查。
这个文件提供两种画法：

1. show_structure(app)  纯文本列出「节点」和「边」，不需要任何额外依赖（推荐先看这个）
2. show_mermaid(app)     输出 Mermaid 代码，粘到 https://mermaid.live 就能看到漂亮的流程图
                         或者在支持 Mermaid 的 Markdown 编辑器（VS Code / Typora）里直接预览
"""

from __future__ import annotations

from typing import Any

from . import pretty


def show_structure(app: Any, title: str = "图结构") -> None:
    """用纯文本打印图的节点与边。

    app.get_graph() 返回的是一张"可绘制"的图对象：
        .nodes  -> {节点名: 节点信息}
        .edges  -> 边的列表，每条边有 source(起点) / target(终点) /
                   conditional(是否条件边) / data(条件边的分支名)
    """
    graph = app.get_graph()
    nodes = list(graph.nodes.keys())
    edges = list(graph.edges)

    pretty.section("%s（%d 个节点 / %d 条边）" % (title, len(nodes), len(edges)))

    print(pretty.bold("  节点（Node）："))
    for name, node in graph.nodes.items():
        # __start__ 和 __end__ 是 LangGraph 自动加的两个"虚拟节点"，
        # 分别代表"图的入口"和"图的出口"，不是我们自己写的节点。
        if name in ("__start__", "__end__"):
            mark = pretty.dim("（内置虚拟节点）")
        else:
            mark = ""
        kind = "节点" if name not in ("__start__", "__end__") else "端点"
        print("    • %-16s %s %s" % (pretty.bold(name), pretty.dim("[%s]" % kind), mark))

    print()
    print(pretty.bold("  边（Edge）："))
    for e in edges:
        arrow = "┈┈▶" if e.conditional else "──▶"
        label = ""
        if e.conditional and e.data:
            label = pretty.yellow("  （条件: %s）" % e.data)
        print("    %-16s %s %-16s%s" % (e.source, arrow, e.target, label))

    print()
    print(pretty.dim("  读法：START 是入口，END 是出口；带 ┈┈▶ 的是条件边，走哪条要看判断函数的结果。"))


def show_mermaid(app: Any) -> None:
    """输出 Mermaid 流程图源码。

    把它整段复制到 https://mermaid.live/ 左侧，就能立刻看到流程图。
    """
    graph = app.get_graph()
    try:
        mermaid = graph.draw_mermaid()
    except Exception as exc:  # pragma: no cover
        pretty.warn("生成 Mermaid 失败：%s" % exc)
        return

    pretty.section("Mermaid 流程图（复制下面内容到 https://mermaid.live 查看图形）")
    for line in mermaid.splitlines():
        print("  " + line)
    print()
    pretty.info("也可以用 graph.get_graph().draw_mermaid_png() 直接生成 PNG（需要额外依赖）。")


def show(app: Any, ascii_only: bool = True) -> None:
    """一步到位：打印结构 + Mermaid。"""
    show_structure(app)
    if not ascii_only:
        show_mermaid(app)
