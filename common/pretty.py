"""终端输出美化工具（纯标准库实现，不依赖第三方包）。

为什么要单独搞一个文件？
因为 LangGraph 是"图"程序，一次运行会产生很多中间状态。
如果只是 print()，初学者很难看清"哪一步是谁在执行、状态发生了哪些变化"。
这里的小工具就是为了把 Graph 的执行过程可视化出来。

注意：这些只是打印工具，与 LangGraph 的运行逻辑无关，可以放心跳过细节。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 一、让 Windows 命令行也能显示颜色
# ---------------------------------------------------------------------------
# Windows 的老控制台默认不解析 ANSI 转义序列（就是 \033[31m 这种），
# 执行一句空命令可以"唤醒"VT100 模式。这是 Windows 上的经典小技巧。
if sys.platform == "win32":  # pragma: no cover
    os.system("")

# 把标准输入输出都切成 UTF-8，避免中文和 emoji 在 Windows 控制台乱码。
# stdin 也要设：否则在部分 Windows 终端里，你输入的中文会被按 GBK 解码而变成乱码。
try:  # pragma: no cover
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# 设置 NO_COLOR=1 可以关闭所有颜色（方便把输出重定向到文件）
_COLOR_ENABLED = os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    """给文本套一层颜色。code 是 ANSI 颜色码。"""
    if not _COLOR_ENABLED:
        return text
    return "\033[%sm%s\033[0m" % (code, text)


# 常用颜色快捷函数
def dim(t: str) -> str:
    return _c(t, "2")  # 灰暗


def red(t: str) -> str:
    return _c(t, "31")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def blue(t: str) -> str:
    return _c(t, "34")


def magenta(t: str) -> str:
    return _c(t, "35")


def cyan(t: str) -> str:
    return _c(t, "36")


def bold(t: str) -> str:
    return _c(t, "1")


# ---------------------------------------------------------------------------
# 二、分节打印：让终端输出有层次
# ---------------------------------------------------------------------------
LINE_WIDTH = 78


def title(text: str) -> None:
    """打印课程大标题（用 ═ 包起来，最醒目）。"""
    print()
    print(cyan("═" * LINE_WIDTH))
    print(cyan(bold("  " + text)))
    print(cyan("═" * LINE_WIDTH))


def section(text: str) -> None:
    """打印小节标题。"""
    print()
    print(bold(yellow("▶ " + text)))
    print(dim("─" * LINE_WIDTH))


def info(text: str) -> None:
    print(blue("ℹ ") + text)


def ok(text: str) -> None:
    print(green("✔ ") + text)


def warn(text: str) -> None:
    print(yellow("⚠ ") + text)


def err(text: str) -> None:
    print(red("✘ ") + text)


def kv(key: str, value: Any) -> None:
    """打印一行"键: 值"，用于展示配置。"""
    print("  %s %s" % (dim(key + ":"), value))


# ---------------------------------------------------------------------------
# 三、把 LangGraph 的消息对象打印成人话
# ---------------------------------------------------------------------------
# LangGraph 里的消息都是 langchain_core.messages 下的对象：
#   SystemMessage  系统提示（给模型设定人设/规则）
#   HumanMessage   用户说的话
#   AIMessage      模型说的话（可能带 tool_calls，即"要求调用工具"）
#   ToolMessage    工具执行的结果（会被塞回消息列表给模型看）
_ROLE_STYLE = {
    "SystemMessage": ("system", magenta),
    "HumanMessage": ("用户", cyan),
    "AIMessage": ("AI  ", green),
    "ToolMessage": ("工具", yellow),
}


def _short(text: str, width: int = 100) -> str:
    """把长文本截断成一行，避免刷屏。"""
    text = " ".join(str(text).split())  # 把换行压成空格
    if len(text) <= width:
        return text
    return text[:width] + "…"


def show_message(msg: Any, prefix: str = "  ") -> None:
    """打印单条消息：角色 + 正文 +（如果有）工具调用信息。"""
    name = type(msg).__name__
    label, color = _ROLE_STYLE.get(name, (name, dim))

    # 1) 正文
    content = msg.content
    if isinstance(content, list):
        # 有的模型会返回结构化内容块（如 [{"type":"text","text":"..."}]），这里做个兼容
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    print("%s%s │ %s" % (prefix, color(label), _short(content)))

    # 2) 工具调用：AIMessage 上的 tool_calls 是"模型要求调用工具"的指令
    #    它的结构是 [{"name": "工具名", "args": {...}, "id": "调用ID"}, ...]
    for call in getattr(msg, "tool_calls", None) or []:
        print(
            "%s%s │ 请求调用工具 → %s(%s)"
            % (
                prefix,
                color("  ↳"),
                bold(call.get("name", "?")),
                yellow(_short(str(call.get("args", {})), 60)),
            )
        )


def show_messages(messages: Iterable[Any], limit: int | None = None, title_text: str | None = None) -> None:
    """打印整个消息列表（这就是 LangGraph 的"对话状态"长什么样）。"""
    msgs = list(messages)
    if title_text:
        print(dim("  ┌─ %s" % title_text))
    shown = msgs if limit is None or len(msgs) <= limit else msgs[-limit:]
    if shown is not msgs:
        print("  " + dim("… 前面还有 %d 条消息已省略 …" % (len(msgs) - len(shown))))
    for m in shown:
        show_message(m)
    if title_text:
        print(dim("  └─ 共 %d 条消息" % len(msgs)))


def show_state(values: dict, keys: list[str] | None = None) -> None:
    """打印图的"状态"(State)字典。

    这是理解 LangGraph 最重要的一件事：
    每个节点执行完都会返回一个"状态更新"，引擎再把它合并进全局状态。
    """
    print(dim("  ┌─ 当前状态 State"))
    for k, v in values.items():
        if keys and k not in keys:
            continue
        if isinstance(v, list) and v and hasattr(v[0], "content"):
            print("  │ %s = 消息列表(%d 条)" % (bold(k), len(v)))
        else:
            print("  │ %s = %s" % (bold(k), _short(repr(v), 90)))
    print(dim("  └─"))


def show_events(event: dict, node_names: set[str] | None = None) -> None:
    """打印 stream(stream_mode="updates") 吐出来的单个事件。

    事件形如 {"节点名": {"字段": 更新值}} —— 说明"哪个节点刚刚跑完、改了什么"。
    """
    for node, update in event.items():
        tag = green("● 节点执行完毕: " + node) if node in (node_names or set()) else dim("● " + node)
        print("  " + tag)
        if isinstance(update, dict):
            for k, v in update.items():
                print("      %s ← %s" % (k, _short(repr(v), 88)))


def hr() -> None:
    print(dim("─" * LINE_WIDTH))


# ---------------------------------------------------------------------------
# 四、课程通用的"运行模式"小工具
# ---------------------------------------------------------------------------
def is_interactive() -> bool:
    """是否处于「交互模式」。

    约定：所有课程默认跑一段自动演示（方便快速看效果、也方便自动化验证）；
    加上 --interactive （或 -i）参数后，才会进入可以和你一问一答的循环。

    例如：
        python lessons/01_basic_chatbot.py                 # 自动演示
        python lessons/01_basic_chatbot.py --interactive   # 自己说话
    """
    return "--interactive" in sys.argv or "-i" in sys.argv


def lesson_header(number: str, title_text: str, goals: list[str], prerequisites: str = "") -> None:
    """打印课程开头：标题 + 学习目标 + 前置知识。"""
    title("第 %s 课 · %s" % (number, title_text))
    print()
    print(bold("  学完这一课，你将掌握："))
    for g in goals:
        print("    " + green("✓") + " " + g)
    if prerequisites:
        print()
        print(dim("  前置知识：" + prerequisites))
    print()


def try_hint() -> None:
    """提示交互模式怎么开。"""
    print()
    print(dim("  ── 想自己动手试试？运行下面这条命令进入交互模式 ──"))
    print(dim("     python %s --interactive" % _this_script()))
    print()


def _this_script() -> str:
    """猜测当前脚本的相对路径，纯粹为了打印提示方便。"""
    try:
        import os

        return os.path.relpath(sys.argv[0], os.getcwd()).replace("\\", "/")
    except Exception:  # pragma: no cover
        return "lessons/xxx.py"


def safe_input(prompt: str) -> str | None:
    """读一行输入；如果环境不支持交互（比如被重定向），返回 None。"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None
