"""综合实战项目的工具集。

一个能用的助理需要哪些工具？这里给了一套完整的示范：

    get_current_time     查时间          —— 只读，安全
    calculator           算数            —— 只读，安全
    search_knowledge     查内部知识库     —— 只读，安全
    get_weather          查天气          —— 只读，安全
    todo                 管理待办事项     —— 有写操作，其中"删除"是危险的
    send_email           发邮件          —— 危险！必须人工审批

★ 注意每个工具的 docstring：它是写给**模型**看的说明书，
  不是写给人看的注释。写得越清楚，模型选错工具的概率越低。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

# 真实项目里这些数据会放在数据库；教学项目里用内存字典模拟，方便你直接看到效果。
_TODOS: dict[str, list[str]] = {}
_KNOWLEDGE_BASE = {
    "退货": "七天无理由退货：商品需保持完好并附带全部配件，联系客服提交申请即可。",
    "保修": "整机保修一年，主要部件保修两年；人为损坏不在保修范围内。",
    "发票": "下单时勾选「开具发票」，电子发票将在发货后 24 小时内发送到你的邮箱。",
    "会员": "会员分为普通、银卡、金卡三级，消费每满 1000 元升一级，金卡享受 9 折。",
    "配送": "默认使用顺丰发货，一线城市次日达，偏远地区 3-5 天。",
}

# 需要人工审批的工具名。assistant.py 会读取这个集合来决定是否弹审批。
DANGEROUS_TOOLS = {"send_email"}


# =============================================================================
# 只读工具（安全，直接执行）
# =============================================================================
@tool
def get_current_time() -> str:
    """查询当前的日期和时间。当用户问「现在几点」「今天几号」「今天星期几」时使用。"""
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return now.strftime("%Y-%m-%d %H:%M:%S") + "，星期" + weekday


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，支持加减乘除和括号。当用户需要做算术、算金额、算比例时使用。

    Args:
        expression: 数学表达式，例如 "128*37+56" 或 "(200-30)*0.8"。
    """
    # 只允许数字与四则运算符号，杜绝执行任意代码（安全第一）
    if not re.fullmatch(r"[\d\.\+\-\*/\(\)\s]+", expression):
        return "表达式只能包含数字、括号和 + - * / 运算符。"
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:
        return "计算出错：%s" % exc
    return "%s = %s" % (expression, value)


@tool
def search_knowledge(query: str) -> str:
    """查询公司内部知识库，获取售后政策、保修规定、发票、会员权益、配送说明等资料。

    Args:
        query: 要查询的问题或关键词。
    """
    for key, value in _KNOWLEDGE_BASE.items():
        if key in query:
            return value
    return "知识库中没有找到相关资料。目前收录的主题有：%s" % "、".join(_KNOWLEDGE_BASE)


@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的天气情况。当用户问天气、气温、是否下雨时使用。

    Args:
        city: 城市名称，例如「北京」。
    """
    table = {
        "北京": "晴，气温 25℃，微风",
        "上海": "多云转阴，22℃，傍晚有阵雨",
        "深圳": "雷阵雨，30℃，湿度较高",
        "杭州": "阴，20℃，适合外出",
    }
    return "%s：%s" % (city, table.get(city, "晴，24℃（模拟数据）"))


# =============================================================================
# 待办管理（有写操作，「删除」属于危险动作）
# =============================================================================
@tool
def todo(action: Literal["add", "list", "done", "delete"], content: str = "", user_name: str = "默认用户") -> str:
    """管理待办事项。可以添加、查看、完成或删除待办。

    当用户说「记一下要做某事」「看看我的待办」「这件事做完了」「删掉那条待办」时使用。

    Args:
        action: 操作类型。add=添加，list=查看，done=标记完成，delete=删除。
        content: 待办的具体内容，例如"买牛奶"。action 为 list 时可以留空。
        user_name: 是谁的待办列表，用于区分不同用户。
    """
    bucket = _TODOS.setdefault(user_name, [])

    if action == "add":
        if not content:
            return "请告诉我要添加什么内容。"
        bucket.append(content)
        return "已添加待办：%s。当前共 %d 条。" % (content, len(bucket))

    if action == "list":
        if not bucket:
            return "你目前没有待办事项。"
        lines = ["%d. %s" % (i, item) for i, item in enumerate(bucket, 1)]
        return "你的待办：\n" + "\n".join(lines)

    if action == "done":
        for i, item in enumerate(bucket):
            if content and content in item:
                bucket.pop(i)
                return "已完成并移除：%s" % item
        return "没有找到匹配的待办：%s" % content

    if action == "delete":
        # ★ 注意：删除类操作会被 assistant.py 拦截并要求人工审批
        for i, item in enumerate(bucket):
            if content and content in item:
                removed = bucket.pop(i)
                return "已删除待办：%s" % removed
        return "没有找到匹配的待办：%s" % content

    return "不支持的操作类型：%s。可用值：add / list / done / delete" % action


# =============================================================================
# 危险工具（需要人工审批）
# =============================================================================
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送一封电子邮件。当用户明确要求「发邮件」「发通知给某人」时使用。

    这是一个不可撤销的操作，执行前会请求用户确认。

    Args:
        to: 收件人邮箱地址。
        subject: 邮件主题。
        body: 邮件正文内容。
    """
    # 真实项目里会调用 SMTP / 邮件服务 API
    return "邮件已发送\n  收件人：%s\n  主题：%s\n  正文：%s" % (to, subject, body)


# 全部工具 + 注册表（万一你想手动按名字找工具）
ALL_TOOLS = [
    get_current_time,
    calculator,
    search_knowledge,
    get_weather,
    todo,
    send_email,
]
TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS}


def tool_catalog() -> list[tuple[str, str, bool]]:
    """返回工具清单，供 CLI 打印。每项是（名字, 说明, 是否需要审批）。"""
    catalog = []
    for t in ALL_TOOLS:
        first_line = (t.description or "").split("\n")[0]
        catalog.append((t.name, first_line, t.name in DANGEROUS_TOOLS))
    return catalog
