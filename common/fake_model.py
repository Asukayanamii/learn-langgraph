"""离线模拟大模型 —— 没有 API Key 也能跑通全部课程。

===============================================================================
为什么需要它？
===============================================================================
LangGraph 是"编排大模型的框架"。要学习它，你本来必须有一个能用的 LLM。
但很多人卡在第一步：注册账号、拿 Key、充值、担心花钱。

于是这里写了这个"假模型"：它实现了和真模型**完全一样的接口**，
所以 LangGraph 根本分不清它是真是假，图的运行逻辑一模一样。

它用规则模拟了大模型的三种行为：
    1. 普通回复        —— 你说话，它回话
    2. 工具调用        —— 判断该用哪个工具，并生成参数（tool_calls）
    3. 结构化输出      —— 按你给的数据结构返回结果（with_structured_output）

===============================================================================
学习提示
===============================================================================
**你不需要读懂这个文件**，它不是 LangGraph 的内容，只是一个"替身演员"。
真正该看的是 lessons/ 里的课程代码。

但它也顺便说明了一个非常重要的原理：
    "大模型"在程序里只是一个「吃消息列表、吐 AI 消息」的函数。
    所谓 Agent，就是围绕这个函数搭起来的一套循环和状态管理。
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterator, List, Optional, Sequence, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

# =============================================================================
# 词汇表：把"工具的英文名"翻译成"用户可能说的中文词"
# -----------------------------------------------------------------------------
# 真模型靠的是语义理解，这里只能靠查表。表格覆盖课程里会用到的所有场景。
# 每一项： 概念 -> (英文线索词, 中文线索词)
# =============================================================================
_SYNONYMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "weather": (("weather", "forecast", "temperature"), ("天气", "气温", "下雨", "温度", "冷不冷", "热不热", "闷不闷")),
    "search": (("search", "web", "google", "browse", "news"), ("搜索", "搜一下", "查一下资料", "百度", "网上", "最新消息", "新闻", "联网")),
    "calculate": (("calc", "calculator", "math", "compute", "arithmetic"), ("计算", "算一下", "算算", "等于几", "等于多少", "是多少")),
    "email": (("email", "mail", "send"), ("邮件", "发信", "发送给", "寄给")),
    "time": (("time", "clock", "date", "now"), ("几点", "时间", "日期", "星期几", "现在几点")),
    "note": (("note", "save", "remember", "store", "memo"), ("记住", "记下来", "保存", "记一笔", "备忘")),
    "document": (("read", "load", "file", "list", "readfile"), ("读取文件", "读一下文件", "查看文件", "列出文件", "打开文件")),
    "translate": (("translate", "translation"), ("翻译", "英文怎么说", "中文意思")),
    "knowledge": (("retrieve", "knowledge", "faq", "policy", "rag", "doc"), ("知识库", "资料库", "退货", "保修", "规定", "政策", "手册", "条款", "会员", "权益", "发票", "配送", "售后")),
    "database": (("sql", "database", "query", "table"), ("数据库", "查表", "订单表", "用户表")),
    "todo": (("todo", "task", "ticket", "issue"), ("待办", "任务", "工单", "事项", "记一下", "记一笔", "帮我记", "记录一下", "提醒我")),
    "write": (("write", "writer", "draft", "article", "compose"), ("写一篇", "撰写", "写作", "润色", "文章", "报告", "总结一下")),
    "research": (("research", "analyst", "investigate"), ("研究", "调研", "分析一下", "查资料")),
    "code": (("code", "python", "execute", "run"), ("写代码", "运行代码", "执行代码", "写个函数")),
}

# 常见城市，用于给"天气类工具"填充参数
_CITIES = (
    "北京 上海 广州 深圳 杭州 成都 重庆 武汉 西安 南京 苏州 天津 长沙 郑州 青岛 厦门 福州 "
    "济南 合肥 昆明 大连 宁波 无锡 沈阳 哈尔滨 石家庄 太原 贵阳 南宁 兰州 乌鲁木齐 海口 三亚 "
    "香港 澳门 台北 东京 首尔 纽约 伦敦 巴黎 柏林 莫斯科 悉尼 新加坡 洛杉矶 旧金山"
).split()

# 客套话前缀，清洗查询语句时用得上。
# 注意：长词必须排在短词前面（正则的 | 是从左往右匹配的），
# 否则「请问」会被「请」先吃掉，只剩一个「问」字。
_LEADING_NOISE = re.compile(
    r"^(请问|帮我|帮忙|麻烦你|麻烦|能不能|可不可以|可以|我想|我要|给我|替我|请|帮|再|又|还|也|另外|顺便|然后)+"
)

# 城市名后面常跟着的时间/语气词，提取城市时要把它们剪掉
_CITY_TAIL = re.compile(r"(今天|明天|后天|昨天|现在|目前|的|那边|这儿|那儿|市)+$")

# 枚举参数的"中文口语提示词"：用户很少直接说 add / delete，
# 所以看到这些词就当作选了对应的枚举值。
_ENUM_HINTS: dict[str, tuple[str, ...]] = {
    "add": ("添加", "新增", "记一下", "记一笔", "记下", "帮我记", "加一个", "加个"),
    "list": ("查看", "列出", "看看", "有哪些", "显示", "查询一下", "查一下"),
    "done": ("完成", "做完", "搞定", "打完勾", "标记完成"),
    "delete": ("删除", "删掉", "移除", "去掉", "清掉", "不要了"),
}

# 待办内容常见的动词前缀，提取"真正要做的事"时要剪掉
_CONTENT_VERBS = re.compile(
    r"^(添加待办|新增待办|添加|新增|记一下|记一笔|记下|帮我记|查看待办|查看|列出|看看|完成|做完|搞定|"
    r"删除|删掉|移除|去掉|清掉|帮我|把|将|待办事项|待办|任务|事项)+[：:\s]*"
)

# 同理，后缀里也常带着指代和动作词，一并剪掉
_CONTENT_TAIL = re.compile(r"((那|这)(条|个|项)?)?(待办事项|待办|事项|任务|事)?(删掉|删除|移除|去掉|清掉|完成|做完|搞定)?$")


def _text_of(message: BaseMessage) -> str:
    """把消息内容统一转成纯字符串（有的模型返回的是结构化内容块列表）。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def _clean(text: str) -> str:
    """去掉"帮我、请、能不能"这类客套前缀，保留真正的诉求，方便后续匹配。"""
    text = text.strip()
    for _ in range(3):  # 可能有多层前缀，比如"请你帮我..."
        new = _LEADING_NOISE.sub("", text)
        if new == text:
            break
        text = new
    return text.strip(" ，,。.！!？?、")


def _human_text(messages: Sequence[BaseMessage]) -> str:
    """找出最后一条用户消息的文本。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _text_of(msg)
    return ""


def _all_text(messages: Sequence[BaseMessage]) -> str:
    """把整段对话拼成一个大字符串（用于判断提示词里出现了哪些候选选项）。"""
    return "\n".join(_text_of(m) for m in messages)


# =============================================================================
# 工具元信息解析
# -----------------------------------------------------------------------------
# LangGraph / LangChain 里"工具"有好几种形态，这里统一成
# {"name": 名字, "description": 描述, "params": {参数名: 参数结构}}
# =============================================================================
def _tool_meta(tool: Any) -> dict:
    if isinstance(tool, dict):
        # OpenAI Function Calling 的 JSON Schema 形态
        fn = tool.get("function", tool)
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", "") or "",
            "params": (fn.get("parameters") or {}).get("properties", {}) or {},
        }

    name = getattr(tool, "name", None) or tool.__class__.__name__
    description = getattr(tool, "description", "") or ""

    params: dict = {}
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            params = schema.model_json_schema().get("properties", {}) or {}
        except Exception:
            params = {}
    return {"name": name, "description": description, "params": params}


_CJK_RUN = re.compile(r"[\u4e00-\u9fa5]{2,}")


def _keywords_for(meta: dict) -> set[str]:
    """从工具的名字和描述里，推断出「用户会用什么词来触发它」。"""
    haystack = (meta["name"] + " " + meta["description"]).lower()
    words: set[str] = set()

    # ① 查同义词表：工具名或描述里出现了某个概念，就把那个概念的中英文说法全加进来
    for concept, (en_words, zh_words) in _SYNONYMS.items():
        if concept in haystack or any(w in haystack for w in en_words):
            words.update(zh_words)
            words.update(en_words)

    # ② 工具名本身（get_weather -> get / weather）
    words.update(part for part in re.split(r"[_\-\s]+", meta["name"].lower()) if len(part) > 2)

    # ③ 从中文描述里切"二元词组"。
    #    中文没有空格，最简单的分词办法就是取相邻两字组合：
    #    "获取售后政策、保修规定、发票、会员权益、配送说明" 会切出
    #    「售后」「后政」「政策」「保修」「修规」「规定」「发票」「会员」…… 等词。
    #    虽然会混入"后政"这种无意义的词，但真正有用的词也一定会被包含进来，
    #    命中即加分，所以够用了。
    for run in _CJK_RUN.findall(meta["description"] or ""):
        for i in range(len(run) - 1):
            words.add(run[i : i + 2])

    return words


def _score_tool(meta: dict, user_text: str) -> int:
    """给某个工具打分：用户的这句话像不像要用它？"""
    lowered = user_text.lower()
    score = 0
    for word in _keywords_for(meta):
        if not word:
            continue
        if word in lowered or word in user_text:
            # 匹配到的词越长，说明越具体，给更高权重
            score += len(word)

    # 特殊规则：如果这个工具需要「算式」参数，而用户的话里确实含有一段算式，
    # 那几乎可以肯定就是要用它 —— 中文里"算"这个字太短，光靠关键词容易漏掉。
    param_names = " ".join(str(k) for k in (meta.get("params") or {})).lower()
    if any(k in param_names for k in ("expression", "expr", "算式", "公式")):
        if _find_expression(user_text):
            score += 20

    return score


# =============================================================================
# 参数填充：从用户的话里"猜"出工具需要的参数
# -----------------------------------------------------------------------------
# 真模型是"理解后生成参数"，这里是"正则 + 猜测"，
# 效果没那么聪明，但足够把课程演示清楚。
# =============================================================================
def _find_city(text: str) -> str | None:
    """从这句话里找出城市名。

    策略：先查内置城市表（最准），查不到再用正则从「XX天气 / XX的天气」里猜。
    """
    # ① 内置城市表：最可靠，直接命中
    for city in _CITIES:
        if city in text:
            return city

    # ② 正则兜底：抓「北京今天天气」这种句子里的城市部分，再剪掉时间词尾巴
    m = re.search(r"([\u4e00-\u9fa5]{2,6}?)(?:市|的天气|天气)", text)
    if m:
        candidate = _CITY_TAIL.sub("", m.group(1))
        if 2 <= len(candidate) <= 5 and candidate not in ("今天", "明天", "现在", "那里", "这边", "什么"):
            return candidate

    # ③ 英文城市名
    m = re.search(r"\b([A-Z][a-zA-Z]{2,15})\b", text)
    return m.group(1) if m else None


def _find_expression(text: str) -> str | None:
    """从话里抠出算式，例如 '帮我算一下 12*(3+4)' -> '12*(3+4)'。"""
    normalized = text.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")").replace("，", ",")
    candidates = re.findall(r"[-+*/^().\d\s]{3,}", normalized)
    best = ""
    for cand in candidates:
        cand = cand.strip()
        if not any(ch.isdigit() for ch in cand):
            continue
        if not any(op in cand for op in "+-*/^"):
            continue
        if len(cand) > len(best):
            best = cand
    return best or None


# 「我叫」后面跟着这些字，说明是在提问（我叫什么？），不是自我介绍
_NOT_A_NAME = ("什么", "啥", "谁", "什麼", "甚麼", "啥子", "哪位", "啥名")


def _find_name(text: str) -> str | None:
    """从「我叫小明」这类句子里取出名字。"""
    m = re.search(r"我(?:的名字)?(?:叫|是)\s*([\u4e00-\u9fa5A-Za-z0-9]{1,12})", text)
    if m:
        candidate = m.group(1)
        if not candidate.startswith(_NOT_A_NAME):
            return candidate
    m = re.search(r"(?:姓名|联系人)[:：]\s*([\u4e00-\u9fa5A-Za-z0-9]{1,12})", text)
    return m.group(1) if m else None


def _fill_args(meta: dict, user_text: str) -> dict:
    """按工具声明的参数名，逐个填值。"""
    args: dict = {}
    cleaned = _clean(user_text)

    for pname, pschema in (meta["params"] or {}).items():
        schema = pschema if isinstance(pschema, dict) else {}
        ptype = schema.get("type", "string")
        low = pname.lower()
        desc = str(schema.get("description", ""))

        # --- 枚举参数：先看用户有没有直接说出选项名，再看中文口语提示词 ---
        if "enum" in schema:
            chosen = None
            for option in schema["enum"]:
                if str(option) in user_text:
                    chosen = option
                    break
            if chosen is None:
                for option in schema["enum"]:
                    if any(hint in user_text for hint in _ENUM_HINTS.get(str(option), ())):
                        chosen = option
                        break
            args[pname] = chosen if chosen is not None else schema["enum"][0]
            continue

        # --- 布尔参数 ---
        if ptype == "boolean":
            args[pname] = not any(neg in user_text for neg in ("不要", "不用", "别", "取消"))
            continue

        # --- 数值参数：优先取"数量/limit/n"这类，其次取第一个数字 ---
        if ptype in ("integer", "number"):
            if any(k in low or k in desc for k in ("limit", "count", "top_k", "num", "数量", "条数", "个数")):
                m = re.search(r"(\d+)", user_text)
                args[pname] = int(m.group(1)) if m else 3
            else:
                m = re.search(r"(-?\d+(?:\.\d+)?)", user_text)
                args[pname] = float(m.group(1)) if m else 1
            continue

        # --- 字符串参数：按语义猜测 ---
        if any(k in low or k in desc for k in ("city", "城市", "地点", "位置")):
            args[pname] = _find_city(user_text) or "北京"
        elif any(k in low or k in desc for k in ("expression", "expr", "算式", "公式", "计算")):
            args[pname] = _find_expression(user_text) or "1+1"
        elif any(k in low or k in desc for k in ("name", "user", "姓名", "人名", "用户名")):
            args[pname] = _find_name(user_text) or "默认用户"
        elif any(k in low or k in desc for k in ("to", "收件人", "email", "邮箱", "地址")):
            m = re.search(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+", user_text)
            args[pname] = m.group(0) if m else "boss@example.com"
        elif any(k in low or k in desc for k in ("query", "question", "keyword", "查询", "关键词", "问题", "搜索词")):
            args[pname] = cleaned or user_text
        elif any(k in low or k in desc for k in ("text", "content", "body", "内容", "正文", "文本", "原文")):
            # 剪掉"再""帮我"这类语气前缀、"记一下""删掉"这类动词、以及"那条待办"这类后缀
            cleaned_content = _CONTENT_VERBS.sub("", _LEADING_NOISE.sub("", user_text))
            cleaned_content = _CONTENT_TAIL.sub("", cleaned_content)
            args[pname] = cleaned_content.strip(" ，,。.：:") or user_text
        elif any(k in low or k in desc for k in ("topic", "subject", "主题", "标题")):
            # 优先从「主题是 XXX」「标题：XXX」里抽出真正的主语，抽不到再退回整句
            m = re.search(r"(?:主题|标题|题目|subject)(?:是|为|叫|[:：])?\s*([^，,。；;！!？?]{1,40})", user_text)
            args[pname] = m.group(1).strip() if m else (cleaned or user_text)
        else:
            args[pname] = cleaned or user_text

    return args


# =============================================================================
# 离线模型本体
# =============================================================================
class OfflineChatModel(BaseChatModel):
    """一个"看起来像大模型"的替身，用规则生成回复。

    它继承自 BaseChatModel —— 这是 LangChain 所有聊天模型的抽象基类。
    只要实现了 _generate 和 _llm_type 两个方法，它就能被 LangGraph 当模型使用。
    （这本身就是个知识点：模型是可替换的零件，框架依赖的是接口，不是具体实现。）
    """

    # --- 可配置字段（pydantic 字段，因为 BaseChatModel 是 pydantic 模型）---
    bound_tools: list[dict] = Field(default_factory=list)  # bind_tools 绑定上来的工具
    delay: float = 0.0  # 模拟流式输出时每个字之间的停顿（秒）
    tag_name: str = "离线模拟模型"  # 显示用的名字

    @property
    def _llm_type(self) -> str:
        """BaseChatModel 要求的抽象属性，用于标识模型类型。"""
        return "offline-demo-model"

    # -------------------------------------------------------------------------
    # bind_tools：把工具"挂"到模型上
    # -------------------------------------------------------------------------
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "OfflineChatModel":
        """真模型调用这个方法后，会把工具定义一起发给 API 服务端；
        这里我们只是把工具的"名字和参数结构"记下来，供后面做规则匹配。
        """
        metas = [_tool_meta(t) for t in tools]
        return self.model_copy(update={"bound_tools": metas})

    # -------------------------------------------------------------------------
    # 核心：根据消息列表决定"下一步说什么"
    # -------------------------------------------------------------------------
    def _decide(self, messages: List[BaseMessage]) -> AIMessage:
        """这就是"大模型"的全部智慧：看一眼对话，决定回复文本或调用工具。"""
        last = messages[-1]
        user_text = _human_text(messages)

        # ---- 情况 1：上一步是工具执行的结果 -> 我们要做总结收尾 ----
        if isinstance(last, ToolMessage):
            return AIMessage(content=self._summarize_tool_results(messages))

        # ---- 情况 2：这一轮已经调用过工具了（防止死循环）----
        already_called = False
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                already_called = True
                break

        # ---- 情况 3：判断要不要调用工具 ----
        if self.bound_tools and not already_called and user_text:
            # 排除"记住了我的名字"这种纯记忆类输入，避免误触发"保存笔记"类工具
            if not self._looks_like_small_talk(user_text):
                best_meta, best_score = None, 0
                for meta in self.bound_tools:
                    score = _score_tool(meta, user_text)
                    if score > best_score:
                        best_meta, best_score = meta, score
                if best_meta is not None and best_score >= 2:
                    args = _fill_args(best_meta, user_text)
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": best_meta["name"],
                                "args": args,
                                "id": "call_%s_%d" % (best_meta["name"], int(time.time() * 1000) % 100000),
                                "type": "tool_call",
                            }
                        ],
                    )

        # ---- 情况 4：普通聊天回复 ----
        return AIMessage(content=self._chat_reply(messages, user_text))

    # -------------------------------------------------------------------------
    # 各种回复的生成逻辑
    # -------------------------------------------------------------------------
    @staticmethod
    def _looks_like_small_talk(text: str) -> bool:
        """判断是不是"自我介绍 / 打招呼"这类不需要调用工具的话。"""
        patterns = (
            r"我(?:的名字)?(?:叫|是)[\u4e00-\u9fa5A-Za-z0-9]{1,12}",
            r"^你好",
            r"^嗨",
            r"^hi\b",
            r"^hello\b",
            r"你是谁",
            r"你叫什么",
        )
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _summarize_tool_results(self, messages: List[BaseMessage]) -> str:
        """把最近一轮工具执行的结果，包装成一句像样的回复。"""
        results = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                results.append(msg)
            else:
                break
        results.reverse()

        if not results:
            return "工具似乎没有返回结果，请再试一次。"

        # 特例：第 04 课的「人工审批未通过」。真模型这时会礼貌地收手，
        # 而不是假装自己"查到了结果"，所以这里也照着模仿一下。
        for r in results:
            content = _text_of(r)
            if "人工审批未通过" in content or "用户拒绝" in content:
                return "好的，我不会执行 %s 这个操作。请问你希望我改做什么？" % getattr(r, "name", "该")

        pieces = []
        for r in results:
            name = getattr(r, "name", "tool")
            pieces.append("%s 返回：%s" % (name, _text_of(r)))

        lines = ["我调用工具查到了结果："]
        lines.extend("  • " + p for p in pieces)
        lines.append("")
        lines.append("（这条回复由「离线模拟模型」拼装，用于演示 Agent 的工具调用闭环。）")
        return "\n".join(lines)

    def _chat_reply(self, messages: List[BaseMessage], user_text: str) -> str:
        """普通对话的回复。会刻意覆盖课程里需要演示的几种场景。"""
        text = user_text.strip()
        lowered = text.lower()

        # ① 记忆演示：用户问"我叫什么"
        if re.search(r"我(?:的名字)?(?:叫|是)?(?:什么|啥|谁)", text) or "记得我叫" in text:
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    found = _find_name(_text_of(msg))
                    if found:
                        return "你叫%s。我是从之前的对话记录里看到的 —— 这正是 LangGraph 的 checkpointer 在起作用。" % found
            return "你还没有告诉我你的名字呢。你可以说「我叫小明」。"

        # ② 用户自我介绍
        name = _find_name(text)
        if name:
            return "你好，%s！我记住你的名字了。你可以在下一轮问我「我叫什么」来验证记忆是否生效。" % name

        # ③ 回忆类问题：把之前说过的话复述出来
        #    这一段专门为第 03 课「记忆」准备 —— 它证明历史消息确实被带进来了。
        if re.search(r"我(刚才|刚刚|之前|前面|上次|上回)|我说过|记得我(说过|讲过)", text):
            previous = [_text_of(m) for m in messages if isinstance(m, HumanMessage)]
            previous = previous[:-1]  # 去掉当前正在问的这一句
            if previous:
                lines = ["我翻了一下我们的对话记录，你之前说过："]
                for i, item in enumerate(previous[-5:], 1):
                    lines.append("  %d. %s" % (i, item))
                lines.append("")
                lines.append("（这些内容来自历史状态，说明 Checkpointer 确实把它存下来并带回来了。）")
                return "\n".join(lines)

        # ④ 常见寒暄
        if re.search(r"^(你好|您好|嗨|哈喽|hi|hello)", lowered):
            return "你好！我是一个用 LangGraph 搭建的助手。你可以问我问题，或者让我调用工具（比如查天气、算数学、查知识库）。"
        if "你是谁" in text or "你叫什么" in text:
            return (
                "我是用 LangGraph 搭建的教学用助手。当前跑的是「离线模拟模型」，"
                "目的是让你在没有 API Key 的情况下也能把 LangGraph 的全部机制跑通。"
            )

        # ⑤ 关于 LangGraph 的问题（课程里经常出现）
        if "langgraph" in lowered:
            return (
                "LangGraph 是一个用来编排「有状态、多步骤」AI 应用的框架。"
                "它把程序写成一张图：节点(Nodes)是工作单元，边(Edges)决定下一步去哪，"
                "状态(State)在节点之间流动。相比一条直线式的调用链，图结构能表达循环、分支、并行和人工介入。"
            )

        # ⑥ 兜底：给一个稳定、可预期的回答，方便观察图的运行
        return "收到。你刚才说的是：「%s」。（当前为离线模拟模型，接入真实 LLM 后这里会变成真正的智能回答。）" % text

    def _stream_reply(self, text: str) -> Iterator[ChatGenerationChunk]:
        """把整段文字切成小块吐出来，模拟大模型的流式输出。"""
        step = 2  # 每块 2 个字符，看起来比较顺滑
        for i in range(0, len(text), step):
            chunk = text[i : i + step]
            if self.delay:
                time.sleep(self.delay)
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    # -------------------------------------------------------------------------
    # 下面是 BaseChatModel 要求的两个实现：非流式 与 流式
    # -------------------------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """非流式调用：一次性返回完整回复。"""
        reply = self._decide(messages)
        # 给消息打上标记，方便在终端里一眼看出"这是模拟模型说的"
        used = [m["name"] for m in self.bound_tools]
        reply.response_metadata = {"model": self.tag_name, "bound_tools": used}
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用：一个字一个字地吐。

        LangGraph 的 stream_mode="messages" 就是靠这个接口拿到 token 的。
        """
        reply = self._decide(messages)

        # 如果要调用工具，就把工具调用信息包装成一个分块返回
        if getattr(reply, "tool_calls", None):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": __import__("json").dumps(call["args"], ensure_ascii=False),
                            "id": call["id"],
                            "index": i,
                            "type": "tool_call_chunk",
                        }
                        for i, call in enumerate(reply.tool_calls)
                    ],
                )
            )
            return

        yield from self._stream_reply(_text_of(reply))

    # -------------------------------------------------------------------------
    # 结构化输出：让模型返回一个符合 schema 的对象
    # -------------------------------------------------------------------------
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """真模型用 Function Calling 实现它；这里用规则来挑选项。

        用法和真模型完全一致：
            router = model.with_structured_output(MySchema)
            result = router.invoke(messages)   # -> MySchema 的实例
        """
        model = self

        def _invoke(input: Any, config: Any = None) -> Any:
            if isinstance(input, (list, tuple)):
                messages = list(input)
            else:
                messages = [HumanMessage(content=str(input))]
            return model._pick_structured(schema, messages)

        return RunnableLambda(_invoke)

    @staticmethod
    def _pick_option(options: List[Any], full_text: str, user_text: str) -> Any:
        """从候选列表里挑一个。

        规则：
          1. 如果提示词里写了「已完成：a, b」，就先把这些选项排除（第 09 课的多智能体路由靠它收敛）
          2. 剩下的选项里，谁的关键词在用户的话里命中最多就选谁
          3. 如果全都完成了，就选 FINISH 之类的终止选项
        """
        finish_words = ("FINISH", "END", "DONE", "STOP")

        # ① 解析「已完成」清单
        done: set[str] = set()
        m = re.search(r"已完成[:：]\s*([^\n]+)", full_text)
        if m:
            done = {w.strip() for w in re.split(r"[,，、\s]+", m.group(1)) if w.strip()}

        workers = [o for o in options if str(o).upper() not in finish_words]
        finishers = [o for o in options if str(o).upper() in finish_words]
        remaining = [w for w in workers if str(w) not in done]

        # ② 全都干完了 → 收工
        if workers and not remaining:
            return finishers[0] if finishers else options[-1]

        # ③ 在还没干的里面挑关键词命中最多的
        pool = remaining or workers or options
        best, best_hits = pool[0], -1
        for option in pool:
            keywords = _keywords_for({"name": str(option), "description": str(option), "params": {}})
            hits = sum(1 for word in keywords if word and (word in user_text or word in full_text))
            if hits > best_hits:
                best, best_hits = option, hits
        return best

    def _pick_structured(self, schema: Any, messages: List[BaseMessage]) -> Any:
        """按 schema 的字段逐个填值。枚举字段用"关键词命中"来挑。"""
        from pydantic import BaseModel

        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            # 非 pydantic schema（比如 TypedDict）就退化成"返回字典"
            return {"result": _human_text(messages)}

        text = _all_text(messages)
        user_text = _human_text(messages)
        fields: dict[str, Any] = {}

        for name, field in schema.model_fields.items():
            annotation = field.annotation
            options = None
            # 取出 Literal["a","b"] 或 Enum 里的候选值
            if getattr(annotation, "__origin__", None) is not None and str(annotation).startswith("typing.Literal"):
                options = list(annotation.__args__)
            elif isinstance(annotation, type) and hasattr(annotation, "__members__"):
                options = [m.value for m in annotation.__members__.values()]

            if options:
                fields[name] = self._pick_option(options, text, user_text)
            elif any(k in name.lower() for k in ("reason", "explain", "why", "理由", "原因")):
                # "理由"类字段：根据前面已经选出的决策，拼一句像样的说明
                chosen = fields.get("next") or fields.get("route") or "下一步"
                fields[name] = "根据当前的完成情况，接下来应该由「%s」继续推进任务。" % chosen
            elif annotation is bool or annotation == "bool":
                fields[name] = True
            elif annotation is int or annotation == "int":
                m = re.search(r"(\d+)", text)
                fields[name] = int(m.group(1)) if m else 0
            else:
                fields[name] = user_text

        try:
            return schema(**fields)
        except Exception:
            return schema.model_construct(**fields)

    # -------------------------------------------------------------------------
    # 让 print(model) 好看一点
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        tools = ", ".join(m["name"] for m in self.bound_tools) or "无"
        return "OfflineChatModel(工具=[%s])" % tools
