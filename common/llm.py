"""统一的"取模型"入口。

所有课程都通过 get_model() 拿模型，好处是：
  - 你只需要在 .env 里改一次配置，全部课程一起生效；
  - 没配 API Key 时自动降级成离线模拟模型，课程照样跑得通。

这在工程上叫「依赖注入」：图只依赖"一个聊天模型接口"，
至于背后是 GPT、Claude、DeepSeek 还是模拟器，图本身不关心。
这也是为什么你以后换模型不用改任何图代码。
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel

from .config import SETTINGS
from .fake_model import OfflineChatModel
from . import pretty

# 记录"是否已经提示过当前用的是哪个模型"，避免每次调用都刷屏
_announced = False


def use_offline() -> bool:
    """是否强制使用离线模拟模型（设置环境变量 OFFLINE=1 即可）。"""
    return os.environ.get("OFFLINE", "").strip() in ("1", "true", "yes")


def _offline_delay() -> float:
    """离线模型流式输出的"打字速度"，验证脚本里可以设成 0 加速。"""
    raw = os.environ.get("OFFLINE_STREAM_DELAY", "0.012")
    try:
        return float(raw)
    except ValueError:
        return 0.012


def _build_real_model() -> BaseChatModel:
    """按 .env 的配置，创建一个真实的聊天模型。"""
    provider = SETTINGS.provider

    if provider in ("openai", "anthropic"):
        # 官方 SDK：走各家默认的环境变量和地址
        from langchain.chat_models import init_chat_model

        model_name = SETTINGS.model
        spec = "%s:%s" % (provider, model_name) if model_name else provider
        return init_chat_model(spec, temperature=0)

    # 其余一律按 "OpenAI 兼容接口" 处理（DeepSeek / Moonshot / 通义 / 智谱 / Ollama ...）
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=SETTINGS.model or "gpt-4o-mini",
        api_key=SETTINGS.api_key or "not-needed",  # 本地 Ollama 不需要 Key，给个占位符
        base_url=SETTINGS.base_url,
        temperature=0,
    )


def get_model(*, quiet: bool = False, force_offline: bool = False) -> BaseChatModel:
    """拿到一个聊天模型。

    参数：
        quiet         是否安静模式（不打印"正在使用哪个模型"的提示）
        force_offline 强制使用离线模拟模型（比如"手写 Agent"那一课想纯粹演示原理）
    """
    global _announced

    if force_offline or use_offline() or not SETTINGS.has_credentials:
        model: BaseChatModel = OfflineChatModel(delay=_offline_delay())
        label = "离线模拟模型（无需 API Key）"
        if not quiet and not _announced:
            pretty.info("当前模型：%s" % pretty.bold(label))
            if not SETTINGS.has_credentials and not use_offline() and not force_offline:
                print(
                    pretty.dim(
                        "   （想接入真正的大模型？把 .env.example 复制成 .env 并填上 API Key 即可，详见 README）"
                    )
                )
            _announced = True
        return model

    model = _build_real_model()
    if not quiet and not _announced:
        pretty.info("当前模型：%s" % pretty.bold(SETTINGS.describe()))
        _announced = True
    return model


def bind_tools(model: BaseChatModel, tools: Sequence[Any]) -> BaseChatModel:
    """给模型绑定工具的小包装。

    为什么要包一层？因为离线模拟模型和真模型的 bind_tools 返回值类型不同，
    这里统一一下，让课程代码保持干净。
    """
    return model.bind_tools(list(tools))


def model_status() -> str:
    """返回当前模型状态的一句话描述，课程开头会打印出来。"""
    if use_offline():
        return "离线模拟模型（已通过 OFFLINE=1 强制指定）"
    return SETTINGS.describe()
