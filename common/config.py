"""读取项目配置（.env 文件）。

设计目标：**零配置也能跑**。
如果你没有填任何 API Key，程序不会报错，而是自动降级为「离线模拟模型」，
让你先把 LangGraph 的原理学明白；等你要接真模型时，只要复制 .env.example 为 .env 填上 key 即可。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件在 common/ 下，往上一级就是根目录）
ROOT_DIR = Path(__file__).resolve().parent.parent

# 读取根目录下的 .env 文件（不存在也不会报错，只是什么都读不到）
load_dotenv(ROOT_DIR / ".env", override=False)

# 课程运行过程中产生的数据（比如 SQLite 记忆库）都统一放这里
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _get(name: str, default: str | None = None) -> str | None:
    """读环境变量，顺便把空字符串当成"没设置"。"""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    """一份不可变的配置快照。"""

    provider: str  # 服务商：openai / anthropic / deepseek / ollama / custom
    api_key: str | None  # API 密钥
    base_url: str | None  # 接口基址（OpenAI 兼容服务需要）
    model: str | None  # 模型名

    @property
    def has_credentials(self) -> bool:
        """是否配置了可用的真实模型。

        规则很简单：
          - 用官方 OpenAI / Anthropic SDK 时，只看有没有 API Key；
          - 用自定义兼容接口（custom / deepseek / ollama ...）时，还需要有 base_url 和模型名。
        本地 Ollama 通常不需要 Key，所以 base_url + model 就够。
        """
        if self.provider in ("openai", "anthropic"):
            return bool(self.api_key)
        return bool(self.base_url and self.model)

    def describe(self) -> str:
        """给人类看的一句话描述。"""
        if not self.has_credentials:
            return "离线模拟模型（未配置 API Key，用规则模拟大模型行为）"
        return "%s / %s" % (self.provider, self.model or "(默认模型)")


def load_settings() -> Settings:
    """从环境变量组装出配置。每次都重新读，方便你在运行中途改 .env。"""
    provider = (_get("LLM_PROVIDER", "offline") or "offline").lower()

    # 不同服务商默认从各自的官方环境变量里取 Key，符合业界习惯
    default_key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)

    api_key = _get("LLM_API_KEY") or (_get(default_key_env) if default_key_env else None)

    # 模型名：优先用 LLM_MODEL，其次用对应服务商的专用变量
    model = _get("LLM_MODEL")
    if model is None and provider == "openai":
        model = _get("OPENAI_MODEL")
    if model is None and provider == "anthropic":
        model = _get("ANTHROPIC_MODEL")

    base_url = _get("LLM_BASE_URL")

    # 小贴心：如果填了 base_url 却没填 provider，就当作"OpenAI 兼容接口"处理
    if provider == "offline" and (base_url or api_key):
        provider = "custom"

    return Settings(provider=provider, api_key=api_key, base_url=base_url, model=model)


SETTINGS = load_settings()
