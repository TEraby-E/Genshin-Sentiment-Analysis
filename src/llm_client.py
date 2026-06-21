"""可复用的主流 AI 模型 API 客户端封装。

默认接入 DeepSeek（采用 OpenAI 兼容协议，因此直接复用官方 `openai` SDK，
只需替换 base_url 与 api_key），也可通过环境变量切换到任意 OpenAI 兼容服务。

设计目标（对应实习职责3「调用主流 AI 模型 API 对非结构化文本做清洗/归类/分析」）：
- 密钥从环境变量 / .env 读取，绝不写进代码；
- 统一的 JSON 输出 + 自动重试 + 失败兜底，让上层的文本工作流不用各自处理网络异常；
- 离线/未配置密钥时给出明确报错，而不是静默退化。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# 加载 .env（若安装了 python-dotenv 且存在 .env 文件）。缺失时不报错，
# 允许用户用真实环境变量注入密钥（如 CI / 生产环境）。
try:
    from dotenv import load_dotenv

    load_dotenv(config.PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv 属于 llm extra，可选
    logger.debug("未安装 python-dotenv，跳过 .env 加载，依赖系统环境变量")


class LLMNotConfiguredError(RuntimeError):
    """未配置 API key 时抛出，避免 LLM 流程静默退化为关键词规则。"""


def get_api_key() -> str:
    key = os.getenv(config.LLM_API_KEY_ENV)
    if not key:
        raise LLMNotConfiguredError(
            f"未找到环境变量 {config.LLM_API_KEY_ENV}：请在 .env 中配置 API key，"
            f"或 export {config.LLM_API_KEY_ENV}=... 后重试。"
        )
    return key


def get_client() -> Any:
    """构造 OpenAI 兼容客户端。延迟导入 openai，使核心分析流程无需该依赖即可运行。"""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - openai 属于 llm extra
        raise LLMNotConfiguredError(
            "未安装 openai SDK：请先 `uv sync --extra llm`。"
        ) from e

    return OpenAI(api_key=get_api_key(), base_url=config.LLM_BASE_URL)


class ServedLLMClient:
    """指向自建 OpenAI 兼容端点（如云端 vLLM 上的微调 Qwen）的客户端包装。

    包住一个 OpenAI 客户端并在每次 create 时强制改写 model 名，因此上层（chat_json /
    text_pipeline）无需改任何 model 参数即可把同一套打标流程切到自建端点（场景 B）。
    json_mode=False 时去掉 response_format，兼容不支持强制 JSON 的服务端。
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str = "EMPTY",
        model: str,
        json_mode: bool = True,
        _inner: Any | None = None,
    ) -> None:
        if _inner is not None:
            self._client = _inner
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        self._model = model
        self._json_mode = json_mode
        # 让 client.chat.completions.create 的链式访问落到本包装上
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> Any:
        kwargs["model"] = self._model  # 覆盖为自建端点对外暴露的模型名
        if not self._json_mode:
            kwargs.pop("response_format", None)
        return self._client.chat.completions.create(**kwargs)


def get_served_client() -> Any | None:
    """按 config.LORA_SERVER_* 构造自建端点客户端；未配置地址则返回 None（轨道自动关闭）。"""
    if not config.LORA_SERVER_BASE_URL:
        return None
    try:
        return ServedLLMClient(
            config.LORA_SERVER_BASE_URL,
            api_key=config.LORA_SERVER_API_KEY,
            model=config.LORA_SERVER_MODEL,
            json_mode=config.LORA_SERVER_JSON_MODE,
        )
    except ImportError:
        logger.warning("配置了 LORA_SERVER_BASE_URL 但未安装 openai SDK，云端 LoRA 轨道不可用")
        return None


def chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
) -> dict:
    """调用 chat completions 并强制 JSON 输出，返回解析后的 dict。

    带指数退避重试：网络抖动、限流、偶发非法 JSON 都会重试，
    超过上限仍失败则向上抛出，由调用方决定是否兜底到关键词基线。
    """
    client = client or get_client()
    model = model or config.LLM_MODEL
    max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:  # noqa: BLE001 - 网络/限流/非法JSON 统一重试
            last_err = e
            wait = min(2 ** (attempt - 1), 8)
            logger.warning(
                "LLM 调用第 %d/%d 次失败：%s（%ss 后重试）", attempt, max_retries, e, wait
            )
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError(f"LLM 调用在 {max_retries} 次重试后仍失败") from last_err
