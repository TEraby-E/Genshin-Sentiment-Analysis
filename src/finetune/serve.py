"""自包含的 OpenAI 兼容推理端点（场景 B）：不依赖 vLLM。

用 transformers + peft 加载微调后的 Qwen（复用已验证可跑的 LocalLLMClassifier），
对外暴露 `/v1/chat/completions`，供本地路由的 `lora_server` 轨道按 OpenAI 协议调用。

为什么不用 vLLM：vLLM 在 import 时会 patch torch 的 inductor，与较新/不匹配的
torch（如 CUDA 13 构建）冲突，直接崩在导入阶段（duplicate template name）。本项目
只需把一个 7B 模型起成 OpenAI 兼容端点服务短文本打标，用不到 vLLM 的高吞吐特性，
故自带一个最小实现：依赖面小（fastapi + uvicorn）、与训练用的 torch 同环境、完全可控。

重依赖（fastapi/uvicorn/torch/transformers）全部延迟到运行时导入，不影响 CI。

用法（云 GPU 上）：
    uv sync --extra finetune --extra serve
    uv run python -m src.finetune.serve --port 8000
然后用 cloudflared / ngrok 把 :8000 映射出公网，填进本地 .env 的 LORA_SERVER_BASE_URL。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Protocol

from .. import config

logger = logging.getLogger(__name__)


class _ChatGenerator(Protocol):
    """端点只依赖一个「消息 -> 文本」的生成器，便于注入假实现做单测。"""

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = ...,
        temperature: float = ...,
    ) -> str: ...


def build_chat_response(content: str, model: str) -> dict[str, Any]:
    """把模型生成的文本包成 OpenAI chat.completion 响应体（与官方字段对齐）。

    纯函数，无重依赖，可独立单测；usage 的 token 统计本地端点不强求，置 0 即可。
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def create_app(
    classifier: _ChatGenerator | None = None,
    *,
    served_model: str | None = None,
    api_key: str | None = None,
) -> Any:
    """构造 FastAPI 应用。classifier 为 None 时加载本地微调模型（首次较慢）。

    fastapi 延迟导入，使本模块的纯函数（build_chat_response）在未装 serve 依赖时也可被引用。
    """
    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as e:  # pragma: no cover - 给出可操作的提示，而非裸 ModuleNotFoundError
        raise SystemExit(
            "缺少 serve 依赖（fastapi）。请先安装（只装这两个纯 Python 包，不会动 torch）：\n"
            "    uv pip install fastapi uvicorn\n"
            "或直接用封装脚本：bash scripts/serve_lora.sh"
        ) from e

    served_model = served_model or config.LORA_SERVER_MODEL
    # 默认沿用 config 里的 key；为 "EMPTY" 或空时视为不鉴权（本地/隧道场景常见）
    api_key = config.LORA_SERVER_API_KEY if api_key is None else api_key

    if classifier is None:
        from ..sentiment_train import LocalLLMClassifier

        logger.info("加载本地微调模型用于服务（首次较慢，请耐心等待）…")
        clf = LocalLLMClassifier()
        clf.load()
        classifier = clf
        logger.info("✅ 模型就绪，端点开始监听")

    app = FastAPI(title="Genshin LoRA Endpoint", version="1.0")

    def _check_auth(authorization: str) -> None:
        if api_key and api_key != "EMPTY":
            if authorization != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": served_model}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": served_model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any], authorization: str = Header(default="")) -> Any:
        _check_auth(authorization)
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        max_tokens = int(body.get("max_tokens") or 512)
        temperature = float(body.get("temperature") or 0.0)
        # response_format 等 OpenAI 字段本端点不强制，模型经微调已稳定输出 JSON
        content = classifier.generate_chat(
            list(messages), max_new_tokens=max_tokens, temperature=temperature
        )
        return build_chat_response(content, served_model)

    return app


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="自包含 OpenAI 兼容 LoRA 端点（无 vLLM）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model", default=config.LORA_SERVER_MODEL)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - 友好提示
        raise SystemExit(
            "缺少 serve 依赖（uvicorn）。请先：uv pip install fastapi uvicorn"
        ) from e

    app = create_app(served_model=args.served_model)
    logger.info(
        "端点已启动：http://%s:%d/v1（model='%s'）",
        args.host, args.port, args.served_model,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
