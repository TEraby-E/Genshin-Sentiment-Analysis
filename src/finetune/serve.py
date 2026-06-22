"""自包含的 OpenAI 兼容推理端点（场景 B）：只用 Python 标准库，零额外依赖。

用 transformers + peft 加载微调后的 Qwen（复用已验证可跑的 LocalLLMClassifier），
对外暴露 `/v1/chat/completions`，供本地路由的 `lora_server` 轨道按 OpenAI 协议调用。

为什么不用 vLLM / FastAPI：
- vLLM 在 import 时 patch torch 的 inductor，与较新/不匹配的 torch（如 CUDA 13）冲突，
  直接崩在导入阶段（duplicate template name）；
- FastAPI/uvicorn 又得在 uv / conda 环境里额外安装，反复触发依赖同步、可能动到脆弱的
  CUDA torch。

本端点只服务短文本打标，用不到高吞吐，故改用标准库 `http.server` 自实现——**不引入任何
新依赖**，直接跑在已装好 torch/transformers/peft 的那个环境里，彻底避开依赖同步问题。
torch/transformers 等重依赖仍延迟到加载模型时才导入，不影响 CI。

用法（云 GPU 上，无需任何 pip/uv 安装）：
    uv run --no-sync python -m src.finetune.serve --port 8000
    # 或直接用已装好 torch 的 python：python -m src.finetune.serve --port 8000
本地只给自己用时，用 SSH 端口转发把笔记本 localhost:8000 接到实例 localhost:8000，
本地 .env 填 LORA_SERVER_BASE_URL=http://localhost:8000/v1（详见 docs/CLOUD_LORA.md）。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _make_handler(
    classifier: _ChatGenerator, served_model: str, api_key: str
) -> type[BaseHTTPRequestHandler]:
    """生成绑定了模型/配置的请求处理器类。用锁串行化推理（单卡模型非线程安全）。"""
    gen_lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # 支持 keep-alive；配合每次必发 Content-Length

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if api_key and api_key != "EMPTY":
                return self.headers.get("Authorization") == f"Bearer {api_key}"
            return True

        def do_GET(self) -> None:  # noqa: N802 - http.server 约定的方法名
            path = self.path.rstrip("/")
            if path == "/health" or path == "":
                self._send(200, {"status": "ok", "model": served_model})
            elif path == "/v1/models":
                self._send(
                    200,
                    {"object": "list", "data": [{"id": served_model, "object": "model"}]},
                )
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802 - http.server 约定的方法名
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            if not self._authorized():
                self._send(401, {"error": {"message": "invalid api key"}})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "invalid json body"}})
                return
            messages = body.get("messages") or []
            if not messages:
                self._send(400, {"error": {"message": "messages is required"}})
                return
            max_tokens = int(body.get("max_tokens") or 512)
            temperature = float(body.get("temperature") or 0.0)
            try:
                with gen_lock:  # 串行推理，避免并发请求挤同一张卡
                    content = classifier.generate_chat(
                        list(messages), max_new_tokens=max_tokens, temperature=temperature
                    )
            except Exception as e:  # noqa: BLE001 - 把推理异常转成 500，不让连接挂死
                logger.exception("推理失败")
                self._send(500, {"error": {"message": f"inference error: {e}"}})
                return
            self._send(200, build_chat_response(content, served_model))

    return _Handler


def build_server(
    classifier: _ChatGenerator,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    served_model: str | None = None,
    api_key: str | None = None,
) -> ThreadingHTTPServer:
    """构造（但不启动）HTTP 服务。port=0 时由系统分配空闲端口，便于单测。"""
    served_model = served_model or config.LORA_SERVER_MODEL
    api_key = config.LORA_SERVER_API_KEY if api_key is None else api_key
    handler = _make_handler(classifier, served_model, api_key)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    classifier: _ChatGenerator | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    served_model: str | None = None,
    api_key: str | None = None,
) -> None:
    """加载模型（classifier 为 None 时）并阻塞式启动服务。"""
    served_model = served_model or config.LORA_SERVER_MODEL
    if classifier is None:
        from ..sentiment_train import LocalLLMClassifier

        logger.info("加载本地微调模型用于服务（首次较慢，请耐心等待）…")
        clf = LocalLLMClassifier()
        clf.load()
        classifier = clf
        logger.info("✅ 模型就绪，端点开始监听")

    httpd = build_server(
        classifier, host=host, port=port, served_model=served_model, api_key=api_key
    )
    logger.info("端点已启动：http://%s:%d/v1（model='%s'）", host, port, served_model)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断，关闭服务")
    finally:
        httpd.server_close()


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="自包含 OpenAI 兼容 LoRA 端点（纯标准库）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model", default=config.LORA_SERVER_MODEL)
    args = parser.parse_args()

    serve(host=args.host, port=args.port, served_model=args.served_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
