# 主应用镜像（CPU）：端到端分析 + Streamlit 看板 + RAG（离线档）+ DeepSeek 调用。
# 刻意不含 torch/GPU 依赖，保持镜像轻量；本地 LoRA 大模型推理走外部 GPU 容器
# （vLLM，见 docs/CLOUD_LORA.md 与 docker-compose.yml 的 lora-server 服务）。

FROM python:3.11-slim

# 复制 uv 静态二进制（多阶段，免在镜像里 pip 装 uv）
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# PyPI 索引：国内构建可在 build 时用 --build-arg 覆盖为镜像源以避免下载卡顿，例如
#   docker compose build --build-arg UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple app
ARG UV_DEFAULT_INDEX=https://pypi.org/simple

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX} \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 先只装依赖，最大化利用 Docker 层缓存（依赖没变就不重装）；
# --mount=type=cache 复用 uv 下载缓存，重建时不必再下一遍（需 BuildKit，compose 默认已启用）。
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra dashboard --extra llm

# 再拷源码并安装项目本身
COPY src ./src
COPY dashboard.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dashboard --extra llm

EXPOSE 8501

# 默认启动看板；data/ 与 outputs/ 由 docker-compose 以数据卷挂载进来
CMD ["uv", "run", "streamlit", "run", "dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
