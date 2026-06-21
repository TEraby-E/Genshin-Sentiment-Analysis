# 运行与部署完全指南

本文讲清楚四件事：整个项目怎么跑、怎么用 Docker 起容器、怎么接外部 GPU 容器、
以及怎么调用全部五档打标功能。

## 0. 拓扑总览

```
                       ┌──────────────────────────────────────┐
   你的笔记本（无 GPU） │  app 容器 / 本地 uv 环境              │
                       │   ├─ Streamlit 看板 (:8501)           │
                       │   ├─ 分析脚本 / CLI                    │
                       │   └─ 智能路由 RouterAgent              │
                       └───────────────┬──────────────────────┘
                            DeepSeek HTTP │   │ LoRA HTTP（场景 B）
                                          ▼   ▼
                              api.deepseek.com   外部 GPU 容器（云端 vLLM）
                                                 Qwen2.5-7B + LoRA 适配器
```

要点：笔记本只跑「轻」的部分；需要本地大模型时，算力由**外部 GPU 容器**提供，
本地通过 OpenAI 兼容协议远程调用，不在本机加载 7B 权重。

---

## 1. 本地直接运行（uv，最快）

```bash
# 安装依赖（按需选 extra）
uv sync                                  # 核心分析
uv sync --extra dashboard --extra llm    # 看板 + DeepSeek 打标

# 配置密钥：复制模板并填入真实 DEEPSEEK_API_KEY
cp .env.example .env

# 一键自检：确认各服务/模型是否可用
uv run python scripts/healthcheck.py

# 跑完整离线分析（产出 outputs/genshin_analysis.png 与终端结论）
uv run genshin-analyze

# 起看板（浏览器开 http://localhost:8501）
uv run streamlit run dashboard.py
```

数据：把 Kaggle 的 5 个 CSV 放进 `data/`（文件名见 `src/config.py` 的 `FILES`）。
没有真实数据时，看板与脚本自动退化到合成演示数据，开箱即跑。

---

## 2. Docker 容器运行（app 容器，CPU）

镜像只装看板 + DeepSeek + 离线 RAG，不含 torch/GPU，保持轻量。

```bash
cp .env.example .env          # 填好 DEEPSEEK_API_KEY，compose 会注入容器

docker compose build app
docker compose up app         # 浏览器开 http://localhost:8501
```

说明：
- `.env` 通过 `env_file` 注入，**不打进镜像**；密钥不入库、不进镜像层。
- `data/` 与 `outputs/` 以**数据卷**挂载，容器用的是宿主机的真实数据与已训练模型
  （例如蒸馏模型 `outputs/sentiment_clf.joblib`）。
- 改了代码重新构建：`docker compose build app`。

---

## 3. 连接外部 GPU 容器（场景 B：本地调用 + 云端算力）

笔记本无 GPU，因此本地 LoRA 大模型轨道由一个**外部 GPU 容器**提供——在 GPU 主机上
用 vLLM 把微调后的 Qwen 起成 OpenAI 兼容端点，本地只需把 `.env` 指过去。两种拓扑：

### 拓扑 A：远程云 GPU（推荐，笔记本无 GPU 时的标准做法）

在 AutoDL / RunPod 等平台的 GPU 实例上：

```bash
git clone <your-repo> && cd genshin-sentiment-analysis
uv sync --extra finetune                       # 用镜像自带的 CUDA torch
uv run python -m src.finetune.dataset_formatter
bash src/finetune/train_lora.sh                # 微调，产出适配器
pip install "vllm>=0.5"
bash scripts/serve_lora.sh                      # 起 OpenAI 兼容端点 :8000
cloudflared tunnel --url http://localhost:8000  # 映射出公网地址
```

回到本地 `.env` 填上隧道地址，路由自动接入云端模型：

```bash
LORA_SERVER_BASE_URL=https://xxxx.trycloudflare.com/v1
LORA_SERVER_MODEL=qwen2.5-7b-lora
LORA_SERVER_API_KEY=EMPTY
```

完整步骤见 [CLOUD_LORA.md](CLOUD_LORA.md)。

### 拓扑 B：你有一台带 NVIDIA GPU 的主机/服务器

直接用本仓库的 compose 把 GPU 服务和 app 起在同一网络（需装 NVIDIA Container Toolkit）：

```bash
# 在 GPU 主机上：先跑完训练，适配器在 outputs/finetune/qwen2.5-7b-lora
docker compose --profile gpu up lora-server     # vLLM 容器，:8000
docker compose up app                           # app 容器
```

同机时把 app 的 `.env` 设为 `LORA_SERVER_BASE_URL=http://lora-server:8000/v1`，
两个容器在 compose 网络里用服务名直连。

> 笔记本无 GPU，**不要**在本机启动 `lora-server`；它只在 GPU 主机上有意义。

接好后，本地一行代码都不用改：路由会自动多出一条 `lora_server` 轨道，难句优先走
你的自训 Qwen，而不是付费的 DeepSeek。原理是 `ServedLLMClient` 复用 OpenAI 兼容协议、
强制改写模型名。

---

## 4. 调用全部五档打标功能

| 档位 | 依赖 | CLI / 命令 | 看板入口 |
|---|---|---|---|
| 关键词基线 | 无（离线） | 见下 Python | 🏷️ 作品打标 → 关键词基线 |
| DeepSeek 语义打标 | `--extra llm` + API key | `scripts/ai_analyze.py` | → AI 语义打标 |
| 蒸馏本地模型 | `--extra llm`（训练时） | `scripts/train_sentiment.py` | → 本地模型（蒸馏） |
| 本地 LoRA 大模型 | `--extra finetune` + GPU | `train_lora.sh` | → 本地微调大模型 |
| 智能路由 | 自动按环境组装 | 见下 Python | → 🧭 智能路由 |

### 看板（最直观）

```bash
uv run streamlit run dashboard.py
# 「🏷️ 作品打标」页五档任选，粘贴文本或上传 CSV 即可
```

### 命令行脚本

```bash
# DeepSeek 端到端：清洗 → 打标 → 总结，产出 outputs/ai_analysis.csv
uv run python scripts/ai_analyze.py --sample 60

# 蒸馏：LLM 标注小样本 → 训练轻量分类器 → outputs/sentiment_clf.joblib
uv run python scripts/train_sentiment.py --sample 600

# 本地 LoRA：数据格式化 → 训练（在 GPU 机器上）
uv run python -m src.finetune.dataset_formatter
bash src/finetune/train_lora.sh
```

### Python API（可嵌入你自己的流程）

```python
import pandas as pd

# 1) 关键词基线（离线）
from src import aspect_sentiment
aspect_sentiment.tag_aspects("抽卡又歪了")              # → ['抽卡']

# 2) DeepSeek 语义打标（需 API key）
from src import text_pipeline
df = pd.DataFrame({"Comment_Content": ["这次又歪了", "剧情真好"]})
text_pipeline.analyze_comments(df, sample=2)            # 返回带 llm_sentiment 的 DataFrame

# 3) 蒸馏本地模型（需先训练出 joblib）
from src import sentiment_train
model = sentiment_train.load_model()
sentiment_train.predict(model, ["抽卡又歪了", "剧情太感人了"])

# 4) 本地 LoRA 大模型（需 GPU + 适配器；或经场景 B 走云端）
from src.sentiment_train import LocalLLMClassifier
LocalLLMClassifier().predict(["抽卡又歪了"])            # is_ready() 为真时可用

# 5) 智能路由（自动在上面几档间按难度分配 + 校验升档）
from src.agents import RouterAgent
router = RouterAgent.from_environment()
for r in router.tag(["剧情真好", "这次又歪了保底白给"]):
    print(r.track, r.sentiment, r.verified, r.escalations)
print(router.last_stats)                                # 各轨道处理量 = 算力分配
```

智能路由是推荐的统一入口：你不用手动选档，它会按评论难度把简单的留给离线轨道、
难句送语义轨道，并在校验不过时自动升档。配了 `LORA_SERVER_BASE_URL` 后，云端 Qwen
也会自动纳入候选轨道。

---

## 5. 随时自检

```bash
uv run python scripts/healthcheck.py            # 含 DeepSeek 实测
uv run python scripts/healthcheck.py --skip-api # 只看离线能力
```

逐项报告 DeepSeek / RAG / 蒸馏 / 云端·本地 LoRA / 路由 / 看板 的可用状态，
`SKIP` 表示未启用、`WARN` 表示配了但不通、`OK` 表示已验证可用。
