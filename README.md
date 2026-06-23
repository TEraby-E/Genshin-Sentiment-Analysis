# 原神 B 站舆情智能路由打标

> 面向原神 B 站社区评论的舆情打标系统：把非结构化评论喂进来，由一个多模型**路由 Agent** 按评论难度自动分配到最省的可行轨道（关键词 / 蒸馏 / 本地 LoRA / RAG-LLM），经「检索 → 推理 → 校验」三角复核后输出结构化的情感与方面标签。配套知识蒸馏、RAG 黑话接地与本地 LoRA 微调，并提供一个开箱即用的 Streamlit 看板。

## 核心能力

- **智能路由编排 Agent**：`src/agents/` 把多条打标能力统一封装成可调度的「轨道」，由 `RouterAgent` 按评论难度（黑话数量 / 是否反讽 / 长度）做算力分配。简单评论走零成本轨道，难句直接起步于语义轨道，并经「检索 → 推理 → 校验」三角复核；校验不过就沿成本阶梯（关键词 → 蒸馏 → 本地 LoRA → RAG-LLM）逐级升档重判，默认允许升到当前环境可用的最高档。整套路由手写实现、不依赖任何 Agent 框架。
- **知识蒸馏（离线轻量分类器）**：直接用 LLM 给全量 40 万条评论打标成本不可接受，因此用 DeepSeek 作为高精度标注源给小样本打标，蒸馏出一个字符级 TF-IDF + 逻辑回归的下游分类器，离线、免 API、毫秒级推理全量评论。
- **RAG 检索增强（对抗黑话误判）**：玩家评论高度依赖社区黑话（「歪了」「下水道」），通用大模型常按字面判错。`src/rag/` 在调用大模型前先从数据集构建本地「梗 & 设定词典」向量库，按单条评论检索黑话释义并注入系统提示，让模型先理解梗再判断。向量库默认是零依赖的纯 numpy 内存版，需要时可切换 ChromaDB。
- **本地 LoRA 微调（QLoRA）**：把高置信的大模型标注转成 alpaca JSONL，用 4-bit QLoRA 在消费级 GPU 上微调 Qwen2.5-7B，得到比 TF-IDF 更懂语义和黑话、同样离线免 API 的分类器；训练产物（适配器）由 `LocalLLMClassifier` 在进程内加载推理。微调脚本自包含，只用 `transformers + peft + bitsandbytes`，不引入 LLaMA-Factory 那套庞大且带原生库的依赖树，环境更稳。
- **校验者（critic）**：默认 `HeuristicVerifier` 用情感词典的极性冲突与结果完整性判断，零成本且确定；需要更高把握时可换成 `LLMVerifier`，让大模型直接当评审员。
- **环境自适应**：拿不到 API、GPU 或模型文件的轨道会被自动跳过，离线时优雅退化到关键词 / 蒸馏，绝不崩溃，CI 无需任何外部资源。
- **工程规范性**：合成数据 fixture + fake LLM/HTTP client + fake 嵌入，不依赖真实数据、网络或 GPU 即可跑通；ruff 静态检查、mypy 类型检查、测试全部接入。

## 快速开始

最省心的是智能路由，它按评论难度自动分配，不用手动选档：

```python
from src.agents import RouterAgent

router = RouterAgent.from_environment()
results = router.tag(["剧情真好", "这次又歪了，保底白给"])
for r in results:
    print(r.track, r.sentiment, r.verified, r.escalations)  # 简单评论走离线轨道，难句走语义轨道
print(router.last_stats)  # 各轨道处理量 / 校验通过数 / 升档次数
```

`from_environment` 会按当前环境自动组装可用轨道：缺少 API 或 GPU 时只用关键词和蒸馏；传入 `client` 或配置 `DEEPSEEK_API_KEY` 即可启用 RAG-LLM 轨道，传入 `retriever` 后该轨道会按单条评论注入黑话证据；本地有 GPU 且训练好 LoRA 适配器时自动多出一条 `lora` 轨道（进程内加载推理），难句优先走自训的 Qwen 而不是付费 API。

## 环境与运行

```bash
# 方案 A：使用 uv（项目主推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <your-repo-url>
cd genshin-sentiment-analysis
uv sync
cp .env.example .env
```

需要启用语义轨道时，在 `.env` 里填入 `DEEPSEEK_API_KEY`。API 模型、base URL、batch size 等都可通过环境变量覆盖。

### 看板：喂数据进去，自动打标并可视化

```bash
uv sync --extra dashboard               # 仅看板（离线退化到关键词/蒸馏轨道）
uv sync --extra dashboard --extra llm   # 叠加 DeepSeek，启用 RAG-LLM 轨道与三角校验
uv run streamlit run dashboard.py
# conda/venv 环境可用：streamlit run dashboard.py
```

看板只聚焦一件事：粘贴评论或上传 CSV → `RouterAgent` 自动选模型打标 → 可视化结果。展示三块内容：
- **逐条打标结果**：每条文本的情感 / 方面 / 命中轨道 / 置信 / 是否校验通过 / 升档次数；
- **情感分布图**：正面 / 中性 / 负面占比；
- **轨道分配 + 校验统计**：各轨道最终处理量（体现算力按难度分配）、校验通过率、升档次数。

### 抓取最新 B 站评论并直接分析近期舆情

```bash
python scripts/analyze_latest_bili.py --keyword 原神 --videos 5 --comments-per-video 30
```

这条链路会按关键词搜索近期 B 站视频，抓取评论，统一成下游需要的 `Comment_Content` 字段，然后交给 `RouterAgent` 用现有轨道打标并聚合报告。默认输出：

- `outputs/latest_bili_sentiment.md`：近期舆情 Markdown 报告；
- `outputs/latest_bili_tagged.csv`：逐条评论打标结果；
- `outputs/latest_bili_sentiment.json`：结构化汇总，便于接看板或自动化任务。

如果公开接口触发风控或评论召回不足，可在环境变量里设置 `BILI_COOKIE` 后重试。

### 用 Docker 运行

```bash
cp .env.example .env          # 填入 DEEPSEEK_API_KEY，compose 会注入容器
docker compose build app
docker compose up app         # 浏览器开 http://localhost:8501
```

app 镜像是 CPU 版，含看板、DeepSeek 打标与离线 RAG；`data/` 与 `outputs/` 以数据卷挂载，密钥经 `.env` 注入而不打进镜像。

### 本地跑微调大模型（lora 轨道）

把训练好的 LoRA 适配器放到 `outputs/finetune/qwen2.5-7b-lora`（或用 `LORA_ADAPTER_DIR` 指定），在本地有 GPU 的机器上 `uv sync --extra finetune` 后，智能路由会自动多出一条 `lora` 轨道，由 `LocalLLMClassifier` 在进程内加载基座 + 适配器做推理，离线、免 API。难句优先走自训的 Qwen 而不是付费的 DeepSeek。

```python
from src.sentiment_train import LocalLLMClassifier

clf = LocalLLMClassifier()                       # 默认读 outputs/finetune/qwen2.5-7b-lora
labels = clf.predict(["抽卡又歪了", "剧情太感人了"])  # → ['负面', '正面']
```

## 数据规模

| 数据 | 规模 | 时间跨度 |
| --- | --- | --- |
| 视频 | 37,802 条 | 2024-07 ~ 2025-11 |
| 评论 | 40.8 万条（已聚类） | — |
| 官方帖子 | 1,607 条 | — |

## 知识蒸馏：LLM 标注小样本 → 离线轻量分类器

```bash
uv sync --extra llm
uv run python scripts/train_sentiment.py --sample 600   # LLM 标注→训练→评估→保存模型
```

标注源是 DeepSeek，下游分类器是字符级 TF-IDF + 逻辑回归（中文无需分词），产出 `outputs/sentiment_clf.joblib`。复用：

```python
from src import sentiment_train

model = sentiment_train.load_model()
labels = sentiment_train.predict(model, ["抽卡又歪了", "剧情太感人了"])  # → ['负面', '正面']
```

## RAG 检索增强：用领域知识接地，对抗黑话误判

```bash
uv sync --extra rag
uv run python -m src.rag.ingestion --max-posts 300 --max-comments 2000
```

```python
import asyncio
from src import text_pipeline
from src.rag.ingestion import build_lore_dictionary

retriever = asyncio.run(build_lore_dictionary())
analyzed = text_pipeline.analyze_comments(comments_df, sample=100, retriever=retriever)
```

检索器是稠密向量与稀疏 BM25 的混合，把两者分数归一化后加权融合：既能召回语义相近的梗，也能精确命中低频专有词。词典语料来自评论聚类关键词、官方动态正文和高赞 UGC 评论，由 `ingestion.py` 异步切块、嵌入后灌入向量库。嵌入默认用确定性特征哈希（无需模型 / 外网 / GPU），需要更高召回时可换 sentence-transformers。`RagLLMTrack` 会按单条评论构造 RAG 上下文，再把上下文相同的评论分组批量调用，避免整批评论共享无关黑话释义。

## 本地 LoRA 微调：自包含 QLoRA 管线

把大模型标注的高置信数据微调进本地 Qwen2.5-7B，用 4-bit QLoRA 在消费级 GPU（24GB 起）上稳定可跑。完整流程：

```bash
# 1. 构建训练集：按主题分层抽样 → DeepSeek 标注 → 高置信筛选 → 切出留出评估集
uv run python scripts/build_finetune_dataset.py --sample 1800

# 2. 安装微调依赖（注意 torch 需与 GPU 的 CUDA 匹配）
uv sync --extra finetune

# 3. QLoRA 微调（自包含，只用 transformers + peft + bitsandbytes）
bash src/finetune/train_lora.sh
#    BATCH_SIZE=4 EPOCHS=3 bash src/finetune/train_lora.sh   # 显存足够时加大批量
#    FLASH_ATTN=fa2 bash src/finetune/train_lora.sh          # 装了 flash-attn 才用，否则默认 sdpa

# 4. 在留出集上评估（准确率 / 宏 F1 / 混淆矩阵 + 反讽错例分析）
uv run python scripts/eval_lora.py --predictor lora

# 5. 生成详细评估报告（Markdown + 逐条预测 CSV），并与基线对比
uv run python scripts/eval_lora.py --predictor lora --with-baselines --report-md
```

### 评估报告生成

报告生成挂在 `scripts/eval_lora.py` 上：评估的同时由 `--report-md` 顺带产出一份可直接展示的报告。默认写到 `outputs/finetune/`：

- `eval_report.md` —— 详细 Markdown 报告，含六节：总体指标、分类别精确率/召回率/F1、混淆矩阵、主要误差模式、与基线对比、错例样本；
- `eval_report.predictions.csv` —— 全部留出样本的逐条预测（文本 / 金标 / 预测 / 是否正确 / 疑似反讽），便于细看复盘。

```bash
# 出报告 + 与 keyword/distilled 基线对比（最能体现微调增益：负面召回从蒸馏的≈0% 提到微调后的高位）
uv run python scripts/eval_lora.py --predictor lora --with-baselines --report-md

# 只出报告、不跑基线（更快）
uv run python scripts/eval_lora.py --predictor lora --report-md

# 自定义报告路径、加列错例
uv run python scripts/eval_lora.py --predictor lora --report-md outputs/finetune/lora_v1.md --max-error-samples 50
```

关键参数：`--report-md [路径]` 触发报告生成（不带值用默认路径）；`--with-baselines` 额外评估 keyword/distilled 基线并在报告中对比（基线模型缺失时自动跳过，不报错）；`--max-error-samples` 控制报告里列出的错例条数（默认 30）。

- `src/finetune/dataset_formatter.py`：把已标注数据筛出高置信样本（情感与方面取值合法、判断依据充分），转成 alpaca JSONL 并生成 `dataset_info.json`，含训练 / 评估切分。
- `src/finetune/train_lora.py`：自包含 QLoRA，仅对 assistant 回答计损失（prompt 部分 label 置 -100），4-bit 量化 + 梯度检查点 + 分页 8-bit 优化器；重依赖延迟导入，不影响其他模块。
- `src/finetune/evaluate.py`：留出集评估 + 报告渲染（`build_markdown_report` / `write_predictions_csv`）+ 反讽错例分析，驱动针对性补样的增量迭代。

推理由 `sentiment_train.LocalLLMClassifier` 封装，加载量化基座 + LoRA 适配器；transformers / peft / torch 等重依赖全部延迟导入，没装也不影响其它模块。

## 智能路由编排：按难度分配算力，配合检索-推理-校验三角

各打标轨道在成本和精度上各有取舍，与其手动挑选，不如交给路由 Agent 自动分配。`RouterAgent` 先用零成本离线规则给每条评论打难度分（黑话数量 / 反讽语气 / 长度），简单评论走便宜轨道，难句进入「检索 → 推理 → 校验」三角：先检索领域证据，再让大模型打标，最后由校验者复核是否可信；校验不过就沿成本阶梯升档重判，默认升到当前环境可用的最高档或直到校验通过。

本地训练好 LoRA 适配器后，路由会自动把 `lora` 轨道纳入成本阶梯，难句优先走自训的 Qwen。项目用到的全部 AI / LLM 工程技术（结构化输出、Prompt 工程、重试与降级、成本分层、知识蒸馏、检索增强、本地 LoRA 微调、智能路由编排等）详见 [docs/AI_ENGINEERING.md](docs/AI_ENGINEERING.md)。

## 开发与测试

```bash
uv sync --extra dev
uv run pytest --cov=src --cov-report=term-missing
uv run ruff check src tests
uv run mypy src
```

测试覆盖数据契约校验、方面级情感、知识蒸馏、RAG 混合检索（fake 嵌入，无 GPU 亦可跑）、LoRA 微调的数据格式化与编码、智能路由编排；均使用 `tests/conftest.py` 中的合成 fixture，无需访问真实数据、网络或 GPU 即可跑通。

## 项目结构

```
genshin-sentiment-analysis/
├── pyproject.toml          # 项目元数据与依赖声明（uv 读取）
├── uv.lock                 # 锁定的精确依赖版本（保证可复现）
├── dashboard.py            # Streamlit 看板：喂数据 → 智能路由打标 → 可视化
├── docs/                   # 工程文档（AI 工程）
├── data/                   # 原始数据（不纳入版本控制，见 .gitignore）
├── outputs/                # 产出（蒸馏模型、微调数据集与评估集）
├── tests/                  # 单元测试 + 合成数据 fixture
├── scripts/
│   ├── build_finetune_dataset.py  # 分层抽样 + DeepSeek 标注 + 高置信筛选 + 切分
│   ├── train_sentiment.py         # 知识蒸馏：LLM 标注 → 训练轻量分类器
│   ├── eval_lora.py               # 留出集评估 + 报告生成（准确率 / 宏 F1 / 反讽错例 / Markdown 报告）
│   └── export_adapter.sh          # 把训练好的 LoRA 适配器提交进 git
└── src/
    ├── config.py           # 路径与参数集中管理（含 RAG / LoRA / 端点配置）
    ├── validate.py         # 数据契约校验（列存在性 / 空值率 / 日期解析率）
    ├── data_loader.py      # 数据加载与清洗
    ├── sample_data.py      # 合成演示数据（无真实数据时让看板开箱即跑）
    ├── aspect_sentiment.py # 方面级情感：关键词基线 + LLM 语义分类
    ├── llm_client.py       # AI 模型 API 客户端（DeepSeek/OpenAI 兼容，JSON + 重试）
    ├── text_pipeline.py    # AI 文本工作流：清洗 →（可选 RAG 接地）→ 归类
    ├── sentiment_train.py  # 知识蒸馏轻量分类器 + 本地 LoRA 大模型分类器
    ├── rag/                # RAG 检索增强：向量库 / 嵌入 / 混合检索 / 异步词典入库
    ├── finetune/           # 自包含 QLoRA 微调：数据格式化 + 训练 + 评估
    └── agents/             # 智能路由编排：难度画像 + 轨道阶梯 + 三角校验 + 升档
```

## 技术栈

- **AI / LLM 工程**：OpenAI 兼容模型 API（默认接入 DeepSeek，可通过环境变量切换）、Qwen2.5-7B + LoRA 微调（QLoRA / 4-bit）、RAG 混合检索（稠密向量 + BM25）、知识蒸馏、手写多模型路由 Agent 与 critic 校验
- **微调与推理**：transformers、peft、bitsandbytes、accelerate（自包含 QLoRA，不依赖 LLaMA-Factory）；适配器由 `LocalLLMClassifier` 在进程内加载推理，离线、免 API
- **数据与建模底座**：pandas、numpy、scikit-learn（TF-IDF / 逻辑回归）
- **可选检索栈**：纯 numpy 内存向量库（默认）/ ChromaDB + sentence-transformers
- **应用与工程化**：Streamlit、uv（依赖锁定）、pytest、ruff、mypy、Docker
```
