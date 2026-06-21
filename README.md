# 原神 B 站玩家生态与舆情分析

[![CI](https://github.com/<your-org>/genshin-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/genshin-sentiment-analysis/actions/workflows/ci.yml)

> 基于 Kaggle 公开数据集 [Genshin Impact Bilibili Video Dataset](https://www.kaggle.com/datasets/kelvinkeekwongyew/genshin-impact-bilibili-videodataset) 的端到端数据分析项目。从游戏外部社区数据出发，分析玩家关注焦点、内容生态与舆情趋势，并搭建一套可解释、可复现、有统计依据的舆情预警方法。

详细的方法论、统计检验过程与局限性讨论见 [docs/REPORT.md](docs/REPORT.md)。

## 项目亮点

- **数据质量审查**：识别并处理数据中约 35% 的主题噪声（混入的泛娱乐视频），用主题聚类分离原神核心内容；加载阶段做列存在性、空值率、日期解析率三类契约校验，避免脏数据静默流入下游分析。
- **玩家生态分析**：量化各内容主题的热度与互动表现，发现角色专题与二创是社区核心。
- **舆情情感监控**：将 40 万条评论的负面情绪按月聚合，发现版本相关的情绪峰值（峰值 21.5% vs 基线 ~10.7%）。预警机制提供两套互相印证的口径：
  - 双比例 z 检验：回答"峰值月份的上升是否统计显著，而非样本噪声"；
  - 滚动窗口 z-score：替代固定倍数阈值，能适应舆情基线本身随版本的缓慢漂移。
- **方面级情感分析**：在原数据集的粗粒度评论聚类之上，加一层关键词方面标签（剧情/抽卡/活动/数值机制/运营/社交），量化各方面的负面占比与关键词覆盖率，用覆盖率本身证明"为什么值得引入 LLM 做语义级方面分类"而不是停留在直觉判断。
- **内部工具看板（Streamlit）**：`dashboard.py` 把全部分析与监控能力封装成业务可直接上手的六个工具——作品打标（关键词基线 + AI 语义打标）、作者分类与增长分析（生命周期分群 + 唤回目标识别 + 主题增长标签）、内容爆点预测与总结（单视频爆款概率交互打分器 + 爆款共性 AI 总结）、竞品动态监测、榜单数据监控、舆情词云 AI 总结。无真实数据/抓取被风控时自动用演示数据开箱即跑。
- **竞品 / 榜单数据监控（B 站抓取）**：`src/scraper` 复刻 B 站公开接口的 WBI 签名算法，抓取排行榜、搜索结果、公开视频统计（仅公开非个人数据，自带限速 + 磁盘缓存 + 优雅降级）；`src/monitor.py` 把抓取结果整理成带时间戳的快照并做榜单变化检测（新上榜/排名变动）、竞品内容产出与热度聚合，为业务决策提供数据参考。
- **AI 文本工作流（调用主流模型 API）**：`text_pipeline.py` 实现「自动化清洗 → 语义级归类 → 舆情总结」三段式工作流，调用 DeepSeek（OpenAI 兼容协议，可一键切换到任意兼容服务）对原始非结构化评论做：规则清洗（去链接/@/回复前缀/重复字符）→ LLM 批量打标（情感极性 + 多方面分类，强制 JSON 输出 + 自动重试 + 非法标签兜底）→ LLM 舆情总结（核心议题、代表性原声、词云关键词、可执行运营建议）。解决关键词规则在口语化、反讽、表情符号场景下的覆盖盲区，密钥从 `.env` 读取、绝不入库。
- **下一代 AI 管线：RAG 反幻觉 + 本地 LoRA 微调**：在不破坏既有 uv 环境、CI 与看板的前提下新增两条进阶轨道。一是 RAG 检索增强，从数据集异步构建本地「梗 & 设定词典」向量库，调用大模型前先用混合检索器拦截评论中的社区黑话、检索释义并注入系统提示，让模型先理解梗再判断，避免按字面误判；向量库默认是零依赖的纯 numpy 内存版本，需要时可换成 ChromaDB。二是本地 LoRA 微调，把高置信的大模型标注转成 LLaMA-Factory 数据集，用 QLoRA 在消费级 GPU（如 RTX 4090, 24GB）上微调 Qwen2.5-7B，再由本地分类器离线推理。两条轨道的重依赖都延迟导入并收进可选依赖组，没有 GPU 的 CI 照常全绿。
- **智能路由编排 Agent**：把上述四条打标能力统一成可调度的轨道，由 RouterAgent 按评论难度分配算力。简单的评论走零成本轨道，含黑话或反讽的难句才进入语义轨道，并经过「检索、推理、校验」三角复核；校验不通过就沿成本阶梯逐级升档重判，把贵的算力只花在确实需要的样本上。校验者默认是零成本的规则实现，也可换成大模型评审员。拿不到 API、GPU 或模型的轨道会被自动跳过，离线时优雅退化，不影响 CI。
- **爆款预测建模**：用随机森林预测视频是否成为爆款（AUC ≈ 0.60），得出"爆款由内容质量主导、而非发布技巧"的业务结论。
- **创作者/内容主题生命周期分群**：基于 UP 主发布频次与活跃时长，划分单发/成长期/稳定期/沉寂期四类创作者，识别出"沉寂期"创作者历史中位播放反而高于"稳定期"——这类创作者是潜在的唤回/再合作目标，而不是简单地把资源都押在当前活跃的稳定创作者上；同时按月度产出增长率给内容主题打"增长/平稳/衰退"标签，指导内容资源倾斜方向。
- **跨界联动效果的因果推断**：用同月匹配 + 置换检验（而非简单看绝对数字）评估跨界联动公告相对同期普通公告的点赞数增量，控制版本节奏等随时间变化的混杂因素；处理组样本极小（n=6）时仍诚实报告效应量与置信区间的不确定性，而不是用大样本假设硬套统计检验。
- **A/B 实验评估框架**：实现样本量/检验功效计算（power analysis）与双样本显著性检验，用模拟数据验证方法本身的正确性——因为数据集是观察性数据，不包含真实随机分流实验，这个模块刻意做成可复用的通用框架，接入真实埋点转化数据即可直接使用。
- **工程规范性**：85+ 单元测试（合成数据 fixture + fake LLM/HTTP client + fake 嵌入，不依赖真实数据、网络或 GPU 即可在 CI 跑通——RAG 检索与微调数据格式化均有 mock 覆盖）、ruff 静态检查、mypy 类型检查、GitHub Actions CI 三件套全部接入。

## 数据规模

| 数据 | 规模 | 时间跨度 |
| --- | --- | --- |
| 视频 | 37,802 条 | 2024-07 ~ 2025-11 |
| 评论 | 40.8 万条（已聚类） | — |
| 官方帖子 | 1,607 条 | — |

## 环境与运行（使用 uv）

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境。

```bash
# 1. 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆项目并进入目录
git clone <your-repo-url>
cd genshin-sentiment-analysis

# 3. 一键创建虚拟环境并安装所有依赖（依据 uv.lock 精确还原）
uv sync

# 4. 准备数据
#    从 Kaggle 下载数据集，将 5 个 CSV 放入 data/ 目录
#    文件名需与 src/config.py 中 FILES 一致

# 5. 运行完整分析
uv run genshin-analyze
```

运行后会在终端打印各项分析结论（含统计检验结果），并在 `outputs/` 生成六图分析面板 `genshin_analysis.png`。

### 用 Docker 运行

```bash
cp .env.example .env          # 填入 DEEPSEEK_API_KEY，compose 会注入容器
docker compose build app
docker compose up app         # 浏览器开 http://localhost:8501
```

app 镜像是 CPU 版，含看板、DeepSeek 打标与离线 RAG；`data/` 与 `outputs/` 以数据卷挂载，密钥经 `.env` 注入而不打进镜像。改了代码用 `docker compose build app` 重建即可。

### 接外部 GPU 容器（本地大模型轨道）

笔记本无 GPU，所以本地 LoRA 大模型这条轨道由一个外部 GPU 容器提供，本地远程调用，不在本机加载 7B 权重。两种接法：

- **远程云 GPU（推荐）**：在 AutoDL / RunPod 上微调并用 vLLM 起服务，再用 cloudflared 或 ngrok 映射出公网地址。本地只需在 `.env` 填 `LORA_SERVER_BASE_URL=https://xxxx.trycloudflare.com/v1`。
- **自有 GPU 主机**：在该主机上 `docker compose --profile gpu up lora-server`，用本仓库的 vLLM 服务起端点；同机时本地 app 用 `LORA_SERVER_BASE_URL=http://lora-server:8000/v1` 直连。

接好后本地一行代码都不用改，智能路由会自动多出一条 `lora_server` 轨道，难句优先走你自训的 Qwen 而不是付费的 DeepSeek。云端部署的逐步命令见 [docs/CLOUD_LORA.md](docs/CLOUD_LORA.md)。

### 五档打标怎么调用

最省心的是智能路由，它按评论难度自动分配，不用手动选档：

```python
from src.agents import RouterAgent
router = RouterAgent.from_environment()
for r in router.tag(["剧情真好", "这次又歪了保底白给"]):
    print(r.track, r.sentiment, r.verified)   # 简单评论走离线轨道，难句走语义轨道
```

五档也能单独调：关键词基线 `aspect_sentiment.tag_aspects(text)`；DeepSeek 打标 `uv run python scripts/ai_analyze.py --sample 60`；蒸馏本地模型 `uv run python scripts/train_sentiment.py --sample 600`；本地 LoRA 先 `uv run python -m src.finetune.dataset_formatter` 再 `bash src/finetune/train_lora.sh`。看板里这五档都在「🏷️ 作品打标」页，粘贴文本或上传 CSV 即可。

更完整的运行、容器化、外接 GPU 与打标调用方式见 [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md)。

## 开发与测试

```bash
# 安装开发依赖（pytest / ruff / mypy）
uv sync --extra dev

# 运行单元测试（基于合成数据 fixture，不需要下载真实数据集）
uv run pytest --cov=src --cov-report=term-missing

# 静态检查
uv run ruff check src tests
uv run mypy src

# 一键自检：逐项探测所有服务/模型是否可用（DeepSeek / RAG / 蒸馏 / 云端·本地 LoRA / 路由 / 看板）
uv run python scripts/healthcheck.py            # 含 DeepSeek 联网实测
uv run python scripts/healthcheck.py --skip-api # 只测离线能力，不联网
```

测试覆盖数据契约校验、四大核心分析函数、统计检验、方面级情感分析、生命周期分群、因果推断、A/B 实验框架，以及 RAG 混合检索（用 `tests/conftest.py` 的 fake 嵌入，无 GPU 亦可跑）与 LoRA 微调的数据格式化；均使用 `tests/conftest.py` 中的合成 fixture，CI 中无需访问真实数据、网络或 GPU 即可全部跑通（见 `.github/workflows/ci.yml`）。

## 项目结构

```
genshin-sentiment-analysis/
├── pyproject.toml          # 项目元数据与依赖声明（uv 读取）
├── uv.lock                 # 锁定的精确依赖版本（保证可复现）
├── README.md
├── docs/REPORT.md          # 完整分析报告：业务背景、方法论、结论、局限性
├── .github/workflows/ci.yml
├── data/                   # 原始数据（不纳入版本控制，见 .gitignore）
├── outputs/                # 分析产出的图表
├── notebooks/              # 探索性分析 notebook（可选）
├── tests/                  # 单元测试 + 合成数据 fixture
└── src/
    ├── __init__.py
    ├── config.py           # 路径与参数集中管理
    ├── validate.py         # 数据契约校验（列存在性/空值率/日期解析率）
    ├── data_loader.py      # 数据加载与清洗
    ├── analysis.py         # 四大分析模块
    ├── stats_tests.py      # 统计显著性检验、滚动 z-score 预警
    ├── aspect_sentiment.py # 方面级情感分析（关键词基线 + LLM 语义分类）
    ├── llm_client.py       # 主流 AI 模型 API 客户端封装（DeepSeek/OpenAI 兼容，JSON+重试）
    ├── text_pipeline.py    # AI 文本工作流：清洗 →（可选 RAG 接地）→ 归类 → 舆情/爆点总结
    ├── sentiment_train.py  # 舆情打标训练：LLM 标注 → 蒸馏轻量分类器 + 本地 LoRA 大模型分类器
    ├── rag/                # RAG 检索增强：向量库/嵌入/混合检索/异步词典入库（黑话接地，对抗幻觉）
    ├── finetune/           # 本地 LLM LoRA 微调：数据格式化（LLaMA-Factory JSONL）+ QLoRA 训练脚本
    ├── agents/             # 智能路由编排：按难度分配轨道 + 检索-推理-校验三角 + 成本阶梯升档
    ├── wordcloud_gen.py    # 词云核心（分词 + AI 关键词加权 + 渲染，脚本/看板共用）
    ├── sample_data.py      # 合成演示数据（无真实数据时让看板开箱即跑）
    ├── scraper/            # B 站公开数据抓取（WBI 签名 + 排行榜/搜索/视频统计）
    ├── monitor.py          # 竞品/榜单监控逻辑（快照 + 变化检测 + 降级兜底）
    ├── lifecycle.py        # 创作者/内容主题生命周期分群
    ├── causal_inference.py # 跨界联动效果的因果推断（同月匹配 + 置换检验）
    ├── ab_test.py          # A/B 实验评估框架（样本量计算 + 显著性检验）
    ├── visualize.py        # 可视化
    └── main.py             # 主入口
dashboard.py                # 内部工具看板（Streamlit）：作品打标/作者分类/爆点预测
```

## 展示素材生成脚本

`scripts/` 下是用于生成 HR/非技术展示材料的辅助脚本，不属于核心分析流程，依赖独立的可选依赖组：

```bash
# 生成展示用的单图表（数据规模、内容生态、舆情趋势等，输出到 Business-report-1/image/）
uv run python scripts/make_report_charts.py

# 生成舆情词云（透明背景 PNG）
uv sync --extra report   # 安装 wordcloud + jieba（含 llm 依赖）
uv run python scripts/make_wordcloud.py                          # 纯 jieba 词频（默认，离线）
uv run python scripts/make_wordcloud.py --source ai --sample 800 # AI 打标分组 + 语义关键词加权
uv run python scripts/make_wordcloud.py --source ai --by-aspect  # 额外按方面分别出词云
```

`--source ai` 用 `text_pipeline` 调用 DeepSeek 逐条打标，按 LLM 情感分组并用 AI 提炼的关键词加权放大，产出 `wordcloud_ai_negative.png` / `wordcloud_ai_positive.png`，与纯词频版本（`wordcloud_negative.png` 等）并存，便于对比。`--sample` 控制 API 成本。

## AI 文本工作流（调用主流模型 API）

对采集到的 / 原始表单导出的非结构化评论做自动化「清洗 → 归类 → 分析」，
协助优化舆情与内容分析工作流。需要配置 API key（默认 DeepSeek，OpenAI 兼容）：

```bash
# 1. 安装 llm 可选依赖（openai SDK + python-dotenv）
uv sync --extra llm

# 2. 配置密钥：复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
cp .env.example .env

# 3. 跑完整 AI 工作流（--sample 控制 API 成本；可用 --input/--text-col 接入任意表单 CSV）
uv run python scripts/ai_analyze.py --sample 60
```

产出：
- `outputs/ai_analysis.csv`：每条评论的清洗文本 + AI 情感极性 + 方面标签 + 判断依据；
- `outputs/ai_analysis.summary.json`：负面舆情的核心议题、代表性原声、词云关键词、可执行运营建议。

代码层面也可直接复用：

```python
from src import text_pipeline

# 端到端：清洗 + AI 归类，返回带 llm_sentiment / llm_aspects 的 DataFrame
analyzed = text_pipeline.analyze_comments(comments_df, sample=100)
# 舆情总结
summary = text_pipeline.summarize_opinions(neg_texts)
```

### 舆情打标训练（知识蒸馏）

全量打标走 API 不划算，因此用 LLM 标注小样本、蒸馏出一个离线免 API 的轻量分类器：

```bash
uv sync --extra llm
uv run python scripts/train_sentiment.py --sample 600   # LLM 标注→训练→评估→保存模型
```

教师是 DeepSeek，学生是字符级 TF-IDF + 逻辑回归（中文无需分词），产出 `outputs/sentiment_clf.joblib`。训练后看板「作品打标」页可选「本地模型（蒸馏·免费秒级）」，毫秒级全量打标。复用：

```python
from src import sentiment_train
model = sentiment_train.load_model()
labels = sentiment_train.predict(model, ["抽卡又歪了", "剧情太感人了"])  # → ['负面', '正面']
```

### RAG 检索增强：用领域知识接地，对抗黑话误判

玩家评论高度依赖社区黑话和时效性强的梗，比如「歪了」指抽卡没出想要的角色、「下水道」指某个角色强度被低估。通用大模型不了解这些社区内生的语义，很容易按字面把评论判错情感或方面。为此 `src/rag/` 在调用大模型之前先做一步检索增强：从数据集里构建一个本地的「梗 & 设定词典」向量库，分类前检索出评论中黑话的释义，注入到系统提示里，让模型先理解梗再做判断。

这套检索栈刻意做得很轻，默认不依赖任何外部服务。向量库默认用纯 numpy 实现的内存版本，可以完全离线运行，也能在 CI 里跑通；当词典规模变大或需要跨会话持久化时，再切换到 ChromaDB 的本地模式。嵌入默认用确定性的特征哈希实现，无需模型、外网或 GPU，需要更高召回质量时可以换成 sentence-transformers 的语义嵌入。检索器是稠密向量与稀疏 BM25 的混合，把两者的分数归一化后加权融合，这样既能召回语义相近的梗，也能精确命中低频的专有词。词典的语料来自评论聚类关键词、官方动态正文和高赞 UGC 评论，由 `ingestion.py` 异步切块、嵌入后灌入向量库。

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

只要传入 `retriever`，分类流程就会自动拦截评论里的黑话、检索释义并注入系统提示；不传则与原来的链路完全一致，对既有调用零侵入。

### 本地 LoRA 微调：QLoRA 进阶轨道

除了 TF-IDF 蒸馏这条轻量轨道，项目还提供一条更重的轨道：把大模型标注的高置信数据进一步微调进一个本地小模型（如 Qwen2.5-7B），得到一个比 TF-IDF 更懂语义和黑话、同样离线免 API 的分类器。训练和推理跑在消费级 GPU（如 RTX 4090, 24GB）上，用 4-bit QLoRA 让 7B 模型显存充裕、稳定可跑。

`dataset_formatter.py` 负责数据准备，它从已标注数据中筛出高置信样本——只保留情感和方面取值合法、且判断依据足够充分的条目——再转成 LLaMA-Factory 要求的 alpaca JSONL 格式，并自动生成可直接注册的 `dataset_info.json`；`scripts/build_finetune_dataset.py` 则能按主题分层抽样、调 DeepSeek 标注并切出留出评估集。训练脚本 `train_lora.sh` 基于 LLaMA-Factory 的 QLoRA 模板，集成 4-bit 量化、FlashAttention-2、梯度累积与分页 8-bit 优化器等高效设置。微调完成后用 `scripts/eval_lora.py` 在留出集上评估（准确率 / 宏 F1 + 反讽错例分析），驱动针对性补样的增量迭代。推理由 `LocalLLMClassifier` 封装，加载量化后的基座模型和 LoRA 适配器做情感分类；transformers、peft、torch 这些重依赖全部延迟到真正加载时才导入，没装也不会影响其它模块和 CI。

```bash
uv run python -m src.finetune.dataset_formatter
uv sync --extra finetune
bash src/finetune/train_lora.sh
```

```python
from src.sentiment_train import LocalLLMClassifier

clf = LocalLLMClassifier()
labels = clf.predict(["抽卡又歪了", "剧情太感人了"])  # → ['负面', '正面']
```

微调完成后，看板「作品打标」页会多出第四档「本地微调大模型」，与关键词基线、AI 语义打标、蒸馏本地模型并列；依赖或适配器缺失时会给出明确引导，而不是静默失败。

### 智能路由编排：按难度分配算力，配合检索-推理-校验三角

上面四条打标轨道各有成本和精度的取舍，与其让用户手动挑选，不如交给一个路由 Agent 按评论难度自动分配。`src/agents/` 把四条能力统一封装成可调度的「轨道」，由 `RouterAgent` 统筹。它先用一套零成本的离线规则给每条评论打难度分，依据是评论里黑话的数量、有没有反讽语气以及长度。简单的评论直接走便宜轨道，难句才会进入「检索、推理、校验」三角：先检索领域证据，再让大模型打标，最后由一个校验者复核结果是否可信。如果校验不通过，就沿着成本阶梯从关键词、蒸馏、本地 LoRA 到 RAG-DeepSeek 逐级升档重判，这样贵的算力只会花在确实需要的样本上。

校验者有两档。默认的 `HeuristicVerifier` 用情感词典的极性冲突和结果完整性来判断，零成本且结果确定；需要更高把握时可以换成 `LLMVerifier`，让大模型直接当评审员，更准但要多花一次调用。整个路由还会自适应环境：拿不到 API、GPU 或模型文件的轨道会被自动跳过，离线时优雅退化到关键词和蒸馏，不会崩溃。每次打标都会返回每条评论命中了哪条轨道、置信度多少、是否通过校验、升档了几次。

```python
from src.agents import RouterAgent

router = RouterAgent.from_environment()
results = router.tag(["剧情真好", "这次又歪了，保底白给"])
for r in results:
    print(r.track, r.sentiment, r.verified, r.escalations)
print(router.last_stats)
```

`from_environment` 会按当前环境自动组装可用轨道，缺少 API 或 GPU 时只用关键词和蒸馏；传入 `client` 或 `retriever` 即可启用 RAG-DeepSeek 轨道。看板「作品打标」页对应新增第五档「智能路由」，逐条展示命中轨道、置信、校验状态和升档次数，并用各轨道的处理量直观呈现算力是怎么分配的。

如果你把微调后的 Qwen 用 vLLM 部署成云端服务，路由还会自动多出一条 `lora_server` 轨道。只要在 `.env` 里填上云端地址，本地不用改任何代码，难句就会优先走你自建的模型而不是付费的 DeepSeek。这条「本地调用、云端算力」的链路靠的是 `ServedLLMClient` 包装现有的 OpenAI 兼容协议、强制改写模型名，完整的云端部署步骤见 [docs/CLOUD_LORA.md](docs/CLOUD_LORA.md)。

> 项目用到的全部 AI / LLM 应用工程技术，包括结构化输出、Prompt 工程、重试与降级、成本分层、知识蒸馏、检索增强、本地 LoRA 微调、智能路由编排和语义聚合等，详见 [docs/AI_ENGINEERING.md](docs/AI_ENGINEERING.md)。

## 竞品 / 榜单数据监控（B 站抓取）

抓取 B 站公开数据做竞品动态与榜单监控，为业务决策提供数据参考：

```bash
uv sync --extra scrape
uv run python scripts/monitor_run.py                       # 跑一轮榜单+竞品监控
uv run python scripts/monitor_run.py --keywords 原神 鸣潮 绝区零 --interval 2
```

- **榜单监控**：抓全站排行榜 → 落地带时间戳快照（`data/snapshots/`）→ 与上次对比，标出新上榜/排名变化。
- **竞品监测**：对一组竞品关键词搜索，聚合各自的内容产出量、总播放、Top 作品。
- **合规与降级**：只抓公开非个人数据，内置请求限速（`--interval`）与磁盘缓存；命中风控或无网络时自动降级到演示数据，流程不中断。「逆向」仅指复刻公开接口的 WBI 签名算法，不绕过登录或抓取私密数据。请遵守 B 站 robots/服务条款，合理控制频率。

## 内部工具看板（Streamlit）

把内容分析能力封装成业务可直接上手的可视化工具，为内容经营提供工具指导：

```bash
uv sync --extra dashboard                              # 仅看板（离线工具可用）
uv sync --extra dashboard --extra llm --extra scrape   # 叠加 AI + 抓取全部能力
uv run streamlit run dashboard.py
```

六个工具页：
- **🏷️ 作品打标**：粘贴或上传 CSV，提供五档打标方式——离线秒出的关键词基线、DeepSeek 的 AI 语义打标、蒸馏出的免费本地模型、GPU 上微调的本地大模型，以及按难度自动分配轨道并带校验升档的智能路由。输出结构化的情感与方面标签，智能路由还会标明每条命中的轨道和校验状态。
- **👤 作者分类与增长**：UP 主生命周期分群（单发/成长/稳定/沉寂）、潜在唤回目标识别、内容主题增长/衰退标签。
- **🚀 内容爆点：预测与总结**：模型 AUC 与特征重要性、单视频爆款概率交互打分器、爆款共性 AI 选题总结。
- **🔭 竞品动态监测**：对一组竞品关键词抓取，聚合内容产出量与热度并对比。
- **🏆 榜单数据监控**：抓取全站排行榜、落地快照、展示较上次的排名变化。
- **☁️ 舆情词云 AI 总结**：对评论批量做语义总结，输出议题、代表性原声、词云关键词与运营建议。

## 可深化方向

- 数据集的情感分组基于传统关键词聚类，颗粒度较粗。`aspect_sentiment.py` 已用关键词规则验证了细粒度方面分析的价值（覆盖率作为量化依据），并已在 `text_pipeline.py` 接入 LLM 做语义级方面分类与情感打标（见上文「AI 文本工作流」），替代关键词规则在口语化、表情符号场景下的覆盖盲区。进一步可用 LLM 标注的小样本蒸馏出一个轻量本地分类器，在全量 40 万条上降本提速。

- 爆款预测目前只用元数据特征（AUC ≈ 0.60），若想验证"内容质量主导"的结论，需要引入标题/封面/正文的语义特征做对照实验。
- 舆情预警目前是月度粒度，结合官方帖子发布时间做事件驱动的更细粒度（周/日）监控，是接近真实舆情运营场景的下一步。
- 生命周期分群目前基于公开内容数据（UP 主发布行为、主题产出量），是"玩家生命周期"的内容侧代理指标，并非第一方玩家账号行为数据；接入真实玩家留存/付费流水后，可以做更准确的玩家分群与 LTV 预测。
- 跨界联动的因果效应评估受限于样本量（n=6），结论方向可信但置信区间较宽；积累更多联动事件样本或设计真正的随机化 A/B 实验，能进一步收窄估计的不确定性。

## 技术栈
- 数据分析与统计底座：pandas, scipy, scikit-learn (随机森林/TF-IDF), 假设检验 (z-test/置换检验)
- 多智能体与 AI 模型：LangGraph, Qwen2.5-7B (LoRA 微调), DeepSeek-V3 API, Qdrant (向量检索)
- 数据流引擎与工程化：FastAPI, Streamlit, Cleanlab (弱监督去噪), uv (依赖锁定), pytest, GitHub Actions