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
- **爆款预测建模**：用随机森林预测视频是否成为爆款（AUC ≈ 0.60），得出"爆款由内容质量主导、而非发布技巧"的业务结论。
- **创作者/内容主题生命周期分群**：基于 UP 主发布频次与活跃时长，划分单发/成长期/稳定期/沉寂期四类创作者，识别出"沉寂期"创作者历史中位播放反而高于"稳定期"——这类创作者是潜在的唤回/再合作目标，而不是简单地把资源都押在当前活跃的稳定创作者上；同时按月度产出增长率给内容主题打"增长/平稳/衰退"标签，指导内容资源倾斜方向。
- **跨界联动效果的因果推断**：用同月匹配 + 置换检验（而非简单看绝对数字）评估跨界联动公告相对同期普通公告的点赞数增量，控制版本节奏等随时间变化的混杂因素；处理组样本极小（n=6）时仍诚实报告效应量与置信区间的不确定性，而不是用大样本假设硬套统计检验。
- **A/B 实验评估框架**：实现样本量/检验功效计算（power analysis）与双样本显著性检验，用模拟数据验证方法本身的正确性——因为数据集是观察性数据，不包含真实随机分流实验，这个模块刻意做成可复用的通用框架，接入真实埋点转化数据即可直接使用。
- **工程规范性**：60+ 单元测试（合成数据 fixture + fake LLM/HTTP client，不依赖真实数据或网络即可在 CI 跑通）、ruff 静态检查、mypy 类型检查、GitHub Actions CI 三件套全部接入。

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

## 开发与测试

```bash
# 安装开发依赖（pytest / ruff / mypy）
uv sync --extra dev

# 运行单元测试（基于合成数据 fixture，不需要下载真实数据集）
uv run pytest --cov=src --cov-report=term-missing

# 静态检查
uv run ruff check src tests
uv run mypy src
```

测试覆盖数据契约校验、四大核心分析函数、统计检验、方面级情感分析、生命周期分群、因果推断、A/B 实验框架，均使用 `tests/conftest.py` 中的合成 fixture，CI 中无需访问真实数据即可全部跑通（见 `.github/workflows/ci.yml`）。

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
    ├── text_pipeline.py    # AI 文本工作流：清洗 → 归类 → 舆情/爆点总结
    ├── sentiment_train.py  # 舆情打标训练：LLM 标注 → 蒸馏轻量本地分类器
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

> 项目用到的全部 AI / LLM 应用工程技术（结构化输出、Prompt 工程、重试/降级、成本分层、知识蒸馏、语义聚合等）见 [docs/AI_ENGINEERING.md](docs/AI_ENGINEERING.md)。

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
- **🏷️ 作品打标**：粘贴或上传 CSV，关键词基线（秒出）/ AI 语义打标（情感 + 方面）二选一，输出结构化标签与覆盖率。
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

pandas · numpy · matplotlib · scikit-learn · scipy · openai SDK（DeepSeek/OpenAI 兼容）· requests · streamlit · pytest · ruff · mypy

## License

MIT
