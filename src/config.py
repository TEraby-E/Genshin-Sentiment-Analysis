"""集中管理项目路径、数据文件名与分析参数。"""

from __future__ import annotations

from pathlib import Path

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---- 数据文件名 ----
# 注意：原始 Kaggle 文件名带时间戳前缀，下载后请重命名为下列简洁名称，
#       或在 data/ 下保留原名并在此处修改。
FILES = {
    "videos": "genshin_B_videos_clustered.csv",
    "video_keywords": "genshin_B_video_cluster_keywords.csv",
    "comments": "genshin_B_comments_clustered.csv",
    "comment_keywords": "genshin_B_comments_cluster_keywords.csv",
    "posts": "genshin_B_posts_cleaned.csv",
}

# ---- 分析参数 ----
# 负面情绪簇（基于数据集自带的聚类命名判定）
NEGATIVE_CLUSTERS = [
    "Community Controversy & Negative Sentiment",
    "Story Criticism & Operational Skepticism",
]

# 噪声主题簇编号（"混合内容"，混入大量泛娱乐视频）
NOISE_TOPIC_CLUSTER = 7

# 爆款定义：播放量分位数阈值
HIT_QUANTILE = 0.80

# 舆情趋势：单月最少评论数（样本过少的月份不纳入趋势）
MIN_MONTHLY_COMMENTS = 200

# 舆情预警：负面占比超过基线的倍数即触发
ALERT_MULTIPLIER = 1.5

# 中文字体候选（按优先级）
CJK_FONTS = ["WenQuanYi Zen Hei", "SimHei", "Noto Sans CJK SC", "Microsoft YaHei"]

# ---- LLM（主流 AI 模型 API）配置 ----
# 默认接入 DeepSeek（OpenAI 兼容协议）；通过 .env / 环境变量覆盖，密钥不写进代码。
import os  # noqa: E402

# 在读取下面的 LLM_* 变量之前就加载 .env，否则任何先导入 config 的模块都会拿到默认值
# 而忽略 .env（python-dotenv 属可选依赖，未安装时静默跳过，依赖系统环境变量）。
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
# 默认模型用官方现存的 deepseek-v4-pro（旧的 deepseek-chat 已不在 API 模型列表中）
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
# 单次请求打包的评论条数：越大越省 token，但要给模型留出输出空间
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "20"))
# 失败重试次数与温度
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# LLM 方面分类候选标签（与 aspect_sentiment.ASPECTS 对齐，作为语义级分类的取值域）
LLM_ASPECT_LABELS = ["剧情", "抽卡", "活动", "数值与机制", "运营", "社交与同人", "其他"]
LLM_SENTIMENT_LABELS = ["正面", "中性", "负面"]

# ---- RAG 检索增强（src/rag）配置 ----
# 嵌入模型：留空用离线哈希嵌入（零下载、无 GPU，默认）；填 sentence-transformers
# 模型名（如 BAAI/bge-small-zh-v1.5）则用语义嵌入，召回更准（需 rag extra）。
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL") or None
# 向量库后端：auto（优先 Chroma，缺失回退内存）/ memory（纯内存）/ chroma（强制本地持久化）
RAG_VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "auto")
# Chroma 持久化目录（仅 chroma 后端用到）
RAG_PERSIST_DIR = Path(os.getenv("RAG_PERSIST_DIR", str(OUTPUT_DIR / "rag_lore")))

# ---- 本地 LoRA 微调大模型（src/finetune、LocalLLMClassifier）配置 ----
# 基座模型：HuggingFace 仓库 id 或本地权重目录（首次加载按需下载到本地缓存）。
LORA_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# LoRA 适配器目录：train_lora.sh 的训练产物默认落在这里。
LORA_ADAPTER_DIR = Path(
    os.getenv("LORA_ADAPTER_DIR", str(OUTPUT_DIR / "finetune" / "qwen2.5-7b-lora"))
)

# ---- 云端自建 LoRA 推理端点（场景 B：本地调用 + 云端算力）----
# 用 vLLM 等把微调后的 Qwen 起成 OpenAI 兼容服务（可经 Cloudflare/Ngrok 映射出公网地址），
# 填下面的地址即可让智能路由的 lora_server 轨道接入你自己的云端模型；留空则该轨道自动关闭。
LORA_SERVER_BASE_URL = os.getenv("LORA_SERVER_BASE_URL") or None
LORA_SERVER_MODEL = os.getenv("LORA_SERVER_MODEL", "qwen2.5-7b-lora")
LORA_SERVER_API_KEY = os.getenv("LORA_SERVER_API_KEY", "EMPTY")
# 若服务端不支持强制 JSON 输出，设为 0/false 以去掉 response_format
LORA_SERVER_JSON_MODE = os.getenv("LORA_SERVER_JSON_MODE", "1").lower() not in ("0", "false", "no")
