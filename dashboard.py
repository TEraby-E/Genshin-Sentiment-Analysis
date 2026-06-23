"""智能路由打标看板 —— 喂数据进去，自动用最新的多模型路由 Agent 打标并可视化。

一个聚焦单一能力的工具看板：把非结构化的原神社区评论喂进来（手动粘贴或上传 CSV），
RouterAgent 按评论难度自动分配到最省的可行轨道（关键词 / 蒸馏 / 本地 LoRA / RAG-DeepSeek），
经「检索 → 推理 → 校验」三角复核，校验不过自动升档重判，最后把结果可视化：
- 逐条打标结果（情感 / 方面 / 命中轨道 / 置信 / 是否校验通过 / 升档次数）；
- 情感分布图；
- 轨道分配 + 校验统计（体现算力按难度分配）。

运行：
    uv sync --extra dashboard               # 仅看板（离线退化到关键词/蒸馏轨道）
    uv sync --extra dashboard --extra llm   # 叠加 DeepSeek，启用 RAG-LLM 轨道与三角校验
    uv run streamlit run dashboard.py

不可用的轨道（无 API / 无 GPU / 无微调适配器）自动跳过，离线也能跑，绝不崩溃。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import sample_data
from src.finetune import (
    DEFAULT_DATASET_PATH,
    build_sentiment_dataset_report,
    load_genshin_sentiment_jsonl,
)

st.set_page_config(page_title="原神舆情 · 智能路由打标", layout="wide", page_icon="🧭")


def llm_available() -> bool:
    try:
        from src import llm_client

        llm_client.get_api_key()
        import openai  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@st.cache_resource(show_spinner="构建 RAG 梗&设定词典…")
def get_lore_retriever():
    """构建混合检索器供路由「检索」一角取证；无数据/失败则返回 None（路由照常退化）。"""
    try:
        import asyncio

        from src.rag.ingestion import build_lore_dictionary

        return asyncio.run(build_lore_dictionary(max_posts=120, max_comments=400))
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner="装配可用打标轨道…")
def get_router(_has_llm: bool):
    """按当前环境组装路由 Agent（缓存：避免每次打标重复加载模型/词典）。"""
    from src import agents, llm_client

    client = llm_client.get_client() if _has_llm else None
    retriever = get_lore_retriever()
    return agents.RouterAgent.from_environment(client=client, retriever=retriever)


# ---- 侧边栏 ----
_has_llm = llm_available()
st.sidebar.title("🧭 智能路由打标")
st.sidebar.caption("多模型路由 Agent · 按难度自动分配算力 + 三角校验")
st.sidebar.write("AI 能力：" + ("✅ 已配置（含 RAG-DeepSeek 轨道）" if _has_llm else "⚠️ 未配置"))
if not _has_llm:
    st.sidebar.caption(
        "配置 .env 的 DEEPSEEK_API_KEY 并 `uv sync --extra llm` 后，"
        "启用语义轨道与 LLM 校验；当前仅离线轨道（关键词 / 蒸馏 / 本地 LoRA）可用。"
    )
st.sidebar.divider()
st.sidebar.markdown(
    "**轨道成本阶梯**\n\n"
    "关键词 → 蒸馏 → 本地 LoRA → 云端 LoRA → RAG-DeepSeek\n\n"
    "容易的评论走便宜轨道；难句（黑话 / 反讽）直接起步于语义轨道，"
    "校验不过再沿阶梯升档重判。"
)

st.title("原神舆情 · 智能路由打标")
st.caption("喂入评论文本 → 路由 Agent 自动选模型打标 → 可视化情感分布与算力分配。")


@st.cache_data(show_spinner="读取 genshin_sentiment.jsonl 并生成分析报告…")
def get_dataset_report(path: str):
    frame = load_genshin_sentiment_jsonl(path)
    return frame, build_sentiment_dataset_report(frame)


def _pct(value: float) -> str:
    return f"{value:.1%}"


st.divider()
st.subheader("数据集分析")
st.caption(f"默认分析文件：{DEFAULT_DATASET_PATH}")

dataset_path = st.text_input("分析文件路径", value=str(DEFAULT_DATASET_PATH))
try:
    dataset_frame, dataset_report = get_dataset_report(dataset_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"无法读取数据集：{exc}")
else:
    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("样本数", dataset_report.n_comments)
    dm2.metric("负面占比", _pct(dataset_report.negative_rate))
    dm3.metric("复合主题占比", _pct(dataset_report.multi_aspect_rate))
    dm4.metric("负面样本数", dataset_report.negative_count)

    st.info(dataset_report.overall)
    for insight in dataset_report.insights:
        st.write(f"- {insight}")

    left, right = st.columns(2)
    with left:
        st.subheader("情感结构")
        sent_df = pd.DataFrame(
            {
                "sentiment": list(dataset_report.sentiment_counts.keys()),
                "count": list(dataset_report.sentiment_counts.values()),
            }
        ).set_index("sentiment")
        st.bar_chart(sent_df)
        st.caption(
            "正面 {} · 中性 {} · 负面 {}".format(
                int(dataset_report.sentiment_counts.get("正面", 0)),
                int(dataset_report.sentiment_counts.get("中性", 0)),
                int(dataset_report.sentiment_counts.get("负面", 0)),
            )
        )

    with right:
        st.subheader("方面负面率排行")
        aspect_df = dataset_report.aspect_frame()
        if not aspect_df.empty:
            st.bar_chart(
                aspect_df.set_index("aspect")["negative_rate"].sort_values(ascending=False)
            )
        st.caption("按方面内负面率排序，越高代表该主题越容易引发负面评论。")

    st.subheader("方面分析明细")
    detail_cols = [
        "aspect",
        "mentions",
        "coverage_rate",
        "negative_mentions",
        "negative_rate",
        "negative_share_of_dataset",
        "negative_contribution_share",
        "delta_vs_overall",
    ]
    aspect_frame = dataset_report.aspect_frame()
    detail_df = (
        aspect_frame[detail_cols].copy()
        if not aspect_frame.empty
        else pd.DataFrame(columns=detail_cols)
    )
    if not detail_df.empty:
        detail_df["coverage_rate"] = detail_df["coverage_rate"].map(_pct)
        detail_df["negative_rate"] = detail_df["negative_rate"].map(_pct)
        detail_df["negative_share_of_dataset"] = detail_df["negative_share_of_dataset"].map(_pct)
        detail_df["negative_contribution_share"] = detail_df[
            "negative_contribution_share"
        ].map(_pct)
        detail_df["delta_vs_overall"] = detail_df["delta_vs_overall"].map(_pct)
    st.dataframe(detail_df, width="stretch", hide_index=True)

    st.subheader("方面 × 情感分布")
    if not dataset_report.sentiment_mix.empty:
        pivot = dataset_report.sentiment_mix.pivot_table(
            index="aspect", columns="sentiment", values="count", fill_value=0
        )
        st.bar_chart(pivot)
    else:
        st.info("没有可用于绘图的方面标签。")


# ---- 输入：手动粘贴 or 上传 CSV ----
src_choice = st.radio("输入来源", ["手动粘贴", "上传 CSV"], horizontal=True)
if src_choice == "手动粘贴":
    default_text = "\n".join(sample_data._COMMENT_TEXTS[:8])
    raw = st.text_area("每行一条评论", value=default_text, height=180)
    texts = [t for t in raw.splitlines() if t.strip()]
else:
    up = st.file_uploader("上传 CSV", type="csv")
    texts = []
    if up is not None:
        df_up = pd.read_csv(up)
        col = st.selectbox("选择文本列", df_up.columns)
        max_n = st.slider("打标条数（控成本/耗时）", 1, min(500, len(df_up)), min(50, len(df_up)))
        texts = df_up[col].dropna().astype(str).tolist()[:max_n]
        st.caption(f"读取 {len(texts)} 条待打标文本")


# ---- 打标 + 可视化 ----
if st.button("🚀 智能路由打标", type="primary") and texts:
    router = get_router(_has_llm)
    with st.spinner("路由分配 + 三角校验中…"):
        results = router.tag(texts)
    stats = router.last_stats

    res = pd.DataFrame(
        [
            {
                "文本": r.text,
                "情感": r.sentiment,
                "方面": "、".join(r.aspects) or "—",
                "命中轨道": r.track,
                "置信": round(r.confidence, 2),
                "已校验": "✅" if r.verified else "⚠️",
                "升档": r.escalations,
            }
            for r in results
        ]
    )

    # 顶部关键指标
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("打标条数", stats["n"])
    m2.metric("校验通过", f"{stats['n_verified']}/{stats['n']}")
    m3.metric("升档次数", stats["n_escalated"])
    m4.metric("可用轨道", len(stats["ladder"]))

    st.subheader("逐条打标结果")
    st.dataframe(res, width="stretch")

    col_left, col_right = st.columns(2)

    # 情感分布图
    with col_left:
        st.subheader("情感分布")
        sent_counts = res["情感"].value_counts()
        st.bar_chart(sent_counts)
        st.caption(
            "正面 {} · 中性 {} · 负面 {}".format(
                int(sent_counts.get("正面", 0)),
                int(sent_counts.get("中性", 0)),
                int(sent_counts.get("负面", 0)),
            )
        )

    # 轨道分配 + 校验统计
    with col_right:
        st.subheader("轨道分配（算力按难度分配）")
        if stats["route_counts"]:
            st.bar_chart(pd.Series(stats["route_counts"]))
        st.caption("可用轨道阶梯：" + " → ".join(stats["ladder"]))
        verify_rate = stats["n_verified"] / stats["n"] * 100 if stats["n"] else 0
        st.caption(
            f"校验通过率 {verify_rate:.0f}%　|　共升档 {stats['n_escalated']} 次"
            "（校验不过自动沿成本阶梯升档重判）"
        )
elif not texts:
    st.info("在上方粘贴评论或上传 CSV，然后点击「智能路由打标」。")
