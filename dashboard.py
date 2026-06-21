"""内容生态创作者管理 & 作品分析工具看板（实习职责1落地）。

一个简易、可复用的内部工具看板，把已有分析能力封装成业务可直接上手的三个工具：
1. 作品打标       —— 关键词基线 + AI 语义打标（情感/方面），支持粘贴或上传 CSV；
2. 作者分类与增长  —— UP 主生命周期分群 + 主题增长/衰退标签，识别唤回/倾斜目标；
3. 爆点预测与总结  —— 单视频爆款概率打分器 + 爆款共性 AI 总结，指导选题。

运行：
    uv sync --extra dashboard            # 仅看板（离线工具可用）
    uv sync --extra dashboard --extra llm  # 叠加 AI 打标/总结
    uv run streamlit run dashboard.py

无真实数据（data/ 下没有 Kaggle CSV）时自动切换到合成演示数据，开箱即跑。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import analysis, aspect_sentiment, config, lifecycle, monitor, sample_data

st.set_page_config(page_title="原神内容生态分析工具看板", layout="wide", page_icon="🎮")


# ---- 数据源：优先真实数据，缺失则用合成演示数据 ----
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    try:
        from src import data_loader

        videos, comments, posts = data_loader.load_all()
        return videos, comments, posts, True
    except Exception:  # noqa: BLE001 - 任何加载失败都回退到演示数据
        videos, comments, posts = sample_data.load_demo()
        return videos, comments, posts, False


@st.cache_resource(show_spinner="训练爆款预测模型…")
def get_hit_model(_videos: pd.DataFrame) -> dict:
    return analysis.train_hit_model(_videos)


@st.cache_resource(show_spinner="构建 RAG 梗&设定词典…")
def get_lore_retriever():
    """构建混合检索器供路由的「检索」一角使用；无数据/构建失败则返回 None（路由照常退化）。"""
    try:
        import asyncio

        from src.rag.ingestion import build_lore_dictionary

        return asyncio.run(build_lore_dictionary(max_posts=120, max_comments=400))
    except Exception:  # noqa: BLE001 - 缺数据等情况下不阻塞路由
        return None


def llm_available() -> bool:
    try:
        from src import llm_client

        llm_client.get_api_key()
        import openai  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


videos, comments, posts, is_real = load_data()

# ---- 侧边栏 ----
st.sidebar.title("🎮 内容生态分析工具")
st.sidebar.caption("内部工具看板 · 为内容经营提供工具指导")
st.sidebar.metric("视频", f"{len(videos):,}")
st.sidebar.metric("评论", f"{len(comments):,}")
st.sidebar.metric("UP 主", f"{videos['Author'].nunique():,}")
st.sidebar.divider()
if is_real:
    st.sidebar.success("数据源：真实数据 (data/)")
else:
    st.sidebar.info("数据源：合成演示数据\n（把 Kaggle CSV 放入 data/ 即用真实数据）")
_has_llm = llm_available()
st.sidebar.write("AI 能力：" + ("✅ 已配置" if _has_llm else "⚠️ 未配置（仅离线工具可用）"))
if not _has_llm:
    st.sidebar.caption("配置 .env 的 DEEPSEEK_API_KEY 并 `uv sync --extra llm` 后启用 AI 打标/总结")

st.title("内容生态创作者管理 & 作品分析工具")

tab_tag, tab_creator, tab_hit, tab_comp, tab_rank, tab_opinion = st.tabs(
    [
        "🏷️ 作品打标",
        "👤 作者分类与增长",
        "🚀 内容爆点：预测与总结",
        "🔭 竞品动态监测",
        "🏆 榜单数据监控",
        "☁️ 舆情词云 AI 总结",
    ]
)


@st.cache_data(ttl=1800, show_spinner="抓取 B 站数据中…")
def cached_ranking() -> tuple[pd.DataFrame, bool]:
    return monitor.fetch_ranking(monitor.BilibiliClient(min_interval=1.0))


@st.cache_data(ttl=1800, show_spinner="搜索竞品动态中…")
def cached_competitors(keywords: tuple[str, ...]) -> tuple[pd.DataFrame, bool]:
    client = monitor.BilibiliClient(min_interval=1.0)
    return monitor.track_competitors(list(keywords), client=client)


def _source_caption(live: bool) -> None:
    st.caption("数据源：" + ("🟢 实时抓取" if live else "🟡 演示数据（抓取被风控/无网，已降级）"))


# ========== 工具1：作品打标 ==========
with tab_tag:
    st.subheader("作品 / 评论打标工具")
    st.caption("给非结构化文本打上情感极性与内容方面标签，沉淀为可分析的结构化数据。")

    mode = st.radio(
        "打标方式",
        [
            "关键词基线（离线·秒出）",
            "AI 语义打标（DeepSeek）",
            "本地模型（蒸馏·免费秒级）",
            "本地微调大模型 (Local LoRA LLM)",
            "🧭 智能路由（自动分配·校验·Router Agent）",
        ],
        horizontal=True,
    )
    default_text = "\n".join(sample_data._COMMENT_TEXTS[:5])
    src_choice = st.radio("输入来源", ["手动粘贴", "上传 CSV"], horizontal=True)

    if src_choice == "手动粘贴":
        raw = st.text_area("每行一条文本", value=default_text, height=160)
        texts = [t for t in raw.splitlines() if t.strip()]
    else:
        up = st.file_uploader("上传 CSV", type="csv")
        col = None
        texts = []
        if up is not None:
            df_up = pd.read_csv(up)
            col = st.selectbox("选择文本列", df_up.columns)
            texts = df_up[col].dropna().astype(str).tolist()
            st.caption(f"读取 {len(texts)} 条，AI 打标默认只处理前 N 条以控成本")

    if mode.startswith("关键词"):
        if st.button("开始打标", type="primary") and texts:
            rows = [
                {"文本": t, "方面标签": "、".join(aspect_sentiment.tag_aspects(t)) or "—"}
                for t in texts
            ]
            res = pd.DataFrame(rows)
            st.dataframe(res, width="stretch")
            cov = sum(r["方面标签"] != "—" for r in rows) / len(rows) * 100
            st.metric(
                "关键词覆盖率", f"{cov:.0f}%",
                help="覆盖率低说明口语化/表情多，正是 AI 语义打标的价值所在",
            )
    elif mode.startswith("本地模型"):
        from src import sentiment_train

        st.caption("用 LLM 标注小样本蒸馏出的轻量分类器，离线、免 API、毫秒级——适合全量打标。")
        if not sentiment_train.DEFAULT_MODEL_PATH.exists():
            st.warning(
                "尚未训练本地模型。先运行 `uv run python scripts/train_sentiment.py --sample 600` "
                "生成 outputs/sentiment_clf.joblib。"
            )
        elif st.button("用本地模型打标", type="primary") and texts:
            model = sentiment_train.load_model()
            preds = sentiment_train.predict(model, texts)
            res = pd.DataFrame({"文本": texts, "情感（本地模型）": preds})
            st.dataframe(res, width="stretch")
            st.bar_chart(pd.Series(preds).value_counts())
    elif mode.startswith("本地微调"):
        from src import sentiment_train

        st.caption(
            "用 LLaMA-Factory QLoRA 在 eGPU 上微调的本地大模型（Qwen2.5 + LoRA），"
            "比蒸馏分类器更懂语义与社区黑话，离线、免 API。"
        )
        clf = sentiment_train.LocalLLMClassifier()
        if not sentiment_train.LocalLLMClassifier.deps_available():
            st.warning(
                "缺少推理依赖。请先 `uv sync --extra finetune`（需 transformers/peft/torch）。"
            )
        elif not clf.adapter_path.exists():
            st.warning(
                "尚未训练 LoRA 适配器。流程：`uv run python -m src.finetune.dataset_formatter` "
                "生成数据集 → `bash src/finetune/train_lora.sh` 微调，产物默认在 "
                f"`{clf.adapter_path}`。"
            )
        elif st.button("用本地微调大模型打标", type="primary") and texts:
            with st.spinner("加载本地大模型并推理中（首次加载较慢）…"):
                preds = clf.predict(texts)
            res = pd.DataFrame({"文本": texts, "情感（LoRA 大模型）": preds})
            st.dataframe(res, width="stretch")
            st.bar_chart(pd.Series(preds).value_counts())
    elif mode.startswith("🧭"):
        from src import agents, llm_client

        st.caption(
            "路由 Agent 按评论难度自动分配到最省的可行轨道（关键词 / 蒸馏 / LoRA / RAG-DeepSeek）："
            "容易的走便宜轨道，难句（黑话 / 反讽）直接起步于语义轨道，并经「检索 → 推理 → 校验」"
            "三角复核，校验不过则自动升档重判。不可用的轨道（无 API / 无 GPU / 无模型）自动跳过。"
        )
        if st.button("智能路由打标", type="primary") and texts:
            client = llm_client.get_client() if _has_llm else None
            retriever = get_lore_retriever()
            router = agents.RouterAgent.from_environment(client=client, retriever=retriever)
            with st.spinner("路由分配 + 三角校验中…"):
                results = router.tag(texts)
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
            st.dataframe(res, width="stretch")
            stats = router.last_stats
            c1, c2, c3 = st.columns(3)
            c1.metric("可用轨道阶梯", " → ".join(stats["ladder"]))
            c2.metric("校验通过", f"{stats['n_verified']}/{stats['n']}")
            c3.metric("升档次数", stats["n_escalated"])
            if stats["route_counts"]:
                st.caption("各轨道最终处理量（体现算力分配）")
                st.bar_chart(pd.Series(stats["route_counts"]))
    else:
        n = st.slider("AI 处理条数（控成本）", 1, 100, min(20, len(texts) or 20))
        if not _has_llm:
            st.warning("未配置 AI 能力，无法使用语义打标。请配置 DEEPSEEK_API_KEY。")
        elif st.button("开始 AI 打标", type="primary") and texts:
            from src import text_pipeline

            with st.spinner("调用 DeepSeek 打标中…"):
                df_in = pd.DataFrame({"Comment_Content": texts})
                analyzed = text_pipeline.analyze_comments(df_in, sample=n)
            show = analyzed[["clean_text", "llm_sentiment", "llm_aspects", "llm_reason"]].rename(
                columns={
                    "clean_text": "清洗后文本", "llm_sentiment": "情感",
                    "llm_aspects": "方面", "llm_reason": "依据",
                }
            )
            st.dataframe(show, width="stretch")
            c1, c2 = st.columns(2)
            c1.bar_chart(analyzed["llm_sentiment"].value_counts())
            c2.dataframe(text_pipeline.aspect_sentiment_summary(analyzed), width="stretch")


# ========== 工具2：作者分类与增长 ==========
with tab_creator:
    st.subheader("UP 主生命周期分群")
    st.caption("按发布频次与活跃度把创作者分为 单发/成长期/稳定期/沉寂期，指导合作资源投向。")

    creators = lifecycle.creator_lifecycle(videos)
    stage_summary = lifecycle.creator_stage_summary(creators)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**各阶段创作者数量**")
        st.bar_chart(stage_summary["creator_count"])
    with c2:
        st.markdown("**各阶段单视频中位播放**")
        st.bar_chart(stage_summary["median_view_per_video"])
    st.dataframe(stage_summary, width="stretch")

    # 唤回目标：沉寂期里历史表现好的创作者
    st.markdown("#### 🎯 潜在唤回目标（沉寂期 · 历史中位播放高）")
    st.caption("这类创作者已沉寂但历史表现优于稳定期均值，是再合作/唤回的高价值对象，而非把资源都押在当前活跃者。")
    dormant = creators[creators["stage"] == "沉寂期"].sort_values("median_view", ascending=False)
    st.dataframe(
        dormant[["n_videos", "median_view", "median_like", "days_since_last"]].head(15),
        width="stretch",
    )

    st.divider()
    st.subheader("内容主题增长分析")
    st.caption("按月度产出量变化率给主题打 增长/平稳/衰退 标签，指导内容资源倾斜。")
    core = videos[videos["Topic_Cluster"] != config.NOISE_TOPIC_CLUSTER]
    topics_lc = lifecycle.topic_lifecycle(core)
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.bar_chart(topics_lc["growth_rate"])
    with cc2:
        st.dataframe(
            topics_lc[["recent_count", "prior_count", "growth_rate", "stage"]],
            width="stretch",
        )


# ========== 工具3：内容爆点预测与总结 ==========
with tab_hit:
    st.subheader("内容爆点：预测 + 总结")
    st.caption("预测回答「这条会不会爆」，总结回答「为什么爆、怎么复制」。")

    trained = get_hit_model(videos)
    m1, m2 = st.columns([1, 2])
    m1.metric("模型 AUC", trained["auc"], help="仅用元数据特征；AUC 有限本身说明爆款由内容质量主导")
    with m2:
        imp = pd.Series(trained["feature_importance"]).sort_values(ascending=True)
        st.markdown("**特征重要性**")
        st.bar_chart(imp)

    st.markdown("#### 🔮 单视频爆款概率打分器")
    st.caption(
        f"爆款定义：播放量 ≥ 全站 {int(config.HIT_QUANTILE * 100)} 分位。输入视频参数试算。"
    )
    enc = trained["length_encoder"]
    f1, f2, f3 = st.columns(3)
    topic_names = sorted(videos["topic"].dropna().unique())
    topic_pick = f1.selectbox("主题", topic_names)
    topic_cluster = int(videos.loc[videos["topic"] == topic_pick, "Topic_Cluster"].mode().iloc[0])
    length_type = f2.selectbox("视频类型", list(enc.classes_))
    duration = f3.slider("时长（秒）", 30, 1500, 300)
    g1, g2, g3 = st.columns(3)
    hour = g1.slider("发布小时", 0, 23, 20)
    dow = g2.slider("星期几（0=周一）", 0, 6, 5)
    title_len = g3.slider("标题字数", 5, 40, 18)

    if st.button("预测爆款概率", type="primary"):
        x = pd.DataFrame(
            [{
                "TimeInSeconds": duration,
                "Topic_Cluster": topic_cluster,
                "hour": hour,
                "dayofweek": dow,
                "title_len": title_len,
                "len_type_enc": int(enc.transform([length_type])[0]),
            }]
        )[trained["features"]]
        prob = float(trained["model"].predict_proba(x)[0, 1])
        st.metric("爆款概率", f"{prob:.1%}")
        st.progress(min(prob, 1.0))
        st.caption("提示：特征重要性显示发布技巧贡献有限，提升内容质量才是关键。")

    st.divider()
    st.markdown("#### 📝 爆款共性 AI 总结")
    st.caption("抽取实际爆款视频标题，让 AI 归纳可复制的选题套路。")
    top_n = st.slider("分析爆款标题数", 20, 200, 60)
    if not _has_llm:
        st.warning("未配置 AI 能力，无法生成爆款总结。")
    elif st.button("生成爆款选题总结"):
        from src import text_pipeline

        floor = videos["Amount_View"].quantile(config.HIT_QUANTILE)
        hit_titles = (
            videos[videos["Amount_View"] >= floor]
            .sort_values("Amount_View", ascending=False)["Video_Title"]
            .astype(str)
            .head(top_n)
            .tolist()
        )
        with st.spinner("AI 归纳爆款选题套路中…"):
            summary = text_pipeline.summarize_hits(hit_titles)
        for p in summary.get("patterns", []):
            st.markdown(f"**{p.get('theme', '')}** — {p.get('summary', '')}")
            for ex in p.get("examples", []):
                st.markdown(f"  - {ex}")
        if summary.get("keywords"):
            st.markdown("**高频关键词：** " + " · ".join(summary["keywords"]))
        if summary.get("advice"):
            st.markdown("**选题建议：**")
            for a in summary["advice"]:
                st.markdown(f"- {a}")


# ========== 工具4：竞品动态监测 ==========
with tab_comp:
    st.subheader("竞品动态监测")
    st.caption("对一组竞品/对标二游在 B 站搜索，聚合各自的内容产出量与热度，辅助竞品对比决策。")
    kw_text = st.text_input(
        "竞品关键词（空格分隔）", value=" ".join(monitor.DEFAULT_COMPETITORS)
    )
    if st.button("抓取竞品动态", type="primary"):
        keywords = tuple(k for k in kw_text.split() if k.strip())
        comp_df, live = cached_competitors(keywords)
        _source_caption(live)
        cc1, cc2 = st.columns([1, 1])
        cc1.bar_chart(comp_df.set_index("竞品")["总播放"])
        cc2.bar_chart(comp_df.set_index("竞品")["中位播放"])
        st.dataframe(comp_df, width="stretch")


# ========== 工具5：榜单数据监控 ==========
with tab_rank:
    st.subheader("B 站全站排行榜监控")
    st.caption("抓取榜单 → 落地带时间戳快照 → 与上次对比，标出新上榜与排名变化。")
    if st.button("抓取最新榜单", type="primary"):
        rk, live = cached_ranking()
        _source_caption(live)

        prev = monitor.list_snapshots("ranking")
        if prev:
            old = pd.read_csv(prev[-1])
            diff = monitor.diff_rankings(old, rk)
            m1, m2 = st.columns(2)
            m1.metric("新上榜", len(diff["newcomers"]))
            m2.metric("跌出榜单", diff["dropped_count"])
            if diff["movements"]:
                st.markdown("**排名变化 Top5**")
                for m in diff["movements"][:5]:
                    arrow = "🔺" if m["change"] > 0 else "🔻"
                    st.markdown(
                        f"{arrow} {abs(m['change'])} 名｜{m['title'][:24]}"
                        f"（{m['from']}→{m['to']}）"
                    )
        else:
            st.info("首次抓取，暂无历史快照可对比。再抓一次即可看到榜单变化。")

        monitor.save_snapshot(rk, "ranking")
        st.dataframe(rk[["rank", "title", "up", "view", "like"]], width="stretch")


def _render_ai_keyword_cloud(keywords: list[str]) -> None:
    """渲染真正的词云图（AI 关键词加权）；无 wordcloud/字体时降级为 HTML 字号云。"""
    if not keywords:
        return
    try:
        from src import wordcloud_gen

        img = wordcloud_gen.render_wordcloud(
            {k: (len(keywords) - i) for i, k in enumerate(keywords)}, colormap="Reds"
        )
        st.image(img, caption="AI 提炼关键词词云", width="stretch")
    except Exception as e:  # noqa: BLE001 - 缺 wordcloud/jieba/字体时降级
        st.caption(f"（词云图不可用，降级为关键词列表：{e}）")
        sizes = [28 - min(i, 12) for i in range(len(keywords))]
        cloud = "  ".join(
            f"<span style='font-size:{s}px'>{k}</span>" for k, s in zip(keywords, sizes)
        )
        st.markdown(f"<div style='line-height:2.2'>{cloud}</div>", unsafe_allow_html=True)


# ========== 工具6：舆情词云 AI 总结 ==========
with tab_opinion:
    st.subheader("舆情词云 AI 总结")
    st.caption(
        "先用 AI 逐条打标（情感 + 方面），再按 LLM 情感分组出词云，"
        "并用 AI 提炼的语义关键词加权——比纯词频更突出真正的舆情议题。"
    )
    op_src = st.radio("评论来源", ["从评论数据抽样", "手动粘贴"], horizontal=True)
    if op_src == "从评论数据抽样":
        op_n = st.slider("AI 打标条数（控成本）", 20, 300, 60)
        op_texts = comments["Comment_Content"].dropna().astype(str).head(op_n).tolist()
    else:
        default_op = "\n".join(sample_data._COMMENT_TEXTS)
        raw = st.text_area("每行一条评论", value=default_op, height=150)
        op_texts = [t for t in raw.splitlines() if t.strip()]

    if not _has_llm:
        st.warning("未配置 AI 能力，无法生成舆情总结。配置 DEEPSEEK_API_KEY 后可用。")
    elif st.button("AI 打标 + 生成词云总结", type="primary") and op_texts:
        from src import text_pipeline

        with st.spinner("AI 逐条打标中…"):
            analyzed = text_pipeline.analyze_comments(
                pd.DataFrame({"Comment_Content": op_texts})
            )
        neg = analyzed[analyzed["llm_sentiment"] == "负面"]
        pos = analyzed[analyzed["llm_sentiment"] != "负面"]
        m1, m2 = st.columns(2)
        m1.metric("AI 判定负面", len(neg))
        m2.metric("非负面", len(pos))

        # 以负面评论为舆情焦点做总结（没有负面则退而用全部）
        focus = neg if len(neg) else analyzed
        with st.spinner("AI 总结舆情焦点中…"):
            summary = text_pipeline.summarize_opinions(focus["clean_text"].tolist())

        if summary.get("overall"):
            st.info("**总体判断：** " + summary["overall"])

        left, right = st.columns([1, 1])
        with left:
            _render_ai_keyword_cloud(summary.get("keywords", []))
        with right:
            for t in summary.get("themes", []):
                share = t.get("share")
                pct = f"（约 {share:.0%}）" if isinstance(share, (int, float)) else ""
                st.markdown(f"**{t.get('topic', '')}**{pct} — {t.get('summary', '')}")
                for q in t.get("quotes", []):
                    st.markdown(f"  > {q}")
        if summary.get("actions"):
            st.markdown("**运营建议：**")
            for a in summary["actions"]:
                st.markdown(f"- {a}")
