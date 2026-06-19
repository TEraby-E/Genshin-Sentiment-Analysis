"""项目主入口：运行完整分析流程并打印结论。

用法：
    uv run genshin-analyze
    # 或
    uv run python -m src.main
"""

from __future__ import annotations

import logging

from . import (
    ab_test,
    analysis,
    aspect_sentiment,
    causal_inference,
    config,
    data_loader,
    lifecycle,
    visualize,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    print("=" * 64)
    print("原神 B 站玩家生态与舆情分析")
    print("=" * 64)

    # ---- 加载数据 ----
    videos, comments, posts = data_loader.load_all()
    print(f"视频 {len(videos):,} 条 | 评论 {len(comments):,} 条 | 帖子 {len(posts):,} 条\n")

    # ---- 1. 数据质量 ----
    dq = analysis.data_quality_report(videos)
    print("【数据质量审查】")
    print(f"  混合内容(噪声)占比: {dq['noise_pct']}%（需用主题聚类分离核心内容）")
    print(f"  播放量中位 {dq['median_view']:,} / 最高 {dq['max_view']:,}（长尾分布）")
    print(f"  头部1%视频贡献了 {dq['top1pct_view_share']}% 的总播放\n")

    # ---- 2. 玩家生态 ----
    eco = analysis.ecosystem_by_topic(videos)
    print("【玩家生态：主题热度 Top5】")
    print(eco.head(5).to_string())
    print()

    # ---- 3. 舆情监控 ----
    monthly = analysis.sentiment_trend(comments, posts)
    alert = analysis.alert_threshold(monthly)
    print("【舆情监控】")
    print(f"  负面占比基线: {alert['baseline']}%")
    print(f"  预警触发线({config.ALERT_MULTIPLIER}×基线): {alert['threshold']}%")
    print(f"  峰值: {alert['peak_month']} 达 {alert['peak_rate']}%")
    print(f"  超过预警线的月份: {alert['breach_months']}")
    sig = alert["significance"]
    sig_desc = "显著" if sig["significant"] else "不显著"
    print(
        f"  峰值({sig['peak_month']}, {sig['rate_a']}%) vs 基线({sig['baseline_month']}, "
        f"{sig['rate_b']}%)：z={sig['z']}, p={sig['p_value']} → {sig_desc}"
    )
    print(f"  滚动z-score异常月份(适应基线漂移): {alert['zscore_anomaly_months']}\n")

    # ---- 3b. 方面级情感分析（若评论数据含原始文本） ----
    if "Comment_Content" in comments.columns:
        comments_with_neg = comments.copy()
        comments_with_neg["is_neg"] = comments_with_neg["cluster_name"].isin(
            config.NEGATIVE_CLUSTERS
        )
        coverage = aspect_sentiment.agreement_with_cluster(comments_with_neg)
        breakdown = aspect_sentiment.aspect_breakdown(comments_with_neg)
        print("【方面级情感分析（关键词基线，验证细粒度可深化方向）】")
        print(f"  关键词方面标签覆盖率: {coverage['coverage_rate']}% "
              f"({coverage['tagged_comments']}/{coverage['total_comments']})")
        print(breakdown.to_string())
        print()

    # ---- 4. 爆款预测 ----
    hit = analysis.hit_prediction(videos)
    print("【爆款预测模型】")
    print(f"  AUC: {hit['auc']}（仅元数据，效果有限 → 爆款由内容质量主导）")
    print(f"  特征重要性: {hit['feature_importance']}\n")

    # ---- 5. 创作者与主题生命周期 ----
    creators = lifecycle.creator_lifecycle(videos)
    stage_summary = lifecycle.creator_stage_summary(creators)
    # 主题生命周期排除噪声主题簇，否则混合内容会因为体量大而排到 Top（详见数据质量审查）
    core_videos = videos[videos["Topic_Cluster"] != config.NOISE_TOPIC_CLUSTER]
    topics_lc = lifecycle.topic_lifecycle(core_videos)
    print("【创作者生命周期分群】")
    print(stage_summary.to_string())
    print("\n【内容主题生命周期 Top5（按近期产出量）】")
    print(topics_lc.head(5)[["recent_count", "growth_rate", "stage"]].to_string())
    print()

    # ---- 6. 跨界联动因果效应 ----
    if "Collaboration_Flag" in posts.columns:
        try:
            collab = causal_inference.matched_control_effect(posts, metric="Like_Count")
            print("【跨界联动效果评估（同月匹配 + 置换检验）】")
            print(
                f"  处理组 n={collab['n_treated']}，相对同月对照组点赞数提升 "
                f"{collab['relative_lift_pct']}%，p={collab['p_value']} → "
                f"{'显著' if collab['significant'] else '不显著'}"
            )
            print(f"  {collab['caveat']}\n")
        except ValueError as e:
            print(f"【跨界联动效果评估】跳过：{e}\n")

    # ---- 7. A/B 实验评估框架演示 ----
    ab_demo = ab_test.simulate_campaign_ab_demo()
    print("【A/B 实验评估框架演示（模拟数据，验证方法本身）】")
    print(
        f"  所需样本量/组: {ab_demo['sizing']['required_n_per_group']:,}（基线 "
        f"{ab_demo['sizing']['baseline_rate']:.0%}, MDE {ab_demo['sizing']['mde']:.0%}）"
    )
    print(
        f"  实验结果：对照 {ab_demo['control_rate']:.2%} vs 实验 {ab_demo['treatment_rate']:.2%}，"
        f"绝对提升 {ab_demo['absolute_lift']:.2%}, 95% CI {ab_demo['ci_95']}, "
        f"p={ab_demo['p_value']} → {'显著' if ab_demo['significant'] else '不显著'}\n"
    )

    # ---- 8. 可视化 ----
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = visualize.make_dashboard(videos, comments, posts, monthly)
    print(f"六图分析面板已保存至: {path}")


if __name__ == "__main__":
    main()
