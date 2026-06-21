"""一键自检：逐项探测项目用到的所有服务 / 模型是否可用，并打印健康报告。

覆盖：DeepSeek API、RAG 检索、蒸馏本地模型、云端 LoRA 端点、本地 LoRA 大模型、
智能路由、Streamlit 看板。可用 --skip-api 跳过联网实测，只看离线能力。

用法：
    uv run python scripts/healthcheck.py
    uv run python scripts/healthcheck.py --skip-api
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

OK, WARN, SKIP = "OK", "WARN", "SKIP"

# 这些是核心服务：状态非 OK 时整体判为不健康（退出码 1）。
CRITICAL = {"智能路由 Router"}

Check = tuple[str, str, str]  # (status, label, detail)


def _check_deepseek(skip: bool) -> Check:
    if skip:
        return (SKIP, "DeepSeek API", "已跳过（--skip-api）")
    try:
        from src import llm_client

        r = llm_client.chat_json("只输出JSON", '返回 {"ping":"pong"}', max_retries=1)
        return (OK, "DeepSeek API", f"实测返回 {r}")
    except Exception as e:  # noqa: BLE001 - 自检不抛错，记录原因
        return (WARN, "DeepSeek API", f"不可用：{type(e).__name__}: {str(e)[:80]}")


def _check_rag() -> Check:
    try:
        from src.rag.embeddings import HashingEmbedding
        from src.rag.retriever import HybridRetriever

        retr = HybridRetriever.from_documents(
            ["歪了：抽卡没出UP角色", "保底：累计必出高星"], embedding_fn=HashingEmbedding(64)
        )
        hit = retr.retrieve("歪了", top_k=1)
        return (OK, "RAG 检索（离线）", f"命中『{hit[0].document[:14]}…』")
    except Exception as e:  # noqa: BLE001
        return (WARN, "RAG 检索（离线）", f"不可用：{e}")


def _check_distilled() -> Check:
    from src import sentiment_train

    if not sentiment_train.DEFAULT_MODEL_PATH.exists():
        return (WARN, "蒸馏本地模型", "模型文件不存在，先跑 scripts/train_sentiment.py")
    try:
        model = sentiment_train.load_model()
        pred = sentiment_train.predict(model, ["抽卡又歪了真烂", "剧情太感人了"])
        return (OK, "蒸馏本地模型", f"预测 {pred}")
    except Exception as e:  # noqa: BLE001
        return (WARN, "蒸馏本地模型", f"加载失败：{e}")


def _check_cloud_lora() -> Check:
    from src import config

    if not config.LORA_SERVER_BASE_URL:
        return (SKIP, "云端 LoRA 端点", "未配置 LORA_SERVER_BASE_URL（场景 B 未启用）")
    try:
        from src.agents.tracks import CloudLoRATrack

        track = CloudLoRATrack()
        if not track.is_available():
            return (WARN, "云端 LoRA 端点", "已配置但客户端构造失败（缺 openai SDK？）")
        out = track.classify(["抽卡又歪了"])
        return (OK, "云端 LoRA 端点", f"实测返回 {out[0].sentiment}")
    except Exception as e:  # noqa: BLE001
        return (WARN, "云端 LoRA 端点", f"不可达：{type(e).__name__}: {str(e)[:80]}")


def _check_local_lora() -> Check:
    from src.sentiment_train import LocalLLMClassifier

    clf = LocalLLMClassifier()
    if clf.is_ready():
        return (OK, "本地 LoRA 大模型", "依赖与适配器就绪")
    reasons = []
    if not LocalLLMClassifier.deps_available():
        reasons.append("缺 torch/transformers/peft")
    if not clf.adapter_path.exists():
        reasons.append("无训练适配器")
    return (SKIP, "本地 LoRA 大模型", "未激活：" + "，".join(reasons))


def _check_router() -> Check:
    try:
        from src import llm_client
        from src.agents import RouterAgent

        client = None
        try:
            client = llm_client.get_client()
        except Exception:  # noqa: BLE001 - 无 API 时路由自动退化，不算失败
            pass
        router = RouterAgent.from_environment(client=client)
        router.tag(["剧情真好", "这次又歪了保底白给"])
        stats = router.last_stats
        return (OK, "智能路由 Router", f"阶梯 {stats['ladder']}，分配 {stats['route_counts']}")
    except Exception as e:  # noqa: BLE001
        return (WARN, "智能路由 Router", f"不可用：{e}")


def _check_dashboard() -> Check:
    avail = importlib.util.find_spec("streamlit") is not None
    detail = "已安装" if avail else "未安装（uv sync --extra dashboard）"
    return (OK if avail else SKIP, "Streamlit 看板", detail)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # 避免 Windows 默认编码渲染异常
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="项目服务一键自检")
    parser.add_argument("--skip-api", action="store_true", help="跳过 DeepSeek 联网实测")
    args = parser.parse_args()

    checks: list[Check] = [
        _check_deepseek(args.skip_api),
        _check_rag(),
        _check_distilled(),
        _check_cloud_lora(),
        _check_local_lora(),
        _check_router(),
        _check_dashboard(),
    ]

    print("服务自检结果：")
    for status, label, detail in checks:
        print(f"  [{status:^4}] {label}  ——  {detail}")

    n_ok = sum(1 for s, _, _ in checks if s == OK)
    print(f"\n汇总：{n_ok}/{len(checks)} 项可用")

    bad = [label for status, label, _ in checks if label in CRITICAL and status != OK]
    if bad:
        print("核心服务不可用：" + "，".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
