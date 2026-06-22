"""在留出评估集上评估打标模型并做错例分析（第 3 步：评估-迭代闭环）。

预测器可选：
    --predictor distilled   蒸馏 TF-IDF 模型（默认，无需 GPU，可立即跑通闭环）
    --predictor lora        本地微调大模型 LocalLLMClassifier（需 GPU + 已训练适配器）
    --predictor keyword     关键词/词典基线（纯离线对照）

输出准确率 / 宏 F1 / 各类指标，并把错例（尤其疑似反讽）单独列出，
据此“去原数据集捞同类样本补充 → 增量微调”，进入下一轮迭代。

用法：
    uv run python scripts/eval_lora.py
    uv run python scripts/eval_lora.py --predictor lora --eval-set outputs/finetune/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src import config
from src.finetune.evaluate import (
    EvalReport,
    Predictor,
    build_markdown_report,
    evaluate,
    load_eval_set,
    write_predictions_csv,
)


def _build_predictor(name: str) -> Predictor:
    if name == "distilled":
        from src import sentiment_train

        model = sentiment_train.load_model()
        return lambda texts: sentiment_train.predict(model, texts)
    if name == "lora":
        from src.sentiment_train import LocalLLMClassifier

        clf = LocalLLMClassifier()
        if not clf.is_ready():
            raise RuntimeError(
                "LoRA 模型未就绪（缺依赖或未训练适配器）。先在 GPU 上跑 train_lora.sh。"
            )
        return clf.predict
    if name == "keyword":
        from src.agents.base import lexicon_polarity

        return lambda texts: [lexicon_polarity(t)[0] for t in texts]
    raise ValueError(f"未知预测器：{name}")


def main() -> int:
    # 让 src.sentiment_train 里的加载/推理进度日志显示出来（否则 lora 模式长时间静默像卡死）
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    parser = argparse.ArgumentParser(description="留出评估 + 错例分析（第 3 步）")
    parser.add_argument(
        "--eval-set", default=str(config.OUTPUT_DIR / "finetune" / "eval.jsonl")
    )
    parser.add_argument(
        "--predictor", choices=["distilled", "lora", "keyword"], default="distilled"
    )
    parser.add_argument("--show-errors", type=int, default=15, help="打印多少条错例")
    parser.add_argument("--report", default=None, help="把原始结果写到 JSON 文件")
    parser.add_argument(
        "--report-md",
        nargs="?",
        const=str(config.OUTPUT_DIR / "finetune" / "eval_report.md"),
        default=None,
        help="生成详细 Markdown 报告（不带值时默认写到 outputs/finetune/eval_report.md），"
        "并在同目录写逐条预测 CSV",
    )
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="额外评估 keyword/distilled 基线并在报告中对比（最能体现微调增益）",
    )
    parser.add_argument("--max-error-samples", type=int, default=30, help="报告里列多少条错例")
    args = parser.parse_args()

    if not Path(args.eval_set).exists():
        print(
            f"评估集不存在：{args.eval_set}\n先跑 scripts/build_finetune_dataset.py 生成。",
            file=sys.stderr,
        )
        return 1

    texts, gold = load_eval_set(args.eval_set)
    print(f"已加载留出集 {len(texts)} 条（{args.eval_set}）", flush=True)
    try:
        predictor = _build_predictor(args.predictor)
    except Exception as e:  # noqa: BLE001 - 预测器不可用时明确报错
        print(f"预测器 {args.predictor} 不可用：{e}", file=sys.stderr)
        return 1

    if args.predictor == "lora":
        print(
            "开始评估：lora 模式会先加载 7B 大模型（可能 1-2 分钟），"
            "随后按条打印推理进度。",
            flush=True,
        )
    report = evaluate(predictor, texts, gold)
    print(f"[{args.predictor}] {report.summary()}")
    print(f"混淆矩阵（行=金标，列=预测，标签序 {report.labels}）：")
    for row in report.confusion:
        print("  ", row)

    irony_errors = [e for e in report.errors if e["irony"]]
    print(
        f"\n错例（共 {len(report.errors)}，疑似反讽 {len(irony_errors)}），"
        f"前 {args.show_errors} 条："
    )
    for e in report.errors[: args.show_errors]:
        flag = "［反讽?］" if e["irony"] else ""
        print(f"  金标={e['gold']} 预测={e['pred']} {flag} {e['text'][:40]}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始结果 JSON 已写入 {args.report}")

    if args.report_md:
        baselines: dict[str, EvalReport] = {}
        if args.with_baselines:
            for name in ("distilled", "keyword"):
                if name == args.predictor:
                    continue
                try:
                    print(f"评估基线 {name} 用于对比…", flush=True)
                    baselines[name] = evaluate(_build_predictor(name), texts, gold)
                except Exception as e:  # noqa: BLE001 - 基线不可用就跳过，不阻塞主报告
                    print(f"  基线 {name} 不可用，跳过：{e}", file=sys.stderr)

        md = build_markdown_report(
            report,
            predictor_name=args.predictor,
            eval_set=args.eval_set,
            baselines=baselines or None,
            max_error_samples=args.max_error_samples,
        )
        md_path = Path(args.report_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        csv_path = md_path.with_suffix(".predictions.csv")
        write_predictions_csv(csv_path, texts, gold, report)
        print(f"[报告] 详细 Markdown 报告已写入 {md_path}")
        print(f"[报告] 逐条预测 CSV 已写入 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
