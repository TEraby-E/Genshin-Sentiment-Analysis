"""舆情打标"训练"：LLM 标注小样本 → 蒸馏出轻量本地分类器。

动机：直接用 LLM 给全量 40.8 万条评论打情感标签，API 成本与耗时都不可接受。
做法是知识蒸馏（knowledge distillation）：
1. 用主流 AI 模型（DeepSeek）给一个**小样本**打高质量情感标签，作为"老师"标注；
2. 在这批 (文本 → LLM 标签) 上训练一个**轻量学生模型**（字符级 TF-IDF + 逻辑回归）；
3. 学生模型可离线、免 API、毫秒级地推理全量评论，成本几乎为零。

学生模型刻意只用 sklearn（项目已有依赖）+ 字符级 n-gram，不依赖分词器，
中文无需 jieba 即可工作；评估用留出集的准确率 / 宏 F1 / 与老师标签的一致率。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import config, text_pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = config.OUTPUT_DIR / "sentiment_clf.joblib"


def build_training_data(
    comments: pd.DataFrame,
    *,
    text_column: str = "Comment_Content",
    sample: int = 600,
    client: Any | None = None,
) -> pd.DataFrame:
    """用 LLM 给抽样评论打情感标签，产出 (clean_text, llm_sentiment) 训练集（老师标注）。"""
    analyzed = text_pipeline.analyze_comments(
        comments, text_column=text_column, sample=sample, client=client
    )
    return analyzed[["clean_text", "llm_sentiment"]].rename(
        columns={"clean_text": "text", "llm_sentiment": "label"}
    )


def _build_pipeline() -> Any:
    """字符级 TF-IDF + 逻辑回归：对中文无需分词，小样本下稳健。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_classifier(
    texts: list[str], labels: list[str], *, test_size: float = 0.25, random_state: int = 42
) -> dict:
    """在 (文本→老师标签) 上训练学生分类器，留出集评估后再用全量重训一个可部署模型。

    返回 model（全量拟合）、metrics（留出集 accuracy / macro_f1 / 与老师一致率 / 分类报告）。
    """
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split

    if len(set(labels)) < 2:
        raise ValueError("训练数据只有单一类别，无法训练分类器（增大 sample 或检查标注分布）")

    # 类别样本过少时分层会失败，回退非分层划分
    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        x_tr, x_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )

    eval_model = _build_pipeline()
    eval_model.fit(x_tr, y_tr)
    pred = eval_model.predict(x_te)

    metrics = {
        "n_total": len(texts),
        "n_train": len(x_tr),
        "n_test": len(x_te),
        "accuracy": round(float(accuracy_score(y_te, pred)), 3),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 3),
        "label_dist": pd.Series(labels).value_counts().to_dict(),
        "report": classification_report(y_te, pred, zero_division=0, output_dict=True),
    }
    # accuracy 即"学生在留出集上与老师标签的一致率"
    metrics["agreement_with_teacher"] = metrics["accuracy"]

    final_model = clone(eval_model).fit(texts, labels)
    return {"model": final_model, "metrics": metrics}


def save_model(model: Any, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    import joblib

    return joblib.load(path)


def predict(model: Any, texts: list[str]) -> list[str]:
    """用训练好的学生模型批量推理情感，免 API、毫秒级。"""
    cleaned = [text_pipeline.clean_text(t) for t in texts]
    return [str(p) for p in model.predict(cleaned)]
