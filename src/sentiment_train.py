"""舆情打标"训练"：LLM 标注小样本 → 蒸馏出轻量本地分类器。

动机：直接用 LLM 给全量 40.8 万条评论打情感标签，API 成本与耗时都不可接受。
做法是知识蒸馏（knowledge distillation）：
1. 用主流 AI 模型（DeepSeek）作为高精度标注源，给一个**小样本**打高质量情感标签；
2. 在这批 (文本 → LLM 标签) 上训练一个**轻量下游分类器**（字符级 TF-IDF + 逻辑回归）；
3. 下游分类器可离线、免 API、毫秒级地推理全量评论，成本几乎为零。

下游分类器刻意只用 sklearn（项目已有依赖）+ 字符级 n-gram，不依赖分词器，
中文无需 jieba 即可工作；评估用留出集的准确率 / 宏 F1 / 与标注源标签的一致率。
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


# ---- 进阶轨道：本地微调大模型分类器（Qwen2.5 + LoRA） ----

DEFAULT_LORA_DIR = config.LORA_ADAPTER_DIR


class LocalLLMClassifier:
    """加载本地微调大模型（基座 + LoRA 适配器）做情感分类，离线、免 API。

    与 TF-IDF 蒸馏并行的「重」轨道：更懂语义与黑话，代价是需要本地 GPU 显存。
    一切重依赖（transformers/peft/torch）延迟到 load() 时导入，未安装也不影响
    本模块其余功能与 CI（属可选 finetune extra）。
    """

    def __init__(
        self,
        adapter_path: str | Path = DEFAULT_LORA_DIR,
        *,
        base_model: str = config.LORA_BASE_MODEL,
        load_in_4bit: bool = True,
        max_new_tokens: int = 48,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    @staticmethod
    def deps_available() -> bool:
        """transformers/peft 是否可导入（不触发实际加载）。"""
        import importlib.util

        return all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))

    def is_ready(self) -> bool:
        """适配器已就绪且依赖可用——dashboard 据此决定是否放行该模式。"""
        return self.adapter_path.exists() and self.deps_available()

    def load(self) -> LocalLLMClassifier:
        """加载基座（可选 4bit 量化）+ LoRA 适配器与分词器。"""
        import torch  # noqa: F401 - 触发 CUDA 初始化并校验依赖
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "加载 LoRA 大模型：base=%s adapter=%s（4bit=%s，首次加载较慢，请耐心等待）",
            self.base_model, self.adapter_path, self.load_in_4bit,
        )
        quant_kwargs: dict[str, Any] = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        logger.info("分词器就绪，正在加载基座权重…")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, device_map="auto", torch_dtype=torch.bfloat16, **quant_kwargs
        )
        logger.info("基座就绪，正在注入 LoRA 适配器…")
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path)).eval()
        logger.info("✅ LoRA 大模型加载完成，开始推理")
        return self

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        from .finetune.dataset_formatter import INSTRUCTION

        return [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": text},
        ]

    def _parse_label(self, generated: str) -> str:
        """从模型输出里抽取合法情感标签，失败兜底为中性。"""
        import json

        try:
            start, end = generated.index("{"), generated.rindex("}") + 1
            sentiment = json.loads(generated[start:end]).get("sentiment")
            if sentiment in config.LLM_SENTIMENT_LABELS:
                return str(sentiment)
        except (ValueError, json.JSONDecodeError):
            pass
        for label in config.LLM_SENTIMENT_LABELS:
            if label in generated:
                return label
        return "中性"

    def predict(self, texts: list[str]) -> list[str]:
        """批量推理情感标签。需先 load()。"""
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        n = len(texts)
        step = max(1, n // 20)  # 大约打印 20 次进度，避免刷屏
        out: list[str] = []
        for i, text in enumerate(texts, 1):
            clean = text_pipeline.clean_text(text)
            prompt = self._tokenizer.apply_chat_template(
                self._build_messages(clean), tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            decoded = self._tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            out.append(self._parse_label(decoded))
            if i == 1 or i % step == 0 or i == n:
                logger.info("LoRA 推理进度 %d/%d", i, n)
        return out
