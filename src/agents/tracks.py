"""把既有的四条打标能力封装成统一的「轨道」，供路由当作算力档位调度。

每条轨道实现同一个 TaggingTrack 接口（name / cost / is_available / classify），
路由因此可以「按需选档、失败升档」，而不关心各轨道内部是关键词、sklearn、
OpenAI 兼容 API 还是本地 LoRA 大模型。所有重依赖都延迟导入，未配置的轨道 is_available()
返回 False，被路由直接跳过——保证离线 / 无 GPU / 无 API 的环境也能跑通。

成本档位（cost）由低到高构成升级阶梯：
    KeywordTrack(0) → DistilledTrack(1) → LoRATrack(2) → RagLLMTrack(3)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import aspect_sentiment
from .base import TaggingTrack, TagResult, lexicon_polarity

logger = logging.getLogger(__name__)


class KeywordTrack:
    """关键词 + 词典轨道：纯规则、零成本、永远可用，作为兜底底座。"""

    name = "keyword"
    cost = 0

    def is_available(self) -> bool:
        return True

    def classify(self, texts: list[str]) -> list[TagResult]:
        out: list[TagResult] = []
        for t in texts:
            sentiment, strength = lexicon_polarity(t)
            out.append(
                TagResult(
                    text=t,
                    sentiment=sentiment,
                    aspects=aspect_sentiment.tag_aspects(t),
                    reason="关键词/词典规则命中",
                    confidence=round(min(0.4 + 0.4 * strength, 0.8), 3),
                    track=self.name,
                )
            )
        return out


class DistilledTrack:
    """蒸馏轨道：字符级 TF-IDF + 逻辑回归，毫秒级、免 API，置信度取预测概率。"""

    name = "distilled"
    cost = 1

    def __init__(self, model_path: str | Path | None = None) -> None:
        from .. import sentiment_train

        self.model_path = Path(model_path) if model_path else sentiment_train.DEFAULT_MODEL_PATH
        self._model: Any = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def _load(self) -> Any:
        if self._model is None:
            from .. import sentiment_train

            self._model = sentiment_train.load_model(self.model_path)
        return self._model

    def classify(self, texts: list[str]) -> list[TagResult]:
        from .. import text_pipeline

        model = self._load()
        cleaned = [text_pipeline.clean_text(t) for t in texts]
        labels = [str(x) for x in model.predict(cleaned)]
        # 学生模型是概率分类器：用最大类概率作为自报置信度
        try:
            proba = model.predict_proba(cleaned)
            confs = [round(float(max(row)), 3) for row in proba]
        except (AttributeError, ValueError):
            confs = [0.6] * len(texts)
        return [
            TagResult(
                text=t,
                sentiment=lab,
                aspects=aspect_sentiment.tag_aspects(t),
                reason="蒸馏分类器预测",
                confidence=c,
                track=self.name,
            )
            for t, lab, c in zip(texts, labels, confs)
        ]


class LoRATrack:
    """本地微调大模型轨道：Qwen2.5 + LoRA，离线、免 API，需本地 GPU 与适配器就绪。"""

    name = "lora"
    cost = 2

    def __init__(self, adapter_path: str | Path | None = None) -> None:
        from .. import sentiment_train

        self._cls = sentiment_train.LocalLLMClassifier
        self._clf = (
            self._cls(adapter_path) if adapter_path is not None else self._cls()
        )

    def is_available(self) -> bool:
        return self._clf.is_ready()

    def classify(self, texts: list[str]) -> list[TagResult]:
        labels = self._clf.predict(texts)
        return [
            TagResult(
                text=t,
                sentiment=lab,
                aspects=aspect_sentiment.tag_aspects(t),
                reason="本地微调大模型判定",
                confidence=0.7,
                track=self.name,
            )
            for t, lab in zip(texts, labels)
        ]


def _classify_via_openai(
    texts: list[str],
    *,
    client: Any | None,
    retriever: Any | None,
    track_name: str,
    confidence: float,
) -> list[TagResult]:
    """走 OpenAI 兼容接口做语义打标的共用实现。

    若给了 retriever，则按单条评论检索黑话释义，再把上下文相同的评论分组批量调用。
    这样既保留批处理，又避免整批评论共享无关黑话证据。
    """
    from .. import text_pipeline

    contexts: list[str | None] = []
    for text in texts:
        if retriever is None:
            contexts.append(None)
        else:
            contexts.append(text_pipeline.build_jargon_context([text], retriever) or None)

    preds: list[dict[str, Any] | None] = [None] * len(texts)
    grouped: dict[str | None, list[tuple[int, str]]] = {}
    for i, (text, context) in enumerate(zip(texts, contexts)):
        grouped.setdefault(context, []).append((i, text))

    for context, items in grouped.items():
        batch = [text for _, text in items]
        batch_preds = text_pipeline.classify_with_llm(
            batch, client=client, extra_context=context
        )
        for (idx, _), pred in zip(items, batch_preds):
            preds[idx] = pred

    out: list[TagResult] = []
    for text, pred in zip(texts, preds):
        if pred is None:
            pred = {"sentiment": "中性", "aspects": ["其他"], "reason": "LLM 结果缺失，兜底"}
        out.append(
            TagResult(
                text=text,
                sentiment=str(pred["sentiment"]),
                aspects=list(pred["aspects"]),
                reason=str(pred.get("reason", "")),
                confidence=confidence,  # 自报置信，交由校验者复核后再定稿
                track=track_name,
            )
        )
    return out


class RagLLMTrack:
    """RAG + LLM 轨道：检索黑话释义注入系统提示后做语义打标（三角的「检索→推理」）。"""

    name = "rag_llm"
    cost = 3

    def __init__(self, *, client: Any | None = None, retriever: Any | None = None) -> None:
        self.client = client
        self.retriever = retriever

    def is_available(self) -> bool:
        if self.client is not None:
            return True
        try:
            from .. import llm_client

            llm_client.get_api_key()
            import openai  # noqa: F401

            return True
        except Exception:  # noqa: BLE001 - 未配置 API / 未装 openai 时该轨道不可用
            return False

    def classify(self, texts: list[str]) -> list[TagResult]:
        return _classify_via_openai(
            texts,
            client=self.client,
            retriever=self.retriever,
            track_name=self.name,
            confidence=0.72,
        )


def build_default_tracks(
    *,
    client: Any | None = None,
    retriever: Any | None = None,
    distilled_path: str | Path | None = None,
    lora_adapter: str | Path | None = None,
) -> list[TaggingTrack]:
    """按环境组装全部轨道（不过滤可用性，交给 RouterAgent 处理）。"""
    tracks: list[TaggingTrack] = [
        KeywordTrack(),
        DistilledTrack(distilled_path),
        RagLLMTrack(client=client, retriever=retriever),
    ]
    # 进程内本地 LoRA（需本地 GPU + 适配器）
    try:
        tracks.append(LoRATrack(lora_adapter))
    except Exception as e:  # noqa: BLE001 - 缺 finetune 依赖时不阻塞其余轨道
        logger.info("本地 LoRA 轨道不可用，跳过：%s", e)
    return tracks
