"""嵌入函数抽象：把文本编码成向量，供向量库检索使用。

提供两档实现，按「能离线就离线」的项目原则分层：
- HashingEmbedding：纯 numpy 的特征哈希嵌入，零依赖、确定性、毫秒级，
  默认档位——保证无 GPU / 无外网的 CI 与本机也能跑通整条 RAG 链路；
- SentenceTransformerEmbedding：可选的语义嵌入（延迟导入，属 rag extra），
  检索质量更高，配好 eGPU 时启用。

EmbeddingFunction 是一个最小协议（只需可调用），因此测试里可直接注入「假嵌入」。
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """中英混排分词：英文/数字按词，中文按字 unigram + bigram（无需 jieba）。"""
    text = (text or "").lower()
    tokens: list[str] = _WORD_RE.findall(text)
    cjk = [c for c in text if "一" <= c <= "鿿"]
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


@runtime_checkable
class EmbeddingFunction(Protocol):
    """可调用对象：一批文本 -> 一批等长向量。测试用假实现也只需满足此协议。"""

    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedding:
    """特征哈希嵌入：把 token 散列到固定维度并带符号累加，再做 L2 归一化。

    无需训练 / 无外部模型，确定性可复现；对中文黑话的「字面相似」召回足够支撑
    词典式检索，是离线与 CI 场景的默认嵌入。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokenize(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vecs[i, idx] += sign
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms
        return vecs.astype(np.float32).tolist()


class SentenceTransformerEmbedding:
    """可选的语义嵌入后端（延迟导入 sentence-transformers，属 rag extra）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入：未装也不影响默认档

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        emb = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(emb, dtype=np.float32).tolist()


def get_embedding_function(
    model_name: str | None = None, *, dim: int = 256
) -> EmbeddingFunction:
    """工厂：给定模型名则尝试语义嵌入，失败/未指定则回退哈希嵌入（永远可用）。"""
    if model_name:
        try:
            return SentenceTransformerEmbedding(model_name)
        except Exception as e:  # noqa: BLE001 - 缺依赖/缺模型时优雅降级
            logger.warning("加载语义嵌入 %s 失败，回退哈希嵌入：%s", model_name, e)
    return HashingEmbedding(dim=dim)
