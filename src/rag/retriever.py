"""混合检索器：稠密向量召回 + 稀疏 BM25 召回，加权融合。

为什么混合：稠密嵌入擅长语义/近义（「保底没了」≈「歪了」），但对低频专有黑话
（角色缩写、活动代号）召回不稳；BM25 这类稀疏检索对「字面精确命中」最敏感。
两者加权融合（alpha 控制偏重稠密还是稀疏）能兼顾「懂梗」与「认词」。

对接 text_pipeline：retrieve_terms 针对一批疑似黑话词逐个取最佳释义，
拼成可注入系统提示的领域知识上下文。
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embeddings import EmbeddingFunction, _tokenize, get_embedding_function
from .vector_store import BaseVectorStore, Hit, InMemoryVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """一次黑话检索结果：触发词 + 命中的领域知识条目。"""

    term: str
    snippets: list[str]


class _BM25:
    """精简 BM25：对中英混排语料给出与查询的稀疏相关性分数。"""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tfs = [Counter(d) for d in corpus_tokens]
        df: Counter[str] = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return scores
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    """稠密 + 稀疏混合检索器。alpha 越大越偏重语义（稠密）召回。"""

    def __init__(
        self,
        store: BaseVectorStore,
        embedding_fn: EmbeddingFunction,
        *,
        documents: Sequence[str],
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
        alpha: float = 0.5,
    ) -> None:
        self.store = store
        self.embed = embedding_fn
        self.alpha = alpha
        self.documents = list(documents)
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(documents))]
        self.metadatas = (
            list(metadatas) if metadatas is not None else [{} for _ in self.documents]
        )
        self._id_to_pos = {i: p for p, i in enumerate(self.ids)}
        self._bm25 = _BM25([_tokenize(d) for d in self.documents])

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[str],
        *,
        embedding_fn: EmbeddingFunction | None = None,
        store: BaseVectorStore | None = None,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
        alpha: float = 0.5,
    ) -> HybridRetriever:
        """便捷构造：嵌入文档、写入向量库并建好稀疏索引，一步到位。"""
        embed = embedding_fn or get_embedding_function()
        store = store or InMemoryVectorStore()
        ids_list = list(ids) if ids is not None else [str(i) for i in range(len(documents))]
        metas = list(metadatas) if metadatas is not None else [{} for _ in documents]
        if documents:
            store.add(ids_list, list(documents), embed(list(documents)), metas)
        return cls(
            store,
            embed,
            documents=documents,
            ids=ids_list,
            metadatas=metas,
            alpha=alpha,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]:
        """对单条查询做混合检索，返回融合后 top_k 命中。"""
        if not self.documents:
            return []
        cand = min(len(self.documents), max(top_k * 4, top_k))
        dense_hits = self.store.query(self.embed([query])[0], top_k=cand)
        dense = np.zeros(len(self.documents), dtype=np.float32)
        for h in dense_hits:
            pos = self._id_to_pos.get(h.id)
            if pos is not None:
                dense[pos] = h.score
        sparse = self._bm25.scores(_tokenize(query))
        fused = self.alpha * _minmax(dense) + (1 - self.alpha) * _minmax(sparse)
        k = min(top_k, len(self.documents))
        order = np.argsort(-fused)[:k]
        return [
            Hit(self.ids[p], self.documents[p], float(fused[p]), self.metadatas[p])
            for p in order
            if fused[p] > 0
        ]

    def retrieve_terms(
        self, terms: Sequence[str], *, per_term: int = 1, min_score: float = 0.05
    ) -> list[RetrievedContext]:
        """针对一批疑似黑话词逐个检索释义，过滤低分命中。"""
        out: list[RetrievedContext] = []
        for term in terms:
            hits = self.retrieve(term, top_k=per_term)
            snippets = [h.document for h in hits if h.score >= min_score]
            if snippets:
                out.append(RetrievedContext(term=term, snippets=snippets))
        return out
