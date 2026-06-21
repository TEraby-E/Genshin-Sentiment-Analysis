"""本地轻量向量库：默认内存档（numpy 余弦），可选 ChromaDB 持久化档。

为什么不直接上 Chroma/Qdrant 服务：项目硬约束是「不引入重型外部基建」。
因此默认 InMemoryVectorStore 用纯 numpy 做余弦相似度，零依赖、零进程、CI 可跑；
当词典规模变大或需要跨会话持久化时，再用 get_vector_store(backend="chroma")
切到 ChromaDB 的本地持久化客户端（延迟导入，属 rag extra）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    """一次检索命中：文档 id、原文、相似度分数与元数据。"""

    id: str
    document: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore:
    """向量库统一接口：add 写入、query 近邻检索、len 规模。"""

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        raise NotImplementedError

    def query(self, embedding: Sequence[float], top_k: int = 5) -> list[Hit]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class InMemoryVectorStore(BaseVectorStore):
    """纯 numpy 内存向量库：写入时归一化，查询时点积即余弦相似度。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._emb: np.ndarray | None = None

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        arr = _normalize(np.asarray(embeddings, dtype=np.float32))
        self._emb = arr if self._emb is None else np.vstack([self._emb, arr])
        self._ids.extend(ids)
        self._docs.extend(documents)
        metas = list(metadatas) if metadatas is not None else [{} for _ in ids]
        self._meta.extend(metas)

    def query(self, embedding: Sequence[float], top_k: int = 5) -> list[Hit]:
        if self._emb is None or len(self._ids) == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._emb @ q
        k = min(top_k, len(self._ids))
        top = np.argsort(-sims)[:k]
        return [
            Hit(self._ids[i], self._docs[i], float(sims[i]), self._meta[i]) for i in top
        ]

    def __len__(self) -> int:
        return len(self._ids)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 本地持久化档（延迟导入）。persist_dir 为空则用临时内存集合。"""

    def __init__(
        self, persist_dir: str | Path | None = None, collection: str = "genshin_lore"
    ) -> None:
        import chromadb  # 延迟导入：未装 chromadb 时默认档仍可用

        if persist_dir is not None:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._client = chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        # Chroma 不接受空 metadata，缺省补一个占位键
        metas = list(metadatas) if metadatas is not None else [{} for _ in ids]
        metas = [m or {"_": ""} for m in metas]
        self._col.add(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(e) for e in embeddings],
            metadatas=metas,
        )

    def query(self, embedding: Sequence[float], top_k: int = 5) -> list[Hit]:
        res = self._col.query(query_embeddings=[list(embedding)], n_results=top_k)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[Hit] = []
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(Hit(str(i), str(doc), 1.0 - float(dist), dict(meta or {})))
        return hits

    def __len__(self) -> int:
        return int(self._col.count())


def get_vector_store(
    *,
    backend: str = "auto",
    persist_dir: str | Path | None = None,
    collection: str = "genshin_lore",
) -> BaseVectorStore:
    """工厂：backend 取 'memory' / 'chroma' / 'auto'。

    'auto'：优先 Chroma（若已安装），否则静默回退内存档——保证默认环境永远可用。
    """
    if backend == "memory":
        return InMemoryVectorStore()
    if backend in ("chroma", "auto"):
        try:
            return ChromaVectorStore(persist_dir, collection)
        except Exception as e:  # noqa: BLE001 - 未装 chromadb 等情况下回退
            if backend == "chroma":
                raise
            logger.info("Chroma 不可用，回退内存向量库：%s", e)
            return InMemoryVectorStore()
    raise ValueError(f"未知向量库后端：{backend!r}")
