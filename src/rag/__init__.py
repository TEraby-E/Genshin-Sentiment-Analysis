"""RAG（检索增强生成）子系统：为 LLM 调用注入原神领域知识，对抗幻觉。

动机：玩家评论里充斥黑话/梗/缩写（如「歪了」「保底」「深渊」「螺旋」「胡桃下水道」），
通用 LLM 缺乏这些时效性强、社区内生的语义，容易误判情感或方面。RAG 在调用前
先从本地「梗 & 设定词典」检索相关释义，注入系统提示，让模型「先懂梗再判断」。

设计原则（与项目既有约束一致）：
- 轻量本地：默认走纯 numpy 的内存向量库 + 哈希嵌入，零外部基建、可离线、CI 友好；
  需要持久化/规模化时再切到 ChromaDB（延迟导入，属可选 rag extra）。
- 不污染核心链路：text_pipeline 仅在显式传入 retriever 时启用，默认行为完全不变。
"""

from __future__ import annotations

from .embeddings import EmbeddingFunction, HashingEmbedding, get_embedding_function
from .retriever import HybridRetriever, RetrievedContext
from .vector_store import BaseVectorStore, Hit, InMemoryVectorStore, get_vector_store

__all__ = [
    "EmbeddingFunction",
    "HashingEmbedding",
    "get_embedding_function",
    "HybridRetriever",
    "RetrievedContext",
    "BaseVectorStore",
    "Hit",
    "InMemoryVectorStore",
    "get_vector_store",
]
