"""RAG 检索器/向量库单测：全部用假嵌入，不触网、不依赖 GPU 或 chromadb。"""

from __future__ import annotations

from src.rag.embeddings import HashingEmbedding, get_embedding_function
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import InMemoryVectorStore, get_vector_store

# ---- 向量库 ----


def test_in_memory_store_query_ranks_by_cosine(fake_embedding_fn):
    store = InMemoryVectorStore()
    docs = ["歪了 抽卡", "圣遗物 词条", "原石"]
    store.add(["a", "b", "c"], docs, fake_embedding_fn(docs))
    hits = store.query(fake_embedding_fn(["歪了"])[0], top_k=2)
    assert len(hits) == 2
    assert hits[0].id == "a"  # 最相关的「歪了 抽卡」排第一
    assert hits[0].score >= hits[1].score


def test_in_memory_store_empty_returns_nothing(fake_embedding_fn):
    store = InMemoryVectorStore()
    assert store.query(fake_embedding_fn(["任意"])[0]) == []
    assert len(store) == 0


def test_get_vector_store_memory_backend():
    assert isinstance(get_vector_store(backend="memory"), InMemoryVectorStore)


def test_get_vector_store_auto_falls_back_without_chroma():
    # auto 在无 chromadb 时应静默回退内存库而非报错
    store = get_vector_store(backend="auto")
    assert store is not None


# ---- 混合检索器 ----


def test_retriever_finds_relevant_lore(rag_retriever):
    hits = rag_retriever.retrieve("歪了", top_k=2)
    assert hits
    assert "歪了" in hits[0].document


def test_retriever_empty_corpus(fake_embedding_fn):
    retriever = HybridRetriever.from_documents([], embedding_fn=fake_embedding_fn)
    assert retriever.retrieve("歪了") == []


def test_retrieve_terms_filters_and_groups(rag_retriever):
    contexts = rag_retriever.retrieve_terms(["歪了", "原石"], per_term=1, min_score=0.0)
    terms = {c.term for c in contexts}
    assert "歪了" in terms
    for c in contexts:
        assert c.snippets  # 每个命中的词都带至少一条释义


def test_retriever_hybrid_uses_sparse_for_exact_token(rag_retriever):
    # 即使语义嵌入维度不含某词，BM25 稀疏侧也能靠字面命中召回
    hits = rag_retriever.retrieve("保底", top_k=1)
    assert hits
    assert "保底" in hits[0].document


# ---- 嵌入 ----


def test_hashing_embedding_is_deterministic_and_normalized():
    emb = HashingEmbedding(dim=64)
    a = emb(["抽卡歪了"])
    b = emb(["抽卡歪了"])
    assert a == b  # 确定性
    norm = sum(x * x for x in a[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-5  # L2 归一化


def test_get_embedding_function_defaults_to_hashing():
    assert isinstance(get_embedding_function(), HashingEmbedding)
    # 给了不存在的模型名也应优雅回退而非抛错
    assert isinstance(get_embedding_function("does/not-exist"), HashingEmbedding)
