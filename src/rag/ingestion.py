"""异步构建「梗 & 设定词典」：把原神数据集切块、嵌入、灌入向量库。

数据来源（2024.06–2025.11 数据集，不改 data_loader，直接按 config.FILES 读取）：
1. 评论聚类关键词（comment_keywords）：每个聚类的 Top_Words + 簇名，是最稠密的
   「社区议题/黑话」释义来源——一行就是一条高质量词典条目；
2. 官方动态正文（posts）：高互动（点赞/评论）的官方贴正文切块，承载活动/角色的
   官方说法与时间线，给模型「权威设定」；
3. 高赞 UGC 评论：作为「梗的真实用法」语料，帮助模型把黑话和语境对上。

异步：嵌入是 CPU/GPU 密集的阻塞调用，用 asyncio.to_thread 把分批嵌入并发出去，
在配了 GPU 的语义嵌入档下能明显缩短整库构建时间；纯哈希嵌入档同样可跑。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd

from .. import config
from .embeddings import EmbeddingFunction, get_embedding_function
from .retriever import HybridRetriever
from .vector_store import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = config.RAG_PERSIST_DIR
_CHUNK_SIZE = 280
_CHUNK_OVERLAP = 40


def _read_csv(name: str) -> pd.DataFrame | None:
    path = config.DATA_DIR / config.FILES[name]
    if not path.exists():
        logger.warning("缺少数据文件 %s，跳过该来源", path)
        return None
    return pd.read_csv(path)


def _chunk(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    text = " ".join(str(text).split())
    if len(text) <= size:
        return [text] if text else []
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def collect_documents(
    *, max_posts: int = 300, max_comments: int = 2000
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """汇总三类来源为 (documents, ids, metadatas)，供向量库写入。"""
    docs: list[str] = []
    ids: list[str] = []
    metas: list[dict[str, str]] = []

    kw = _read_csv("comment_keywords")
    if kw is not None and {"Top_Words", "Cluster_Name"}.issubset(kw.columns):
        for _, row in kw.iterrows():
            entry = f"【社区议题】{row['Cluster_Name']}：常见词 {row['Top_Words']}"
            docs.append(entry)
            ids.append(f"kw-{row.get('Cluster', len(ids))}")
            metas.append({"source": "comment_keywords", "cluster": str(row.get("Cluster", ""))})

    posts = _read_csv("posts")
    if posts is not None and "Content" in posts.columns:
        engagement = posts.get("Like_Count")
        if engagement is not None:
            posts = posts.sort_values("Like_Count", ascending=False)
        for _, row in posts.head(max_posts).iterrows():
            for j, chunk in enumerate(_chunk(row.get("Content", ""))):
                docs.append(f"【官方动态】{chunk}")
                ids.append(f"post-{row.get('Post_ID', len(ids))}-{j}")
                metas.append({"source": "posts", "post_id": str(row.get("Post_ID", ""))})

    comments = _read_csv("comments")
    if comments is not None and "Comment_Content" in comments.columns:
        seen: set[str] = set()
        for _, row in comments.head(max_comments * 3).iterrows():
            text = " ".join(str(row.get("Comment_Content", "")).split())
            if len(text) < 6 or text in seen:
                continue
            seen.add(text)
            docs.append(f"【玩家用法】{text}")
            ids.append(f"cmt-{len(ids)}")
            metas.append({"source": "comments", "cluster": str(row.get("Cluster", ""))})
            if len(seen) >= max_comments:
                break

    logger.info("汇总词典条目 %d 条（关键词/官方/UGC）", len(docs))
    return docs, ids, metas


async def _embed_async(
    embed: EmbeddingFunction, docs: list[str], *, batch_size: int = 256
) -> list[list[float]]:
    """分批把嵌入放到线程池里并发执行（嵌入是阻塞调用）。"""
    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
    results = await asyncio.gather(*(asyncio.to_thread(embed, b) for b in batches))
    out: list[list[float]] = []
    for r in results:
        out.extend(r)
    return out


async def build_lore_dictionary(
    *,
    store: BaseVectorStore | None = None,
    embedding_fn: EmbeddingFunction | None = None,
    persist_dir: str | Path | None = DEFAULT_PERSIST_DIR,
    embedding_model: str | None = None,
    max_posts: int = 300,
    max_comments: int = 2000,
) -> HybridRetriever:
    """异步构建词典并返回可直接检索的 HybridRetriever。"""
    embed = embedding_fn or get_embedding_function(embedding_model or config.RAG_EMBEDDING_MODEL)
    store = store or get_vector_store(
        backend=config.RAG_VECTOR_BACKEND, persist_dir=persist_dir
    )
    docs, ids, metas = collect_documents(max_posts=max_posts, max_comments=max_comments)
    if not docs:
        raise RuntimeError("未采集到任何词典条目：请确认 data/ 下已放置数据集 CSV。")
    embeddings = await _embed_async(embed, docs)
    store.add(ids, docs, embeddings, metas)
    logger.info("词典已写入向量库，规模 %d 条", len(store))
    return HybridRetriever(store, embed, documents=docs, ids=ids, metadatas=metas)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构建原神梗&设定词典向量库")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument(
        "--embedding-model",
        default=config.RAG_EMBEDDING_MODEL,
        help="留空用离线哈希嵌入；可由 RAG_EMBEDDING_MODEL 环境变量配置",
    )
    parser.add_argument("--max-posts", type=int, default=300)
    parser.add_argument("--max-comments", type=int, default=2000)
    args = parser.parse_args()

    retriever = asyncio.run(
        build_lore_dictionary(
            persist_dir=args.persist_dir,
            embedding_model=args.embedding_model,
            max_posts=args.max_posts,
            max_comments=args.max_comments,
        )
    )
    demo = retriever.retrieve("歪了", top_k=2)
    print(f"构建完成，库内 {len(retriever.store)} 条。检索『歪了』示例：")
    for h in demo:
        print(f"  [{h.score:.3f}] {h.document[:60]}")


if __name__ == "__main__":
    main()
