"""智能路由 / 编排 Agent：按难度把每条评论分配到最省的可行轨道，
并以「检索 → 推理 → 校验」三角作为算力分配策略，校验不过则沿成本阶梯升档重判。

设计要点：
- 路由是「算力分配器」：先用零成本的离线难度画像（黑话 / 反讽 / 长度）判断该花多少算力；
  容易的评论走便宜轨道（关键词 / 蒸馏），难的直接起步于语义轨道（LoRA / RAG-DeepSeek）。
- 三角即「按需投入」：只有进入语义轨道的评论才会触发「检索证据 → LLM 推理 → critic 校验」；
  校验不通过就升一档重判（keyword→distilled→lora→rag_llm），直到通过或阶梯到顶。
- 环境自适应：不可用的轨道（无 API / 无 GPU / 无模型）被自动过滤，离线时优雅退化到
  关键词 / 蒸馏，绝不崩溃，CI 无需任何外部资源。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .. import text_pipeline
from .base import Difficulty, TaggingTrack, TagResult, Verifier
from .tracks import build_default_tracks
from .verifier import HeuristicVerifier

logger = logging.getLogger(__name__)

# 反讽 / 阴阳怪气标记：字面与真实情感常相反，需升级到语义轨道才能判准。
_IRONY_MARKERS = (
    "呵呵", "好家伙", "笑死", "行吧", "懂的都懂", "绝了", "蚌埠住了", "典",
    "乐", "阴阳", "牛", "666", "好评（", "谢谢你米哈游", "格局打开",
)


class RouterAgent:
    """编排多轨道打标的路由 Agent。"""

    def __init__(
        self,
        tracks: Sequence[TaggingTrack],
        *,
        verifier: Verifier | None = None,
        retriever: Any | None = None,
        max_escalations: int = 2,
    ) -> None:
        # 只保留可用轨道，并按成本升序排成升级阶梯
        avail = [t for t in tracks if t.is_available()]
        if not avail:
            raise RuntimeError("没有任何可用打标轨道（至少应有永远可用的 KeywordTrack）。")
        self.ladder: list[TaggingTrack] = sorted(avail, key=lambda t: t.cost)
        self._by_name = {t.name: t for t in self.ladder}
        self.verifier = verifier or HeuristicVerifier()
        self.retriever = retriever
        self.max_escalations = max_escalations
        self.last_stats: dict[str, Any] = {}

    @classmethod
    def from_environment(
        cls,
        *,
        client: Any | None = None,
        retriever: Any | None = None,
        served_client: Any | None = None,
        verifier: Verifier | None = None,
        distilled_path: Any | None = None,
        lora_adapter: Any | None = None,
        max_escalations: int = 2,
    ) -> RouterAgent:
        """按当前环境自动组装可用轨道并构建路由（看板 / 脚本一行即用）。"""
        tracks = build_default_tracks(
            client=client,
            retriever=retriever,
            served_client=served_client,
            distilled_path=distilled_path,
            lora_adapter=lora_adapter,
        )
        return cls(
            tracks, verifier=verifier, retriever=retriever, max_escalations=max_escalations
        )

    # ---- 难度画像与路由决策 ----

    def difficulty(self, text: str) -> Difficulty:
        """离线算出难度画像：黑话越多、含反讽、越长 → 越该往语义轨道倾斜。"""
        jargon = text_pipeline.detect_jargon([text])
        irony = any(m in text for m in _IRONY_MARKERS)
        length = len(text)
        score = min(1.0, 0.25 * len(jargon) + (0.4 if irony else 0.0) + min(length, 80) / 400)
        return Difficulty(score=round(score, 3), jargon=jargon, irony=irony, length=length)

    def _semantic_tracks(self) -> list[TaggingTrack]:
        """阶梯中具备语义理解能力的轨道（云端/本地 LoRA、RAG-LLM）。"""
        return [t for t in self.ladder if t.name in ("lora_server", "lora", "rag_llm")]

    def initial_track(self, diff: Difficulty) -> TaggingTrack:
        """选起步轨道：难的直接起步于最省的语义轨道，容易的走最省的轨道。"""
        if diff.needs_llm:
            sem = self._semantic_tracks()
            if sem:
                return sem[0]
        return self.ladder[0]

    def _next_track(self, current: TaggingTrack) -> TaggingTrack | None:
        """成本阶梯上的下一档（用于校验不过时升级），到顶返回 None。"""
        for t in self.ladder:
            if t.cost > current.cost:
                return t
        return None

    def _evidence(self, text: str) -> list[str]:
        """为校验/推理取证据：命中黑话则从词典检索释义片段。"""
        if self.retriever is None:
            return []
        terms = text_pipeline.detect_jargon([text])
        snippets: list[str] = []
        for ctx in self.retriever.retrieve_terms(terms):
            snippets.extend(ctx.snippets)
        return snippets

    # ---- 主流程：分配 → 三角 → 升档 ----

    def tag(self, texts: list[str]) -> list[TagResult]:
        """对一批评论编排打标，返回与输入等长、带轨道/置信/校验状态的结果。"""
        results: dict[int, TagResult] = {}
        # 每个待办项：(原始下标, 文本, 当前轨道, 已升档次数)
        queue: list[tuple[int, str, TaggingTrack, int]] = []
        for i, t in enumerate(texts):
            queue.append((i, t, self.initial_track(self.difficulty(t)), 0))

        route_counts: dict[str, int] = defaultdict(int)
        n_escalated = 0

        while queue:
            # 按轨道分组，便于各轨道内部批量调用（尤其 LLM 省 token）
            groups: dict[str, list[tuple[int, str, TaggingTrack, int]]] = defaultdict(list)
            for item in queue:
                groups[item[2].name].append(item)

            next_queue: list[tuple[int, str, TaggingTrack, int]] = []
            for name, items in groups.items():
                track = self._by_name[name]
                batch = [it[1] for it in items]
                outs = track.classify(batch)
                for (idx, text, cur, esc), res in zip(items, outs):
                    res.track = name
                    res.escalations = esc
                    evidence = self._evidence(text)
                    verdict = self.verifier.verify(res, evidence=evidence)
                    res.confidence = verdict.confidence
                    higher = self._next_track(cur)
                    can_escalate = higher is not None and esc < self.max_escalations
                    if verdict.ok or not can_escalate:
                        res.verified = verdict.ok
                        # 终判仍不可信且 critic 给了纠正，采纳纠正并记录
                        if not verdict.ok and verdict.corrected_sentiment:
                            corrected = verdict.corrected_sentiment
                            res.sentiment = corrected
                            res.reason = f"{res.reason}；校验纠正→{corrected}".strip("；")
                        route_counts[name] += 1
                        results[idx] = res
                    else:
                        n_escalated += 1
                        assert higher is not None
                        next_queue.append((idx, text, higher, esc + 1))
            queue = next_queue

        self.last_stats = {
            "n": len(texts),
            "ladder": [t.name for t in self.ladder],
            "route_counts": dict(route_counts),
            "n_escalated": n_escalated,
            "n_verified": sum(1 for r in results.values() if r.verified),
        }
        return [results[i] for i in range(len(texts))]
