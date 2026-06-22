"""智能路由 / 多轨道编排 Agent 层。

把既有的四条打标能力（关键词 / 蒸馏 / 本地 LoRA / RAG-DeepSeek）统一成可调度的
「轨道」，由 RouterAgent 按评论难度做算力分配，并以「检索 → 推理 → 校验」三角
作为内部策略：校验不过则沿成本阶梯升档重判。详见 router / tracks / verifier。
"""

from __future__ import annotations

from .base import Difficulty, TaggingTrack, TagResult, Verdict, Verifier
from .router import RouterAgent
from .tracks import (
    DistilledTrack,
    KeywordTrack,
    LoRATrack,
    RagLLMTrack,
    build_default_tracks,
)
from .verifier import HeuristicVerifier, LLMVerifier

__all__ = [
    "Difficulty",
    "TaggingTrack",
    "TagResult",
    "Verdict",
    "Verifier",
    "RouterAgent",
    "KeywordTrack",
    "DistilledTrack",
    "LoRATrack",
    "RagLLMTrack",
    "build_default_tracks",
    "HeuristicVerifier",
    "LLMVerifier",
]
