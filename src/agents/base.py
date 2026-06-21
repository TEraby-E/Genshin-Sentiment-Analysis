"""智能路由层的公共契约：数据结构、协议与共享情感词典。

这里只放「轻」的东西（dataclass / Protocol / 纯函数词典），不导入任何重依赖，
保证 router/tracks/verifier 都能零成本引用，且 mypy / CI 无 GPU 也能过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

SENTIMENTS = ("正面", "中性", "负面")

# 轻量情感词典：供「关键词轨道」打分，也供「校验者」判断标签是否与字面信号冲突。
POS_WORDS = (
    "好", "喜欢", "强", "赞", "爱", "香", "良心", "舒服", "期待", "支持",
    "可爱", "帅", "顶", "绝", "感动", "用心", "惊艳", "值", "爽", "牛逼",
)
NEG_WORDS = (
    "歪", "差", "烂", "坑", "崩", "卡顿", "退环境", "失望", "拉胯", "难受",
    "恶心", "削", "白嫖", "退游", "摆烂", "敷衍", "下水道", "拷打", "垃圾", "骂",
)


def lexicon_polarity(text: str) -> tuple[str, float]:
    """返回 (情感标签, 强度0~1)：纯词典命中计数，离线、确定性，用作校验基准信号。"""
    if not isinstance(text, str) or not text:
        return ("中性", 0.0)
    pos = sum(text.count(w) for w in POS_WORDS)
    neg = sum(text.count(w) for w in NEG_WORDS)
    total = pos + neg
    if total == 0:
        return ("中性", 0.0)
    if pos > neg:
        return ("正面", (pos - neg) / total)
    if neg > pos:
        return ("负面", (neg - pos) / total)
    return ("中性", 0.0)


@dataclass
class TagResult:
    """一条评论的打标结果，贯穿全链路（轨道产出 → 校验 → 路由汇总）。"""

    text: str
    sentiment: str
    aspects: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    track: str = ""
    verified: bool = False
    escalations: int = 0


@dataclass
class Difficulty:
    """路由对单条评论的难度画像（离线、零成本算出）。"""

    score: float  # 0~1，越大越难，越需要把算力往语义轨道倾斜
    jargon: list[str]
    irony: bool
    length: int

    @property
    def needs_llm(self) -> bool:
        """是否应直接起步于语义（LLM/LoRA）轨道，而非便宜轨道。"""
        return self.score >= 0.5 or bool(self.jargon) or self.irony


@dataclass
class Verdict:
    """校验者（critic）对一条结果的裁决。"""

    ok: bool
    confidence: float
    corrected_sentiment: str | None = None
    note: str = ""


@runtime_checkable
class TaggingTrack(Protocol):
    """打标轨道统一接口：路由据此把任意轨道当作可替换的算力档位调度。"""

    name: str
    cost: int  # 相对算力/成本档位，路由按它排升级阶梯

    def is_available(self) -> bool: ...

    def classify(self, texts: list[str]) -> list[TagResult]: ...


@runtime_checkable
class Verifier(Protocol):
    """校验者接口：给定结果与（可选）检索证据，裁决是否可信。"""

    def verify(self, result: TagResult, *, evidence: list[str]) -> Verdict: ...
