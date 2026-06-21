"""校验者（critic）：三角的「校验」一角，复核打标结果是否可信。

两档实现，体现「算力分配」思想：
- HeuristicVerifier：纯规则、零成本、确定性——用词典极性与结果完整性给出置信与裁决，
  默认档，离线 / CI 可跑；
- LLMVerifier：可选的 LLM 评审员，把「评论 + 检索证据 + 拟定标签」交给模型复核，
  更准但要花一次调用，仅在需要更高把握时启用（可注入 fake client 测试）。

裁决不通过（ok=False）时，路由会据此把该条升到更高算力的轨道重判。
"""

from __future__ import annotations

import logging
from typing import Any

from .. import config
from .base import TagResult, Verdict, lexicon_polarity

logger = logging.getLogger(__name__)


class HeuristicVerifier:
    """规则校验：词典极性 vs. 标签是否冲突 + 结果完整性，给出置信与裁决。"""

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold

    def verify(self, result: TagResult, *, evidence: list[str]) -> Verdict:
        pol_label, strength = lexicon_polarity(result.text)

        # 硬冲突：字面强烈一极，标签却判成相反一极 → 直接打回并给出纠正
        opposite = {"正面": "负面", "负面": "正面"}
        if (
            pol_label in opposite
            and result.sentiment == opposite[pol_label]
            and strength >= 0.5
        ):
            return Verdict(
                ok=False,
                confidence=round(min(result.confidence, 0.3), 3),
                corrected_sentiment=pol_label,
                note=f"字面强烈{pol_label}，与判定{result.sentiment}冲突",
            )

        conf = result.confidence if result.confidence > 0 else 0.5
        # 与词典方向一致则加分，中性/弱信号不奖不罚
        if pol_label != "中性":
            conf += 0.12 if result.sentiment == pol_label else -0.1
        # 完整性：方面合法 + 有依据
        if any(a in config.LLM_ASPECT_LABELS for a in result.aspects):
            conf += 0.05
        if result.reason.strip():
            conf += 0.05
        # 命中黑话却没检索到任何证据，降低把握（促使升档去取证据）
        if evidence:
            conf += 0.03
        conf = round(max(0.0, min(conf, 1.0)), 3)
        return Verdict(ok=conf >= self.threshold, confidence=conf, note="规则校验")


_VERIFY_SYSTEM = (
    "你是游戏舆情标注质检员。给定一条玩家评论、可选的领域释义证据，以及一个拟定的情感判定，"
    "判断该判定是否正确。只输出 JSON，形如 "
    '{"ok": true, "sentiment": "负面", "note": "理由"}。'
    f"sentiment 必须是其中之一：{config.LLM_SENTIMENT_LABELS}。"
)


class LLMVerifier:
    """LLM 评审员：花一次调用让模型复核标签，准但有成本（默认不启用）。"""

    def __init__(self, *, client: Any | None = None) -> None:
        self.client = client

    def verify(self, result: TagResult, *, evidence: list[str]) -> Verdict:
        from .. import llm_client

        ev = ("\n证据：\n" + "\n".join(f"- {e}" for e in evidence)) if evidence else ""
        user = (
            f"评论：{result.text}\n拟定情感：{result.sentiment}{ev}\n"
            "这个情感判定正确吗？"
        )
        try:
            raw = llm_client.chat_json(_VERIFY_SYSTEM, user, client=self.client)
        except Exception as e:  # noqa: BLE001 - 评审调用失败则不阻断，按通过处理
            logger.warning("LLM 校验失败，跳过复核：%s", e)
            return Verdict(ok=True, confidence=result.confidence, note="校验调用失败，放行")

        ok = bool(raw.get("ok", True))
        corrected = raw.get("sentiment")
        if corrected not in config.LLM_SENTIMENT_LABELS:
            corrected = None
        conf = 0.9 if ok else 0.35
        return Verdict(
            ok=ok,
            confidence=conf,
            corrected_sentiment=None if ok else corrected,
            note=str(raw.get("note", "LLM 复核")),
        )
