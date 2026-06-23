"""智能路由 Agent 单测：用可编程假轨道 + 假 LLM client，确定性验证
难度画像、路由分配、校验三角与成本阶梯升档；全程不触网、不依赖 GPU。
"""

from __future__ import annotations

from src.agents.base import TagResult
from src.agents.router import RouterAgent
from src.agents.tracks import KeywordTrack, RagLLMTrack
from src.agents.verifier import HeuristicVerifier


def _router(tracks, **kw):
    return RouterAgent(tracks, **kw)


# ---- 难度画像 ----


def test_difficulty_flags_jargon_and_irony(make_fake_track):
    r = _router([make_fake_track("keyword", 0)])
    hard = r.difficulty("这次又歪了，保底白给")
    assert hard.jargon  # 命中黑话
    assert hard.needs_llm
    easy = r.difficulty("今天天气不错")
    assert not easy.jargon
    assert not easy.needs_llm


# ---- 路由分配 ----


def test_easy_comment_routes_to_cheapest_track(make_fake_track):
    kw = make_fake_track("keyword", 0, sentiment_fn=lambda _t: "正面")
    llm = make_fake_track("rag_llm", 3, sentiment_fn=lambda _t: "正面")
    r = _router([kw, llm])
    out = r.tag(["剧情真好，角色塑造用心"])
    assert out[0].track == "keyword"  # 容易 → 走最省轨道
    assert llm.calls == []  # 贵轨道未被调用


def test_hard_comment_starts_at_semantic_track(make_fake_track):
    kw = make_fake_track("keyword", 0, sentiment_fn=lambda _t: "中性")
    llm = make_fake_track("rag_llm", 3, sentiment_fn=lambda _t: "负面")
    r = _router([kw, llm])
    out = r.tag(["这次又歪了，保底白给，策划是不是在拷打玩家"])
    assert out[0].track == "rag_llm"  # 含黑话 → 直接起步于语义轨道
    assert out[0].verified


# ---- 校验三角 + 升档 ----


def test_verify_failure_escalates_up_cost_ladder(make_fake_track):
    # 便宜轨道对强负面评论硬判成正面 → 校验冲突 → 升级到语义轨道重判
    kw = make_fake_track("keyword", 0, sentiment_fn=lambda _t: "正面")
    llm = make_fake_track("rag_llm", 3, sentiment_fn=lambda _t: "负面")
    r = _router([kw, llm], verifier=HeuristicVerifier())
    out = r.tag(["这游戏真垃圾，烂透了"])
    assert out[0].track == "rag_llm"
    assert out[0].escalations == 1
    assert out[0].verified
    assert r.last_stats["n_escalated"] == 1


def test_verify_failure_can_escalate_to_top_track_by_default(make_fake_track):
    # 默认 max_escalations=None 时，应允许从最低档一路升到最高可用轨道。
    kw = make_fake_track("keyword", 0, sentiment_fn=lambda _t: "正面")
    distilled = make_fake_track("distilled", 1, sentiment_fn=lambda _t: "正面")
    lora = make_fake_track("lora", 2, sentiment_fn=lambda _t: "正面")
    llm = make_fake_track("rag_llm", 3, sentiment_fn=lambda _t: "负面")
    r = _router([kw, distilled, lora, llm], verifier=HeuristicVerifier())
    out = r.tag(["这游戏真垃圾，烂透了"])
    assert out[0].track == "rag_llm"
    assert out[0].escalations == 3
    assert r.last_stats["n_escalated"] == 3


def test_no_escalation_when_top_track_still_fails(make_fake_track):
    # 只有 keyword 可用且判错：无更高档可升，按最终结果落地（不崩、不死循环）
    kw = make_fake_track("keyword", 0, sentiment_fn=lambda _t: "正面")
    r = _router([kw], verifier=HeuristicVerifier())
    out = r.tag(["这游戏真垃圾，烂透了"])
    assert len(out) == 1
    assert out[0].track == "keyword"
    assert not out[0].verified
    assert out[0].sentiment == "负面"  # critic 给出纠正并被采纳


def test_tag_preserves_order_and_length(make_fake_track):
    kw = make_fake_track("keyword", 0, sentiment_fn=lambda t: "负面" if "烂" in t else "正面")
    r = _router([kw])
    texts = ["很好玩", "太烂了", "还行"]
    out = r.tag(texts)
    assert [o.text for o in out] == texts


# ---- 校验者单测 ----


def test_heuristic_verifier_detects_contradiction():
    v = HeuristicVerifier()
    res = TagResult(text="这游戏真垃圾，烂透了", sentiment="正面", confidence=0.6)
    verdict = v.verify(res, evidence=[])
    assert not verdict.ok
    assert verdict.corrected_sentiment == "负面"


def test_heuristic_verifier_passes_on_agreement():
    v = HeuristicVerifier()
    res = TagResult(
        text="剧情太感人了，真喜欢", sentiment="正面", reason="x", confidence=0.7
    )
    verdict = v.verify(res, evidence=[])
    assert verdict.ok
    assert verdict.confidence >= 0.55


# ---- 真实轨道适配器（注入 fake client / retriever，不触网）----


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _FakeMessage(content)})()]


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = self
        self.completions = self
        self.last_system = None
        self.systems = []

    def create(self, **kwargs):
        self.last_system = kwargs["messages"][0]["content"]
        self.systems.append(self.last_system)
        return _FakeResp(self._content)


def test_rag_llm_track_available_with_injected_client_and_injects_evidence(rag_retriever):
    payload = '{"results": [{"id": 0, "sentiment": "负面", "aspects": ["抽卡"], "reason": "歪了"}]}'
    client = _FakeClient(payload)
    track = RagLLMTrack(client=client, retriever=rag_retriever)
    assert track.is_available()
    out = track.classify(["这次又歪了，保底白给"])
    assert out[0].sentiment == "负面"
    assert out[0].track == "rag_llm"
    assert "黑话" in client.last_system  # RAG 证据已注入系统提示


def test_rag_llm_track_uses_per_comment_evidence_groups(rag_retriever):
    payload = (
        '{"results": ['
        '{"id": 0, "sentiment": "负面", "aspects": ["抽卡"], "reason": "x"}'
        "]}"
    )
    client = _FakeClient(payload)
    track = RagLLMTrack(client=client, retriever=rag_retriever)
    out = track.classify(["这次又歪了", "剧情真好"])
    assert len(out) == 2
    assert len(client.systems) == 2
    assert any("歪了" in system for system in client.systems)
    assert any("黑话" not in system for system in client.systems)


def test_keyword_track_always_available_offline():
    track = KeywordTrack()
    assert track.is_available()
    out = track.classify(["抽卡又歪了，真烂"])
    assert out[0].sentiment == "负面"
    assert out[0].track == "keyword"


def test_router_from_environment_runs_offline():
    # 不注入 client：rag_llm/lora 多半不可用，应优雅退化到 keyword/蒸馏并跑通
    r = RouterAgent.from_environment()
    out = r.tag(["剧情很好", "抽卡歪了真烂"])
    assert len(out) == 2
    assert all(o.track in r.last_stats["ladder"] for o in out)
