"""text_pipeline 测试：规则清洗可直接断言；LLM 部分用 fake client 注入，不触网。"""

from __future__ import annotations

import pandas as pd
import pytest

from src import text_pipeline

# ---- 清洗：纯规则，可确定性断言 ----


def test_clean_text_strips_url_mention_reply_and_repeats():
    raw = "回复 @某人 ：哈哈哈哈哈这个太强了 https://b23.tv/abc @另一个人"
    out = text_pipeline.clean_text(raw)
    assert "http" not in out
    assert "@" not in out
    assert "回复" not in out
    assert "哈哈哈哈" not in out  # 重复字符被压缩
    assert "太强了" in out


@pytest.mark.parametrize("bad", [None, 123, float("nan")])
def test_clean_text_handles_non_str(bad):
    assert text_pipeline.clean_text(bad) == ""


def test_clean_corpus_dedupes_and_drops_short():
    texts = ["好评好评", "好评好评", "啊", "", "另一条评论"]
    out = text_pipeline.clean_corpus(texts)
    assert out == ["好评好评", "另一条评论"]  # 去重 + 丢弃过短/空


# ---- 归类：用 fake client 模拟 API 返回 ----


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class FakeClient:
    """记录调用次数，按预设 JSON 串依次返回，模拟 OpenAI 兼容客户端。"""

    def __init__(self, payloads: list[str]):
        self._payloads = payloads
        self.calls = 0
        self.chat = self  # 让 client.chat.completions.create 可链式访问
        self.completions = self

    def create(self, **kwargs):
        content = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return _FakeResp(content)


def test_classify_batch_aligns_and_validates():
    payload = (
        '{"results": ['
        '{"id": 0, "sentiment": "负面", "aspects": ["抽卡"], "reason": "歪了"},'
        '{"id": 1, "sentiment": "正面", "aspects": ["剧情"], "reason": "好"}'
        "]}"
    )
    client = FakeClient([payload])
    out = text_pipeline.classify_batch(["抽卡歪了", "剧情好"], client=client)
    assert out[0]["sentiment"] == "负面"
    assert out[0]["aspects"] == ["抽卡"]
    assert out[1]["sentiment"] == "正面"


def test_classify_batch_fills_missing_and_rejects_invalid_labels():
    # 只返回 id=0，且 sentiment 非法、aspect 不在取值域
    payload = '{"results": [{"id": 0, "sentiment": "狂喜", "aspects": ["天气"]}]}'
    client = FakeClient([payload])
    out = text_pipeline.classify_batch(["a", "b"], client=client)
    assert len(out) == 2
    assert out[0]["sentiment"] == "中性"  # 非法情感兜底
    assert out[0]["aspects"] == ["其他"]  # 非法方面兜底
    assert out[1]["sentiment"] == "中性"  # 缺失项兜底


def test_classify_with_llm_batches(monkeypatch):
    payload = '{"results": [{"id": 0, "sentiment": "中性", "aspects": ["其他"]}]}'
    client = FakeClient([payload])
    texts = [f"评论{i}" for i in range(5)]
    out = text_pipeline.classify_with_llm(texts, client=client, batch_size=2)
    assert len(out) == 5
    assert client.calls == 3  # 5 条 / batch=2 → 3 批


def test_analyze_comments_end_to_end_with_fake_client():
    payload = (
        '{"results": ['
        '{"id": 0, "sentiment": "负面", "aspects": ["运营"], "reason": "x"},'
        '{"id": 1, "sentiment": "正面", "aspects": ["社交与同人"], "reason": "y"}'
        "]}"
    )
    client = FakeClient([payload])
    df = pd.DataFrame({"Comment_Content": ["服务器又崩了", "二创真不错"]})
    out = text_pipeline.analyze_comments(df, client=client)
    assert list(out["llm_sentiment"]) == ["负面", "正面"]
    summary = text_pipeline.aspect_sentiment_summary(out)
    assert "运营" in summary.index


def test_analyze_comments_missing_column_raises():
    with pytest.raises(KeyError):
        text_pipeline.analyze_comments(pd.DataFrame({"x": [1]}), client=FakeClient(["{}"]))
