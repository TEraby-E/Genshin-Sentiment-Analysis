"""llm_client 测试：密钥缺失报错、chat_json 解析与重试，全部用 fake client，不触网。"""

from __future__ import annotations

import pytest

from src import config, llm_client


def test_get_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv(config.LLM_API_KEY_ENV, raising=False)
    with pytest.raises(llm_client.LLMNotConfiguredError):
        llm_client.get_api_key()


class _Msg:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, c):
        self.message = _Msg(c)


class _Resp:
    def __init__(self, c):
        self.choices = [_Choice(c)]


class _FlakyClient:
    """前 n_fail 次抛错，之后返回合法 JSON，用于验证重试逻辑。"""

    def __init__(self, n_fail: int, content: str):
        self.n_fail = n_fail
        self.content = content
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise RuntimeError("模拟网络抖动")
        return _Resp(self.content)


def test_chat_json_parses(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    client = _FlakyClient(0, '{"ok": true}')
    out = llm_client.chat_json("sys", "user", client=client)
    assert out == {"ok": True}


def test_chat_json_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    client = _FlakyClient(2, '{"ok": 1}')
    out = llm_client.chat_json("sys", "user", client=client, max_retries=3)
    assert out == {"ok": 1}
    assert client.calls == 3


def test_chat_json_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    client = _FlakyClient(99, "{}")
    with pytest.raises(RuntimeError):
        llm_client.chat_json("sys", "user", client=client, max_retries=2)
    assert client.calls == 2
