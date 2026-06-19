"""wordcloud_gen 测试：AI 关键词加权是纯函数可确定性断言；分词需 jieba（report extra）则跳过。"""

from __future__ import annotations

from collections import Counter

import pytest

from src import wordcloud_gen


def test_boost_existing_keyword_is_multiplied():
    freq = Counter({"剧情": 2, "哈基米": 10})
    out = wordcloud_gen.boost_with_ai_keywords(freq, ["剧情"])
    assert out["剧情"] == 2 * wordcloud_gen.AI_KEYWORD_BOOST
    assert out["哈基米"] == 10  # 未点名的词不变


def test_boost_missing_keyword_added_at_top_level():
    freq = Counter({"哈基米": 10})
    out = wordcloud_gen.boost_with_ai_keywords(freq, ["兑换码"])
    assert out["兑换码"] == 10  # 补到最高词频量级，保证显眼


def test_boost_strips_emoji_and_short_keywords():
    freq: Counter[str] = Counter({"剧情": 5})
    out = wordcloud_gen.boost_with_ai_keywords(freq, ["😡", "a", "新地图😡"])
    assert "😡" not in out
    assert "a" not in out  # 长度 < 2 被丢弃
    assert "新地图" in out  # emoji 被清洗后保留中文部分


def test_tokenize_filters_stopwords_and_short():
    pytest.importorskip("jieba")
    freq = wordcloud_gen.tokenize(["这个 剧情 真的 很 好 剧情"])
    assert "的" not in freq and "很" not in freq  # 停用词被过滤
    assert freq.get("剧情", 0) >= 1
