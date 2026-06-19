"""可复用的词云生成核心：分词、AI 关键词加权、渲染。

被 scripts/make_wordcloud.py（保存 PNG）与 dashboard.py（内存渲染后 st.image 展示）共用，
避免两处各写一份分词/停用词/加权逻辑。jieba 与 wordcloud 属于 report extra，
在此延迟导入，使无该依赖的环境（如 CI）仍能 import 本模块并测试纯函数逻辑。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Windows / Linux 常见中文字体候选；wordcloud 需显式字体路径才能渲染中文
FONT_CANDIDATES = [
    "C:/WINDOWS/fonts/msyh.ttc",
    "C:/WINDOWS/fonts/simhei.ttf",
    "C:/WINDOWS/fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

STOPWORDS: frozenset[str] = frozenset({
    "的", "了", "是", "我", "你", "他", "她", "它", "这", "那", "在", "就",
    "都", "也", "很", "还", "不", "和", "与", "啊", "吧", "呢", "吗", "啦",
    "把", "被", "对", "上", "下", "中", "有", "没", "没有", "什么",
    "一个", "这个", "那个", "可以", "因为", "所以", "但是", "而且", "就是",
    "真的", "感觉", "觉得", "知道", "现在", "已经", "还是", "自己", "之后",
    "这样", "那样", "怎么", "为什么", "一下", "一直", "一些", "比较", "非常",
    "我们", "你们", "他们", "她们", "大家", "这么", "那么", "这些", "那些",
    "不会", "不是", "是不是", "只是", "只有", "还有", "其他", "其实", "以为",
    "出来", "起来", "时候", "之前", "应该", "希望", "如果", "虽然",
    "不要", "不能", "终于", "今天", "最后", "直接", "结果", "支持",
    "可能", "需要", "肯定", "估计", "反正", "确实", "好像", "似乎",
    "一次", "一定", "想要", "看看", "啊啊啊", "好好", "而是", "真是",
    "这次", "为了", "然后", "几个", "或者", "哪些", "这是", "怎么办",
})

WORD_LEN_MIN = 2
TEXT_JOIN_SEP = "\n"
NON_CJK_LATIN_PATTERN = re.compile(r"[^一-龥a-zA-Z\n]")
AI_KEYWORD_BOOST = 3.0


def resolve_font_path() -> str:
    """返回第一个真实存在的中文字体文件路径。"""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        f"未找到可用的中文字体文件，已尝试: {FONT_CANDIDATES}。请补充本机字体路径。"
    )


def tokenize(texts: list[str]) -> Counter[str]:
    """对一批文本整体分词并统计词频，过滤停用词/纯数字/过短词。"""
    valid = [t for t in texts if isinstance(t, str)]
    if not valid:
        return Counter()
    import jieba

    merged = NON_CJK_LATIN_PATTERN.sub(" ", TEXT_JOIN_SEP.join(valid))
    counter: Counter[str] = Counter()
    words = (w.strip() for w in jieba.cut(merged))
    counter.update(
        w for w in words if len(w) >= WORD_LEN_MIN and w not in STOPWORDS and not w.isdigit()
    )
    return counter


def boost_with_ai_keywords(freq: Counter[str], keywords: list[str]) -> dict[str, float]:
    """用 AI 提炼的语义关键词加权：已有词放大 BOOST 倍，缺失词补到最高词频量级。

    关键词先清洗掉 emoji/标点，只保留中英文字符；纯函数，无外部依赖，便于测试。
    """
    weighted: dict[str, float] = dict(freq)
    top = max(freq.values()) if freq else 1.0
    for kw in keywords:
        clean = NON_CJK_LATIN_PATTERN.sub("", str(kw)).strip()
        if len(clean) < WORD_LEN_MIN:
            continue
        weighted[clean] = weighted[clean] * AI_KEYWORD_BOOST if clean in weighted else top
    return weighted


def render_wordcloud(weights: dict[str, float], *, colormap: str = "Reds", font_path=None) -> Any:
    """根据词频/权重渲染词云，返回 PIL.Image（不落盘），供看板 st.image 或脚本保存。"""
    if not weights:
        raise ValueError("词频/权重为空，无法生成词云")
    from wordcloud import WordCloud

    wc = WordCloud(
        font_path=font_path or resolve_font_path(),
        width=1600,
        height=900,
        mode="RGBA",
        background_color=None,
        colormap=colormap,
        max_words=120,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(dict(weights))
    return wc.to_image()


def wordcloud_from_ai(
    texts: list[str], keywords: list[str], *, colormap: str = "Reds", font_path=None
) -> Any:
    """端到端：分词 + AI 关键词加权 + 渲染，返回 PIL.Image。供看板一键调用。"""
    weighted = boost_with_ai_keywords(tokenize(texts), keywords)
    return render_wordcloud(weighted, colormap=colormap, font_path=font_path)
