"""finetune.dataset_formatter 单测：纯数据变换，不加载任何大模型/依赖。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.finetune import dataset_formatter as df


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_text": ["抽卡又歪了", "剧情很好", "短", "服务器崩了"],
            "llm_sentiment": ["负面", "正面", "负面", "狂喜"],  # 末条情感非法
            "llm_aspects": ["['抽卡']", "['剧情']", "['其他']", "['运营']"],
            "llm_reason": ["保底歪了体验差", "角色塑造到位", "x", ""],  # 第3条理由过短、第4条空
        }
    )


def test_parse_aspects_handles_str_list_and_invalid():
    assert df._parse_aspects("['抽卡']") == ["抽卡"]
    assert df._parse_aspects(["剧情", "天气"]) == ["剧情"]  # 过滤取值域外
    assert df._parse_aspects("") == []


def test_to_alpaca_records_filters_low_confidence():
    records = df.to_alpaca_records(_sample_df())
    # 仅前两条满足：合法情感 + 理由足够长 + 文本不过短
    assert len(records) == 2
    assert all(set(r) == {"instruction", "input", "output"} for r in records)
    out0 = json.loads(records[0]["output"])
    assert out0["sentiment"] == "负面"
    assert out0["aspects"] == ["抽卡"]


def test_to_alpaca_records_output_is_valid_json():
    for r in df.to_alpaca_records(_sample_df()):
        parsed = json.loads(r["output"])
        assert parsed["sentiment"] in {"正面", "中性", "负面"}


def test_format_dataset_writes_jsonl_and_info(tmp_path):
    src = tmp_path / "ai_analysis.csv"
    _sample_df().to_csv(src, index=False)
    res = df.format_dataset(src, tmp_path / "out")
    assert res["n_records"] == 2

    jsonl_path = res["jsonl"]
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["instruction"]

    info = json.loads(res["dataset_info"].read_text(encoding="utf-8"))
    assert df.DATASET_NAME in info
    assert info[df.DATASET_NAME]["file_name"] == "genshin_sentiment.jsonl"


def test_format_dataset_raises_when_no_high_confidence(tmp_path):
    bad = pd.DataFrame(
        {"clean_text": ["短"], "llm_sentiment": ["负面"], "llm_aspects": ["['其他']"],
         "llm_reason": [""]}
    )
    src = tmp_path / "bad.csv"
    bad.to_csv(src, index=False)
    with pytest.raises(RuntimeError):
        df.format_dataset(src, tmp_path / "out")


def test_load_labeled_requires_columns(tmp_path):
    src = tmp_path / "x.csv"
    pd.DataFrame({"foo": [1]}).to_csv(src, index=False)
    with pytest.raises(KeyError):
        df.load_labeled(src)


# ---- 训练/评估切分 ----


def test_split_records_preserves_all_and_no_overlap():
    df_big = pd.DataFrame(
        {
            "clean_text": [f"评论内容{i}" for i in range(10)],
            "llm_sentiment": (["负面", "正面"] * 5),
            "llm_aspects": ["['抽卡']"] * 10,
            "llm_reason": ["理由足够长用于通过过滤"] * 10,
        }
    )
    records = df.to_alpaca_records(df_big)
    train, ev = df.split_records(records, eval_ratio=0.2)
    assert len(train) + len(ev) == len(records)
    assert len(ev) >= 1
    train_inputs = {r["input"] for r in train}
    assert all(r["input"] not in train_inputs for r in ev)  # 无重叠


# ---- 第 3 步：评估与错例分析 ----


def test_evaluate_perfect_predictor():
    from src.finetune.evaluate import evaluate

    texts = ["剧情好", "抽卡歪了"]
    gold = ["正面", "负面"]
    rep = evaluate(lambda ts: ["正面", "负面"], texts, gold)
    assert rep.accuracy == 1.0
    assert rep.errors == []  # 全对，无错例
    # macro_f1 在固定三标签集上对缺席的「中性」会计 0，这是 sklearn 既定行为，不强求 1.0


def test_evaluate_finds_errors_and_flags_irony():
    from src.finetune.evaluate import evaluate

    texts = ["这运营真是好家伙", "剧情很好"]
    gold = ["负面", "正面"]
    rep = evaluate(lambda ts: ["正面", "正面"], texts, gold)  # 第一条判错
    assert rep.accuracy == 0.5
    assert len(rep.errors) == 1
    assert rep.errors[0]["irony"] is True  # “好家伙”被标为疑似反讽
    assert rep.predictions == ["正面", "正面"]  # 逐条预测被保留，供报告/CSV 使用


def test_build_markdown_report_has_all_sections():
    from src.finetune.evaluate import build_markdown_report, evaluate

    texts = ["剧情好", "抽卡歪了", "还行吧", "运营太烂"]
    gold = ["正面", "负面", "中性", "负面"]
    rep = evaluate(lambda ts: ["正面", "中性", "中性", "负面"], texts, gold)
    md = build_markdown_report(rep, predictor_name="lora", eval_set="eval.jsonl")
    assert "# LoRA 微调评估报告" in md
    assert "## 1. 总体指标" in md
    assert "## 2. 分类别指标" in md
    assert "## 3. 混淆矩阵" in md
    assert "## 4. 主要误差模式" in md
    for label in ("正面", "中性", "负面"):
        assert label in md


def test_build_markdown_report_with_baselines_adds_comparison():
    from src.finetune.evaluate import build_markdown_report, evaluate

    texts = ["剧情好", "抽卡歪了"]
    gold = ["正面", "负面"]
    main = evaluate(lambda ts: ["正面", "负面"], texts, gold)
    base = evaluate(lambda ts: ["中性", "中性"], texts, gold)
    md = build_markdown_report(
        main, predictor_name="lora", eval_set="eval.jsonl", baselines={"keyword": base}
    )
    assert "与基线对比" in md
    assert "负面召回" in md


def test_write_predictions_csv_roundtrip(tmp_path):
    import csv

    from src.finetune.evaluate import evaluate, write_predictions_csv

    texts = ["剧情好", "这运营真是好家伙"]
    gold = ["正面", "负面"]
    rep = evaluate(lambda ts: ["正面", "正面"], texts, gold)
    out = tmp_path / "preds.csv"
    write_predictions_csv(out, texts, gold, rep)
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 2
    assert rows[0]["是否正确"] == "✓"
    assert rows[1]["是否正确"] == "✗"
    assert rows[1]["疑似反讽"] == "是"  # “好家伙”命中反讽标记


def test_load_eval_set_roundtrip(tmp_path):
    from src.finetune.evaluate import load_eval_set

    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"instruction":"x","input":"抽卡歪了","output":"{\\"sentiment\\": \\"负面\\"}"}\n',
        encoding="utf-8",
    )
    texts, gold = load_eval_set(p)
    assert texts == ["抽卡歪了"]
    assert gold == ["负面"]


# ---- 自包含 QLoRA 训练脚本的数据编码（用假分词器，无需 GPU/transformers）----


class _FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "P:" + messages[-1]["content"] + "|"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_to_messages_shape():
    from src.finetune.train_lora import to_messages

    msgs = to_messages({"instruction": "sys", "input": "评论"})
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "评论"


def test_encode_masks_prompt_tokens():
    from src.finetune.train_lora import _encode

    rec = {"instruction": "sys", "input": "抽卡歪了", "output": "负面"}
    tok = _FakeTokenizer()
    enc = _encode(rec, tok, cutoff_len=1000)
    prompt_len = len("P:抽卡歪了|")
    # prompt 部分全部 -100，回答部分保留真实 token id
    assert enc["labels"][:prompt_len] == [-100] * prompt_len
    assert enc["labels"][prompt_len:] == enc["input_ids"][prompt_len:]
    assert len(enc["input_ids"]) == len(enc["attention_mask"]) == len(enc["labels"])


def test_load_records_reads_jsonl(tmp_path):
    from src.finetune.train_lora import load_records

    p = tmp_path / "train.jsonl"
    p.write_text(
        '{"instruction":"a","input":"b","output":"c"}\n\n'
        '{"instruction":"d","input":"e","output":"f"}\n',
        encoding="utf-8",
    )
    recs = load_records(p)
    assert len(recs) == 2
    assert recs[0]["input"] == "b"


# ---- 场景 B：自包含 OpenAI 兼容端点（纯标准库，无 GPU/无 fastapi，CI 直接跑）----


def test_build_chat_response_shape():
    from src.finetune.serve import build_chat_response

    resp = build_chat_response('{"sentiment": "负面"}', model="qwen2.5-7b-lora")
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "qwen2.5-7b-lora"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert resp["choices"][0]["message"]["content"] == '{"sentiment": "负面"}'
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert "usage" in resp


class _FakeGenerator:
    """假生成器：记录收到的消息并回显固定 JSON，免 GPU 测端点协议。"""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def generate_chat(self, messages, *, max_new_tokens=None, temperature=0.0):
        self.calls.append(messages)
        return '{"sentiment": "负面", "aspects": ["抽卡"]}'


class _ServerThread:
    """在后台线程起一个真实的 http.server 端点，用 urllib 打它，测完即关。"""

    def __init__(self, generator, *, api_key="EMPTY"):
        import threading

        from src.finetune.serve import build_server

        self.httpd = build_server(
            generator, host="127.0.0.1", port=0, served_model="qwen2.5-7b-lora", api_key=api_key
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path, *, body=None, headers=None):
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))


def test_endpoint_health_and_models():
    with _ServerThread(_FakeGenerator()) as s:
        code, body = s.request("GET", "/health")
        assert code == 200 and body["status"] == "ok"
        code, body = s.request("GET", "/v1/models")
        assert body["data"][0]["id"] == "qwen2.5-7b-lora"


def test_endpoint_chat_completions_roundtrip():
    gen = _FakeGenerator()
    with _ServerThread(gen) as s:
        code, body = s.request(
            "POST",
            "/v1/chat/completions",
            body={
                "model": "ignored-name",
                "messages": [
                    {"role": "system", "content": "你是舆情助手"},
                    {"role": "user", "content": "这次又歪了"},
                ],
                "temperature": 0,
            },
            headers={"Content-Type": "application/json"},
        )
    assert code == 200
    assert body["model"] == "qwen2.5-7b-lora"  # 端点强制对外模型名
    assert json.loads(body["choices"][0]["message"]["content"])["sentiment"] == "负面"
    assert gen.calls and gen.calls[0][-1]["content"] == "这次又歪了"


def test_endpoint_requires_api_key_when_set():
    payload = {"messages": [{"role": "user", "content": "x"}]}
    with _ServerThread(_FakeGenerator(), api_key="secret") as s:
        code, _ = s.request("POST", "/v1/chat/completions", body=payload)
        assert code == 401
        code, _ = s.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Authorization": "Bearer secret"},
        )
        assert code == 200


def test_endpoint_rejects_empty_messages():
    with _ServerThread(_FakeGenerator()) as s:
        code, body = s.request("POST", "/v1/chat/completions", body={"messages": []})
    assert code == 400
