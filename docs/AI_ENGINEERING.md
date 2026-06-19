# AI / LLM 应用工程技术清单

> 本文枚举项目中**实际落地**的 AI / 大模型应用工程技术，并标注每项的代码位置。
> 项目不是多智能体系统，因此这里如实归类为「LLM 应用 / Agent 工程」技术，不夸大为「多 Agent」。

## 1. 主流模型 API 接入（Provider-agnostic）

- 通过 **OpenAI 兼容协议**接入 DeepSeek：复用官方 `openai` SDK，仅替换 `base_url` 与 `api_key`，即可在任意 OpenAI 兼容服务间一键切换（`src/llm_client.py`、`src/config.py`）。
- 模型、base_url、batch、温度等全部走环境变量/`.env`，**密钥从不写进代码**，`.env` 入 `.gitignore`（`src/config.py` 在读取配置前 `load_dotenv`）。
- 运行时用 `/models` 接口核实模型 id 与官方一致（默认 `deepseek-v4-pro`）。

## 2. 结构化输出（Structured Output / JSON Mode）

- 调用时设 `response_format={"type": "json_object"}` 强制模型输出 JSON，下游直接解析为 dict，无需脆弱的正则抽取（`llm_client.chat_json`）。
- Prompt 里**显式给出 JSON schema 形状**（字段名、数组结构、id 对应关系），约束模型产出可解析、可对齐的结果（`text_pipeline._build_classify_prompt`）。

## 3. Prompt 工程

- **system / user 角色分离**：system 固定角色与输出契约，user 携带具体数据（分类、总结、爆点总结各有专用 system prompt）。
- **受限标签空间（constrained generation）**：把情感、方面的合法取值域写进 prompt，把开放生成收敛成可枚举的分类问题（`config.LLM_SENTIMENT_LABELS` / `LLM_ASPECT_LABELS`）。
- **编号批处理格式**：一次给多条带序号的评论，要求按 id 一一对应返回，便于对齐与校验。

## 4. 输出校验与对齐（Guardrails）

- 返回结果**按输入长度对齐**：缺失项补默认、id 错位重排（`text_pipeline._normalize_result`）。
- **非法标签兜底**：模型若返回取值域外的情感/方面，回退到「中性 / 其他」，绝不把脏标签放进下游。

## 5. 鲁棒性与降级

- **指数退避重试**：网络抖动、限流、偶发非法 JSON 统一重试，超限才抛出（`llm_client.chat_json`）。
- **单批失败隔离**：批量打标时某一批失败只兜底该批为中性/其他，不让整个 40 万条流程崩溃（`text_pipeline.classify_with_llm`）。
- **优雅降级**：抓取被风控/无网时回退演示数据；词云缺 wordcloud/字体时降级为 HTML 关键词云；API 未配置时明确报错而非静默退化为关键词规则。

## 6. 成本控制

- **两段式分层**：先用零成本的规则清洗（去链接/@/重复字符/去重），再把干净文本送贵的 LLM——便宜的活不占用 token（`text_pipeline.clean_text` → `analyze_comments`）。
- **批处理**（`LLM_BATCH_SIZE`）摊薄每条开销；**抽样**（各脚本 `--sample`）在迭代期控制调用量。

## 7. 数据驱动的模型选型

- 用关键词基线在 40.8 万条真实评论上实测**覆盖率仅 18.2%**，量化证明关键词规则的盲区，以此驱动「引入 LLM 语义打标」的决策——而非凭直觉（`aspect_sentiment.agreement_with_cluster`、`scripts/keyword_vs_ai.py`）。

## 8. LLM 作为标注者 + 知识蒸馏（Knowledge Distillation）

- **教师**：LLM 给小样本打高质量情感标签；**学生**：在 (文本→LLM标签) 上训练字符级 TF-IDF + 逻辑回归的轻量分类器（`src/sentiment_train.py`）。
- 学生模型离线、免 API、毫秒级，可低成本推理全量；评估用留出集准确率 / 宏 F1 / **与教师标签一致率**。

## 9. LLM 作为分析者（Map → Reduce 式语义聚合）

- 把一批离散评论/标题**归纳**成结构化结论：舆情总结（议题/占比/代表性原声/运营建议）、爆点选题总结（套路/关键词/建议）（`text_pipeline.summarize_opinions` / `summarize_hits`）。
- AI 提炼的关键词反哺词云：对其在词频里**加权放大**，让词云突出语义议题词而非高频口水词（`src/wordcloud_gen.py`）。

## 10. 可复用与可验证

- AI 能力封装成可复用模块（`llm_client` / `text_pipeline` / `sentiment_train` / `wordcloud_gen`），CLI 与 Streamlit 看板共用。
- **测试不触网**：用 fake client / fake session 注入模拟 API 返回，合成数据训练学生模型，CI 无需密钥或网络即可全部跑通。

---

| 技术 | 主要代码位置 |
| --- | --- |
| OpenAI 兼容接入 / 密钥管理 | `src/llm_client.py`, `src/config.py` |
| 结构化 JSON 输出 + Prompt 工程 | `src/text_pipeline.py` |
| 输出校验/对齐/兜底 | `text_pipeline._normalize_result` |
| 重试 / 批失败隔离 / 降级 | `llm_client.chat_json`, `text_pipeline.classify_with_llm` |
| 知识蒸馏（LLM 标注→轻量分类器） | `src/sentiment_train.py`, `scripts/train_sentiment.py` |
| 语义总结 / 关键词加权词云 | `text_pipeline.summarize_*`, `src/wordcloud_gen.py` |
| 数据驱动选型（覆盖率论证） | `scripts/keyword_vs_ai.py` |
