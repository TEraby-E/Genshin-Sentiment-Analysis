# AI / LLM 应用工程技术清单

> 本文枚举项目中**实际落地**的 AI / 大模型应用工程技术，并标注每项的代码位置。
> 项目不是多智能体系统，因此这里如实归类为「LLM 应用 / Agent 工程」技术，不夸大为「多 Agent」。

## 1. 主流模型 API 接入（Provider-agnostic）

- 通过 **OpenAI 兼容协议**接入 DeepSeek：复用官方 `openai` SDK，仅替换 `base_url` 与 `api_key`，即可在任意 OpenAI 兼容服务间一键切换（`src/llm_client.py`、`src/config.py`）。
- 模型、base_url、batch、温度等全部走环境变量/`.env`，**密钥从不写进代码**，`.env` 入 `.gitignore`（`src/config.py` 在读取配置前 `load_dotenv`）。
- 默认模型名走 `LLM_MODEL` 环境变量配置；未配置时使用项目默认值，便于在 DeepSeek 或其他 OpenAI 兼容服务间切换。

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
- **优雅降级**：缺少 API、GPU、模型文件或 RAG 语料时，对应轨道会被跳过或回退到离线能力；API 未配置时给出明确报错而非静默伪装成高阶语义能力。

## 6. 成本控制

- **两段式分层**：先用零成本的规则清洗（去链接/@/重复字符/去重），再把干净文本送贵的 LLM——便宜的活不占用 token（`text_pipeline.clean_text` → `analyze_comments`）。
- **批处理**（`LLM_BATCH_SIZE`）摊薄每条开销；**抽样**（各脚本 `--sample`）在迭代期控制调用量。

## 7. 数据驱动的模型选型

- 用关键词基线在评论样本上计算方面标签覆盖率，量化证明关键词规则的盲区，以此驱动「引入 LLM 语义打标」的决策——而非凭直觉（`aspect_sentiment.agreement_with_cluster`）。

## 8. LLM 作为标注者 + 知识蒸馏（Knowledge Distillation）

- **教师**：LLM 给小样本打高质量情感标签；**学生**：在 (文本→LLM标签) 上训练字符级 TF-IDF + 逻辑回归的轻量分类器（`src/sentiment_train.py`）。
- 学生模型离线、免 API、毫秒级，可低成本推理全量；评估用留出集准确率 / 宏 F1 / **与教师标签一致率**。

## 9. LLM 作为分析者（Map → Reduce 式语义聚合）

- 把一批离散评论/标题**归纳**成结构化结论：舆情总结（议题/占比/代表性原声/运营建议）、爆点选题总结（套路/关键词/建议）（`text_pipeline.summarize_opinions` / `summarize_hits`）。

## 10. 可复用与可验证

- AI 能力封装成可复用模块（`llm_client` / `text_pipeline` / `sentiment_train` / `agents`），CLI 与 Streamlit 看板共用。
- **测试不触网**：用 fake client / fake session 注入模拟 API 返回，合成数据训练学生模型，CI 无需密钥或网络即可全部跑通。

## 11. 检索增强生成（RAG）：领域知识接地，对抗黑话误判

- **问题驱动**：玩家评论充斥社区黑话与时效性梗（「歪了」「保底」「深渊螺旋」「圣遗物词条」），通用模型缺乏社区内生语义，易按字面误判——这是一个典型的「知识接地（grounding）」而非「模型能力」问题，故用 RAG 而非换更大模型来解。
- **轻量本地向量库**：默认纯 numpy 余弦的内存库（零外部基建、可离线、CI 友好），可选切换 ChromaDB 本地持久化模式（`src/rag/vector_store.py`）。
- **分层嵌入**：默认确定性特征哈希嵌入（无模型/无外网/无 GPU），可选 `sentence-transformers` 语义嵌入（`src/rag/embeddings.py`）；嵌入函数抽象成最小协议，便于测试注入「假嵌入」。
- **混合检索（Hybrid Retrieval）**：稠密向量召回 + 稀疏 BM25 召回，min-max 归一化后加权融合（`alpha` 调偏重），兼顾近义梗与低频专有词（`src/rag/retriever.py`）。
- **异步语料构建**：把评论聚类关键词、官方动态正文、高赞 UGC 评论切块并发嵌入，构建「梗 & 设定词典」（`src/rag/ingestion.py`）。
- **无侵入注入**：`analyze_comments(..., retriever=)` 在送 LLM 前拦截评论中的黑话、检索释义并注入 system prompt；`RagLLMTrack` 会按单条评论构造 RAG 上下文，再把上下文相同的评论分组批量调用，避免整批共享无关证据；不传 `retriever` 时与原链路行为完全一致（`text_pipeline.detect_jargon` / `build_jargon_context`）。

## 12. 本地 LLM LoRA 微调（QLoRA）

- **与蒸馏并行的「重」轨道**：把 LLM 标注的高置信数据进一步微调进轻量本地大模型（Qwen2.5-7B），得到比 TF-IDF 更懂语义/黑话、仍离线免 API 的分类器（`src/sentiment_train.LocalLLMClassifier`）。
- **严格数据契约**：从已标注数据筛高置信样本（合法标签 + 非空依据 + 最小长度），转 alpaca JSONL 并生成 `dataset_info.json`（`src/finetune/dataset_formatter.py`）。
- **自包含高效训练**：自带最小化 QLoRA 训练脚本（`src/finetune/train_lora.py`，仅用 transformers + peft + bitsandbytes，不依赖 LLaMA-Factory），集成 4-bit 量化、梯度检查点、梯度累积、分页 8-bit 优化器；注意力默认用 torch 自带的 sdpa，装了 flash-attn 才用 FlashAttention-2。`src/finetune/train_lora.sh` 是便捷封装；配套 `scripts/build_finetune_dataset.py` 做分层抽样标注、`scripts/eval_lora.py` 做留出集评估与反讽错例分析，形成评估-迭代闭环。
- **延迟导入与优雅缺省**：`torch`/`transformers`/`peft` 等重依赖延迟到 `load()` 时导入，`deps_available()` / `is_ready()` 守卫；依赖或适配器缺失时看板给出明确引导而非静默失败——CI 无 GPU 也不受影响。

## 13. 智能路由 / 多轨道编排 Agent（Router + 检索-推理-校验三角）

- **路由即算力分配器**：把四条打标能力（关键词 / 蒸馏 / 本地 LoRA / RAG-LLM）统一成可调度的「轨道」（`src/agents/tracks.py`），`RouterAgent` 先用零成本的离线难度画像（黑话数 / 反讽标记 / 长度）判断该投多少算力——容易的评论走便宜轨道，难句直接起步于语义轨道（`src/agents/router.py`）。
- **检索-推理-校验三角（compute allocation strategy）**：进入语义轨道的评论触发「检索证据（RAG）→ LLM 推理 → critic 校验」；校验不通过则沿成本阶梯升一档重判（keyword→distilled→lora→rag_llm），默认允许升到当前环境可用的最高档，把贵算力**只花在 critic 认为值得的样本上**。
- **critic 双档**：默认 `HeuristicVerifier`（词典极性冲突检测 + 完整性，零成本、确定性），可选 `LLMVerifier`（LLM 评审员，准但有成本）（`src/agents/verifier.py`）；硬冲突（字面强烈一极、判定相反）直接打回并给出纠正。
- **环境自适应 + 批量分组**：不可用轨道（无 API / 无 GPU / 无模型）被自动过滤，离线时优雅退化到关键词 / 蒸馏；同轨道评论分组批量调用以摊薄 token。整套逻辑用 fake 轨道 + fake client 全覆盖，CI 无任何外部资源即可验证路由/升档/校验。

---

| 技术 | 主要代码位置 |
| --- | --- |
| OpenAI 兼容接入 / 密钥管理 | `src/llm_client.py`, `src/config.py` |
| 结构化 JSON 输出 + Prompt 工程 | `src/text_pipeline.py` |
| 输出校验/对齐/兜底 | `text_pipeline._normalize_result` |
| 重试 / 批失败隔离 / 降级 | `llm_client.chat_json`, `text_pipeline.classify_with_llm` |
| 知识蒸馏（LLM 标注→轻量分类器） | `src/sentiment_train.py`, `scripts/train_sentiment.py` |
| 检索增强（RAG）：向量库 / 嵌入 / 混合检索 / 异步入库 | `src/rag/`（`vector_store.py`, `embeddings.py`, `retriever.py`, `ingestion.py`） |
| RAG 黑话拦截与上下文注入 | `text_pipeline.detect_jargon` / `build_jargon_context` |
| 本地 LoRA 微调（QLoRA）+ 数据集构建/评估 | `src/finetune/`（`dataset_formatter.py`, `train_lora.py`, `evaluate.py`, `train_lora.sh`）, `scripts/build_finetune_dataset.py`, `scripts/eval_lora.py` |
| 智能路由 / 多轨道编排（Router + 检索-推理-校验三角 + 成本阶梯升档） | `src/agents/`（`router.py`, `tracks.py`, `verifier.py`） |
