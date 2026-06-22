# 云端 LoRA 微调 + 自建推理端点（场景 B）

把训练放云端、推理也留云端，本地只负责发起调用——既绕开本机无 GPU、`bitsandbytes`
在 Windows 跑不通的死结，又不必把 7B 基座下回本地慢推。本地通过现有的 OpenAI 兼容
协议直连云端服务，路由的 `lora_server` 轨道会自动接入。

```
云端 GPU（AutoDL/RunPod）           本地（笔记本，无 GPU）
┌─────────────────────────┐        ┌──────────────────────────┐
│ git clone + 下载数据集   │        │ 智能路由 RouterAgent      │
│ train_lora.sh  → 适配器  │        │  ├ keyword / distilled    │
│ serve_lora.sh  → vLLM    │  HTTP  │  ├ rag_llm（DeepSeek）     │
│ cloudflared    → 公网URL │◄───────│  └ lora_server（你的Qwen）│
└─────────────────────────┘        └──────────────────────────┘
```

## 一、云端：微调

在 AutoDL / RunPod 选一台带 GPU 的实例（RTX 4090 24G 足够 7B QLoRA），用预装
PyTorch + CUDA 的镜像。

```bash
git clone <your-repo-url> && cd genshin-sentiment-analysis
# 数据集是公开 Kaggle CSV，直接下载放进 data/，无需从本地上传
uv sync --extra finetune            # 注意：用镜像自带的 CUDA torch，必要时给 uv 配 CUDA 源
uv run python scripts/build_finetune_dataset.py --sample 1800   # 分层抽样 + 标注 + 切分，生成训练/留出 JSONL
bash src/finetune/train_lora.sh     # QLoRA 微调，适配器产物在 outputs/finetune/qwen2.5-7b-lora
```

> **国内服务器拉不到 HuggingFace**（`[Errno 101] Network is unreachable`）：默认脚本已把
> `HF_ENDPOINT` 指向镜像 `hf-mirror.com`。建议训练前先把基座权重预下载到本地缓存，避免
> 训练跑到一半才因网络中断：
>
> ```bash
> pip install -U "huggingface_hub[cli]"
> export HF_ENDPOINT=https://hf-mirror.com
> huggingface-cli download Qwen/Qwen2.5-7B-Instruct
> # 想更快可：pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
> ```
>
> AutoDL 用户也可改用平台自带加速：`source /etc/network_turbo` 后直连 huggingface.co。

> **依赖更省心**：训练脚本自包含（`src/finetune/train_lora.py`，只用 transformers + peft +
> bitsandbytes），不引入 LLaMA-Factory 那套带原生库（torchaudio/torchvision/gradio）的依赖树，
> 因此不会再出现 `libtorchaudio_sox.so: cannot open shared object file` 之类的崩溃。flash-attn
> 也不是必需：默认用 torch 自带的 sdpa，装了再 `FLASH_ATTN=fa2 bash src/finetune/train_lora.sh`。

> 适配器只有几十 MB，可以下回本地存档；但**不要在本地无 GPU 的机器上加载 7B 推理**，
> 那会退化成 CPU 慢推。推理交给下一步的云端服务。

## 二、云端：起推理服务并暴露公网

```bash
pip install "vllm>=0.5"
bash scripts/serve_lora.sh          # vLLM 把基座 + LoRA 起成 OpenAI 兼容端点，默认 :8000
# 另开一个终端，把端口映射出公网（任选其一）
cloudflared tunnel --url http://localhost:8000
# 或 ngrok http 8000
```

记下隧道给出的 `https://xxxx.trycloudflare.com` 地址。

## 三、本地：填 .env，路由自动接入

在本地 `.env` 填上云端地址（其余配置已就绪）：

```bash
LORA_SERVER_BASE_URL=https://xxxx.trycloudflare.com/v1
LORA_SERVER_MODEL=qwen2.5-7b-lora
LORA_SERVER_API_KEY=EMPTY            # 若 serve 时设了 --api-key，这里填同一个
```

之后无需改任何代码：

```python
from src.agents import RouterAgent
router = RouterAgent.from_environment()       # 自动探测到 lora_server 轨道并纳入阶梯
print(router.last_stats["ladder"])            # 含 'lora_server'
results = router.tag(["这次又歪了，保底白给"]) # 难句自动走你的云端 Qwen，而非 DeepSeek
```

看板「作品打标」页的「🧭 智能路由」也会自动多出 `lora_server` 轨道。

## 工作原理（为什么本地零改动）

`config.LORA_SERVER_BASE_URL` 一旦非空，`llm_client.get_served_client()` 就会构造一个
`ServedLLMClient`：它包住标准 OpenAI 客户端，并在每次请求时把 `model` 强制改写成
`LORA_SERVER_MODEL`。因此 `text_pipeline` 那套打标流程一个字都不用改，就能把请求发到
你的云端端点。`CloudLoRATrack` 用的就是这个客户端，成本档位（2）低于付费的 DeepSeek（3），
所以路由对难句会**优先选你自建的 Qwen**，把 DeepSeek 留作更高一档的兜底。

## 成本与取舍

- **训练**：单次几十块封顶，按小时计费，跑完即停。
- **推理**：云端常驻才计费；不想一直开，就让本地路由平时只用「关键词 + 蒸馏 + DeepSeek」，
  需要时再开云端点亮 `lora_server`——路由会自适应，开/关都不报错。
- **隧道**：`cloudflared`/`ngrok` 适合开发与演示；生产应换成带鉴权的托管推理端点。
