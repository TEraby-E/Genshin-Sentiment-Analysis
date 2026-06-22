#!/usr/bin/env bash
# =============================================================================
# 把训练好的 LoRA 适配器导出到 git（在云端 AutoDL 上运行），只挑最终适配器文件，
# 自动排除 checkpoint-*/ 等几个 GB 的训练中间产物，并校验单文件未超 GitHub 100MB 限制。
#
# 用法（在云端实例、仓库根目录）：
#   bash scripts/export_adapter.sh                       # 暂存 + 提交，最后提示你 push
#   ADAPTER_DIR=outputs/finetune/qwen2.5-7b-lora bash scripts/export_adapter.sh
#   PUSH=1 bash scripts/export_adapter.sh                # 提交后直接 git push origin <当前分支>
#
# 前置：已在 AutoDL 上配置好 GitHub 凭据（PAT/SSH），且仓库与 origin 同步（先 git pull）。
# =============================================================================
set -euo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-outputs/finetune/qwen2.5-7b-lora}"

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "找不到适配器目录：$ADAPTER_DIR（先在本机跑完 train_lora.sh）" >&2
    exit 1
fi
if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo "目录里没有 adapter_model.safetensors，训练可能未正常保存：$ADAPTER_DIR" >&2
    exit 1
fi

echo "适配器目录顶层文件（仅这些会被导出，checkpoint-*/ 子目录会被跳过）："
ls -lh "$ADAPTER_DIR" | grep -v '^d' || true

# 校验：GitHub 单文件硬上限 100MB；超了就别走普通 git，提示改用 Release / LFS。
limit=$((100 * 1024 * 1024))
oversized=0
while IFS= read -r -d '' f; do
    size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
    if [ "$size" -ge "$limit" ]; then
        echo "⚠️  超过 100MB，普通 git 推不上去：$f（$((size / 1024 / 1024))MB）" >&2
        oversized=1
    fi
done < <(find "$ADAPTER_DIR" -maxdepth 1 -type f -print0)

if [ "$oversized" = "1" ]; then
    echo "" >&2
    echo "该文件过大。两种办法（任选其一）：" >&2
    echo "  1) GitHub Release： gh release create lora-v1 \"$ADAPTER_DIR/adapter_model.safetensors\"" >&2
    echo "  2) Git LFS：       git lfs install && git lfs track '*.safetensors' && 重新提交" >&2
    exit 2
fi

# 只暂存顶层文件（-maxdepth 1），天然排除 checkpoint-*/ 等子目录；-f 以防被 .gitignore 误挡。
find "$ADAPTER_DIR" -maxdepth 1 -type f -print0 | xargs -0 git add -f
git add .gitignore 2>/dev/null || true

if git diff --cached --quiet; then
    echo "没有变更需要提交（适配器可能已在版本库中）。"
    exit 0
fi

git commit -m "导出微调 LoRA 适配器权重（$ADAPTER_DIR）"
echo "✅ 已提交。"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "${PUSH:-0}" = "1" ]; then
    git push origin "$branch"
    echo "✅ 已推送到 origin/$branch。"
else
    echo "下一步推送： git push origin $branch"
fi
