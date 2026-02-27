# Export Evolved Assets for Hugging Face

本文档说明需要上传到 Hugging Face 的文件，以便用户下载到 `evolved/` 目录运行 GEO。

## 上传文件清单

### 1. Critic 模型 (evolved/critic/)

| 本地路径 (outputs) | HF 路径 | 说明 |
|-------------------|---------|------|
| `outputs/ea_training/geobench/critic/value_head.bin` | `evolved/critic/value_head.bin` | Critic 回归头权重 |
| `outputs/ea_training/geobench/critic/lora_adapter/adapter_model.safetensors` | `evolved/critic/lora_adapter/adapter_model.safetensors` | LoRA 适配器权重 |
| `outputs/ea_training/geobench/critic/lora_adapter/adapter_config.json` | `evolved/critic/lora_adapter/adapter_config.json` | LoRA 配置 |

### 2. 策略库 (evolved/archive/)

| 本地路径 (outputs) | HF 路径 | 说明 |
|-------------------|---------|------|
| *(需转换)* `outputs/ea_training/geobench/checkpoints/final/archive_final.json` | `evolved/archive/strategies.json` | 进化策略（需用脚本转换格式） |

---

## 策略 JSON 格式说明

`archive_final.json` 是 EA 训练输出的原始格式（genotype 结构），`run_geo` 需要的是带 `short_prompt` 和 `full_prompt` 的格式。

运行转换脚本：
```bash
python scripts/archive_to_strategies.py
```

输出 `evolved/archive/strategies.json` 结构示例：
```json
{
  "strategies": [
    {
      "genotype_id": "5d3961d32edf0ce5",
      "strategy_type": "authoritative",
      "short_prompt": "Strategy: ...\nConstraints: ...",
      "full_prompt": "## Objective\n...\n## Constraints\n...",
      "scores": {"total_score": 0.85}
    }
  ]
}
```

---

## HF 仓库目录结构

上传后的 Hugging Face 仓库建议结构：

```
<repo>/
├── evolved/
│   ├── critic/
│   │   ├── value_head.bin
│   │   └── lora_adapter/
│   │       ├── adapter_model.safetensors
│   │       └── adapter_config.json
│   └── archive/
│       └── strategies.json
```

---

## 打包脚本（上传前准备）

一键准备 `evolved/` 目录（转换 + 复制）：

```bash
python scripts/prep_evolved_for_hf.py
```

或手动执行：

```bash
# 1. 转换 archive 为 strategies.json
python scripts/archive_to_strategies.py

# 2. 复制 critic 文件到 evolved/
# (Windows) mkdir evolved\critic evolved\archive
# (Linux)   mkdir -p evolved/critic evolved/archive
cp outputs/ea_training/geobench/critic/value_head.bin evolved/critic/
cp -r outputs/ea_training/geobench/critic/lora_adapter evolved/critic/
# strategies.json 已由上一步生成到 evolved/archive/
```

## 用户下载到本地

用户可使用 `huggingface_hub` 下载：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="your-org/agentic-geo-evolved", local_dir=".", allow_patterns="evolved/*")
```

或 `huggingface-cli`：
```bash
huggingface-cli download your-org/agentic-geo-evolved evolved/* --local-dir .
```
