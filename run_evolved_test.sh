#!/bin/bash

# ============================================================
# GEO System - Evolved Strategies Test Script
# ============================================================
# This script will:
# 1. Start vLLM server (Qwen2.5-32B) on GPU 0 for content generation
# 2. Load evolved Critic model on GPU 1 for strategy selection
# 3. Run GEO experiments using evolved strategies
# ============================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# Configuration
# ============================================

# GPU Assignment
VLLM_GPU=0          # GPU for vLLM (Qwen2.5-32B generation engine)
CRITIC_GPU=1        # GPU for Critic model (strategy selection)

# vLLM Configuration
MODEL_PATH="${VLLM_MODEL_PATH:-/root/autodl-fs/Qwen/Qwen2.5-32B-Instruct}"
MODEL_NAME="qwen-32b"
VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
GPU_MEMORY_UTIL=0.90
TENSOR_PARALLEL_SIZE=1
SWAP_SPACE=32
VLLM_WAIT_TIMEOUT="${VLLM_WAIT_TIMEOUT:-1800}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---disable-frontend-multiprocessing --enforce-eager --uvicorn-log-level info}"

# Evolved Critic Configuration
# Base model for Critic structure (will auto-detect)
# EVOLVED_BASE_MODEL="/autodl-fs/data/modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"

# Pretrained backbone weights (from surrogate training)
EVOLVED_PRETRAINED_BACKBONE="${EVOLVED_PRETRAINED_BACKBONE:-surrogate_full1/stage1/checkpoint-610/exported_bin/pytorch_model.bin}"

# Trained value head from agentic_geo
EVOLVED_VALUE_HEAD="${EVOLVED_VALUE_HEAD:-outputs/ea_training/critic/value_head.bin}"

# Trained LoRA adapter (will auto-detect if exists)
# EVOLVED_LORA_ADAPTER="outputs/ea_training/critic/lora_adapter"

# Evolved strategies JSON file
EVOLVED_STRATEGIES="${EVOLVED_STRATEGIES:-outputs/ea_training/final_strategies_with_prompts.json}"

# Critic reward scale (should match training config)
CRITIC_REWARD_SCALE="${CRITIC_REWARD_SCALE:-10.0}"

# GEO Configuration
MAX_WORKERS=5
USE_CONCURRENT=True
USE_ASYNC=True
DATASET_SPLIT="${DATASET_SPLIT:-test}"
DATASET_TYPE="${DATASET_TYPE:-geobench}"  # Options: 'geobench' or 'msdata'

# ============================================
# Helper Functions
# ============================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${BLUE}[====]${NC} $1"
}

# Check if vLLM server is already running
check_existing_server() {
    python3 - <<PY
import sys
PORT = int("${VLLM_PORT}")
HEX_PORT = f"{PORT:04X}"

def has_listen(path: str) -> bool:
    try:
        lines = open(path).read().splitlines()[1:]
    except Exception:
        return False
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        local = parts[1]
        st = parts[3]  # 0A = LISTEN
        try:
            lp = local.split(":")[1].upper()
        except Exception:
            continue
        if lp == HEX_PORT and st == "0A":
            return True
    return False

sys.exit(0 if (has_listen("/proc/net/tcp") or has_listen("/proc/net/tcp6")) else 1)
PY
}

# ============================================
# Main Script
# ============================================

log_section "=========================================="
log_section "GEO System - Evolved Strategies Test"
log_section "=========================================="
echo ""

# Print GPU assignment
log_info "GPU Assignment:"
log_info "  GPU ${VLLM_GPU}: vLLM (Qwen2.5-32B generation engine)"
log_info "  GPU ${CRITIC_GPU}: Evolved Critic (strategy selection)"
echo ""

# ============================================
# Step 1: Check/Start vLLM Server
# ============================================
log_section "Step 1: vLLM Server"

if check_existing_server; then
    log_info "✓ vLLM server already running on port ${VLLM_PORT}"
else
    log_info "Starting vLLM server on GPU ${VLLM_GPU}..."
    log_info "  Model: ${MODEL_PATH}"
    log_info "  Model Name: ${MODEL_NAME}"
    log_info "  Port: ${VLLM_PORT}"
    log_info "  GPU Memory Utilization: ${GPU_MEMORY_UTIL}"
    log_info "  Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
    
    # Start vLLM on specified GPU
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} python3 vllm_wrapper.py \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --host "${VLLM_HOST}" \
        --port ${VLLM_PORT} \
        --dtype half \
        --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --swap-space ${SWAP_SPACE} \
        ${VLLM_EXTRA_ARGS} \
        > vllm_evolved_test.log 2>&1 &
    
    VLLM_PID=$!
    echo ${VLLM_PID} > vllm_evolved_test.pid
    log_info "vLLM server started with PID: ${VLLM_PID}"
    log_info "Logs: vllm_evolved_test.log"
fi

# Wait for server to be ready (regardless of whether we started it or it was already running)
log_info "Waiting for vLLM server to be ready..."
python3 check_llm_server.py "http://localhost:${VLLM_PORT}" ${VLLM_WAIT_TIMEOUT}

if [ $? -ne 0 ]; then
    log_error "vLLM server failed to start or become ready"
    if [ -f "vllm_evolved_test.log" ]; then
        log_error "Check vllm_evolved_test.log for details"
        log_error "Last 20 lines of log:"
        tail -20 vllm_evolved_test.log
    else
        log_error "Server was already running but not responding properly"
        log_error "Check if the server on port ${VLLM_PORT} is the correct vLLM instance"
    fi
    exit 1
fi

log_info "✓ vLLM server is ready!"

echo ""

# ============================================
# Step 2: Configure Environment
# ============================================
log_section "Step 2: Environment Configuration"

# Enable evolved surrogate model
export USE_EVOLVED_SURROGATE=1

# Set Critic to use specific GPU
# Note: vLLM runs as a separate server process (already started with CUDA_VISIBLE_DEVICES=${VLLM_GPU})
# The Critic model and GEO experiments will run on a different GPU
export CUDA_VISIBLE_DEVICES=${CRITIC_GPU}
export EVOLVED_CRITIC_DEVICE="cuda:0"  # Relative to CUDA_VISIBLE_DEVICES (i.e., actual GPU ${CRITIC_GPU})

# Evolved model paths (relative paths will be resolved by run_geo.py)
export EVOLVED_PRETRAINED_BACKBONE="${EVOLVED_PRETRAINED_BACKBONE}"
export EVOLVED_VALUE_HEAD="${EVOLVED_VALUE_HEAD}"
export EVOLVED_STRATEGIES="${EVOLVED_STRATEGIES}"
export CRITIC_REWARD_SCALE="${CRITIC_REWARD_SCALE}"

# vLLM API endpoint (vLLM runs on GPU ${VLLM_GPU}, accessed via HTTP API)
export USE_LOCAL_LLM=True
export LOCAL_LLM_MODEL="${MODEL_NAME}"
export API_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export API_KEY="token-abc123"

# GEO configuration
export USE_CONCURRENT=${USE_CONCURRENT}
export MAX_WORKERS=${MAX_WORKERS}
export USE_ASYNC=${USE_ASYNC}
export DATASET_SPLIT=${DATASET_SPLIT}
export DATASET_TYPE=${DATASET_TYPE}
export STATIC_CACHE=True

log_info "GPU Configuration:"
log_info "  vLLM server: GPU ${VLLM_GPU} (running as separate server process)"
log_info "  Critic model: GPU ${CRITIC_GPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo ""
log_info "Evolved Critic Configuration:"
log_info "  USE_EVOLVED_SURROGATE: ${USE_EVOLVED_SURROGATE}"
log_info "  Pretrained backbone: ${EVOLVED_PRETRAINED_BACKBONE}"
log_info "  Value head: ${EVOLVED_VALUE_HEAD}"
log_info "  Strategies: ${EVOLVED_STRATEGIES}"
log_info "  Critic reward scale: ${CRITIC_REWARD_SCALE}"
echo ""
log_info "GEO Configuration:"
log_info "  Dataset split: ${DATASET_SPLIT}"
log_info "  Dataset type: ${DATASET_TYPE}"
log_info "  USE_CONCURRENT: ${USE_CONCURRENT}"
log_info "  MAX_WORKERS: ${MAX_WORKERS}"
log_info "  LLM API: ${API_BASE_URL}"

echo ""

# ============================================
# Step 3: Verify Critic Model Loading
# ============================================
log_section "Step 3: Verify Critic Model"

log_info "Testing Critic model loading..."

# Get project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}"

# Use the same path resolution logic as run_geo.py
python3 -c "
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join('${PROJ_ROOT}', 'src'))

# Import after adding to path
from surrogate_model_evolved import EvolvedStrategySelector

# Get project root (same logic as run_geo.py)
proj_root = '${PROJ_ROOT}'

# Resolve base model path:
# 1) EVOLVED_BASE_MODEL (abs or relative to proj_root)
# 2) default to <proj_root>/base_model
base_model_path = os.environ.get('EVOLVED_BASE_MODEL', None)
if base_model_path and not os.path.isabs(base_model_path):
    base_model_path = os.path.join(proj_root, base_model_path)

if not base_model_path:
    base_model_path = os.path.join(proj_root, 'base_model')

if not os.path.exists(base_model_path):
    # Legacy fallbacks for older environments
    default_base_paths = [
        '/autodl-fs/data/modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct',
        '/autodl-fs/data/modelscope_cache/Qwen/Qwen2___5-1___5B-Instruct',
    ]
    for p in default_base_paths:
        if os.path.exists(p):
            base_model_path = p
            break

# Resolve paths (same logic as run_geo.py)
def resolve_path(p, default_rel=None):
    if not p and default_rel:
        p = default_rel
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(proj_root, p)

pretrained_backbone = resolve_path(os.environ.get('EVOLVED_PRETRAINED_BACKBONE', ''), 'surrogate_full1/stage1/checkpoint-610/exported_bin/pytorch_model.bin')
value_head = resolve_path(os.environ.get('EVOLVED_VALUE_HEAD', ''), 'outputs/ea_training/critic/value_head.bin')
strategies = resolve_path(os.environ.get('EVOLVED_STRATEGIES', ''), 'outputs/ea_training/final_strategies_with_prompts.json')

# Auto-detect LoRA adapter
lora_adapter = resolve_path('outputs/ea_training/critic/lora_adapter')
if not os.path.exists(lora_adapter):
    lora_adapter = None

print(f'[Test] Project root: {proj_root}')
print(f'[Test] Base model: {base_model_path}')
print(f'[Test] Pretrained backbone: {pretrained_backbone}')
print(f'[Test] Value head: {value_head}')
print(f'[Test] LoRA adapter: {lora_adapter}')
print(f'[Test] Strategies: {strategies}')

# Verify paths exist
if not os.path.exists(value_head):
    print(f'[Test] ERROR: Value head not found: {value_head}')
    exit(1)
if not os.path.exists(strategies):
    print(f'[Test] ERROR: Strategies file not found: {strategies}')
    exit(1)

# Load model
selector = EvolvedStrategySelector(
    base_model_path=base_model_path,
    value_head_path=value_head,
    strategies_path=strategies,
    lora_adapter_path=lora_adapter,
    pretrained_backbone_path=pretrained_backbone,
    device='cuda:0',
    critic_reward_scale=float(os.environ.get('CRITIC_REWARD_SCALE', '10.0'))
)

# Quick test
query = 'Test query'
context = 'Test content for strategy selection.'
best_id, result = selector.select_best_strategy(query, context)
print(f'[Test] Strategy selection successful!')
print(f'[Test] Best strategy: {best_id[:8]}... reward={result[\"expected_reward\"]:.4f}')
" 2>&1

if [ $? -ne 0 ]; then
    log_error "Critic model loading failed!"
    exit 1
fi

log_info "✓ Critic model loaded successfully"
echo ""

# ============================================
# Step 4: Test vLLM API
# ============================================
log_section "Step 4: Test vLLM API"

log_info "Testing vLLM API with a simple generation request..."
python3 -c "
import requests
import json

api_url = 'http://localhost:${VLLM_PORT}/v1/chat/completions'
headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer token-abc123'}
data = {
    'model': '${MODEL_NAME}',
    'messages': [{'role': 'user', 'content': 'Say hello in one word.'}],
    'max_tokens': 10,
    'temperature': 0
}

try:
    response = requests.post(api_url, headers=headers, json=data, timeout=60)
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f'[Test] vLLM API response: \"{content[:50]}...\"' if len(content) > 50 else f'[Test] vLLM API response: \"{content}\"')
        print('[Test] ✓ vLLM API is working!')
    else:
        print(f'[Test] ✗ vLLM API returned status {response.status_code}')
        print(f'[Test] Response: {response.text[:200]}')
        exit(1)
except Exception as e:
    print(f'[Test] ✗ vLLM API test failed: {e}')
    exit(1)
" 2>&1

if [ $? -ne 0 ]; then
    log_error "vLLM API test failed!"
    log_error "Make sure vLLM server is running and accessible at http://localhost:${VLLM_PORT}"
    exit 1
fi

log_info "✓ vLLM API is ready"
echo ""

# ============================================
# Step 5: Run GEO Experiments
# ============================================
log_section "Step 5: Run GEO Experiments"

log_info "Starting GEO experiments with evolved strategies..."
log_info "=========================================="
echo ""

python3 src/run_geo.py

EXIT_CODE=$?

# ============================================
# Summary
# ============================================
echo ""
log_section "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    log_info "✓ GEO experiments completed successfully!"
else
    log_error "✗ GEO experiments failed with exit code: ${EXIT_CODE}"
fi
log_section "=========================================="

exit ${EXIT_CODE}
