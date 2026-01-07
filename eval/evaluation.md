# ⚖️ Evaluation

## 1. Prepare Benchmarks
First, download the adapted benchmarks (formatted for VLMEvalKit) from HuggingFace and place them in your local `eval/LMUData` directory:

* **HuggingFace:** [🤗 ChartVerse_EvalBench](https://huggingface.co/datasets/ChartVerse_EvalBench)

## 2. Install Dependencies
Before running the evaluation, you need to set up the environment by installing VLMEvalKit in editable mode and installing necessary inference libraries.

```bash
cd eval/
git clone [https://github.com/open-compass/VLMEvalKit.git](https://github.com/open-compass/VLMEvalKit.git)
cd VLMEvalKit
pip install -e .

pip install vllm qwen-vl-utils openai
```

## 3. Run Evaluation (Inference)
We employ vLLM for high-throughput inference. Use the following script to start the model server and launch the evaluation loop. This script automatically handles server startup checks and dataset iteration.

```bash
# --- Configuration ---
export LMUData=LMUData/
export DATASET=CharXiv_reasoning_val,ChartX,ChartMuseum,EvoChart,ChartQAPro,ChartQA_TEST,CharXiv_descriptive_val,ChartBench
export MODEL=YOUR_MODEL_PATH
export MODEL_BASENAME=$(basename $MODEL)
export head_ip=localhost
export port=8000

mkdir -p server_logs
mkdir -p distill_logs

# --- 1. Start vLLM Server ---
vllm serve $MODEL \
    --max-model-len 65536 --tensor-parallel-size 8 \
    --limit-mm-per-prompt.video 0 \
    --gpu-memory-utilization 0.9 --enable-chunked-prefill \
    --max-num-seqs 512 --port $port \
    2>&1 | tee server_logs/${MODEL_BASENAME}.log &

echo "Waiting for vLLM service to start..."

# --- 2. Health Check Loop ---
TIMEOUT=600
ELAPSED=0
INTERVAL=20

while [ $ELAPSED -lt $TIMEOUT ]; do
    if grep -q "Application startup complete." server_logs/${MODEL_BASENAME}.log 2>/dev/null; then
        echo "vLLM service started successfully!"
        break
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "Waited ${ELAPSED}s..."
done

# --- 3. Run Inference ---
if [ $ELAPSED -lt $TIMEOUT ]; then
    python run.py --datasets $DATASET --model_name $MODEL --url $head_ip --port $port 2> 1 | tee "distill_logs/${MODEL_BASENAME}.log"
else
    echo "Error: vLLM failed to start within timeout."
fi
```

## 4. Metric Verification & Aggregation
After inference, we use Compass-Verifier-7B to evaluate the correctness of the generated responses and aggregate the final scores.

```bash
export MODEL=YOUR_MODEL_PATH
export MODEL_BASENAME=$(basename $MODEL)
export GPUS_PER_NODE=4
python verify.py \
    --model_name $MODEL_PATH \
    --tensor_parallel_size $GPUS_PER_NODE \
    2>&1 | tee verify_logs/${MODEL_BASENAME}.log
```