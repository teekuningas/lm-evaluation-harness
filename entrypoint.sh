#!/bin/bash
set -e

# Configuration
SERVER_URL=${SERVER_URL:-"http://localhost:7999"}
MODEL_NAME=${MODEL_NAME:-"gpt-oss-120b"}
OUTPUT_DIR=${OUTPUT_DIR:-"/results"}
EXTRA_ARGS="$@"

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}.json"

echo "=========================================="
echo "Finnish Benchmark Evaluation"
echo "=========================================="
echo ">>> Model name: $MODEL_NAME"
echo ">>> Server: $SERVER_URL"
echo ">>> Output: $OUTPUT_FILE"
echo ">>> Timestamp: $TIMESTAMP"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Wait for server to be ready
echo ">>> Checking server connectivity..."
for i in {1..30}; do
    if curl -s "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
        echo ">>> Server is ready!"
        break
    fi
    echo ">>> Waiting for server... ($i/30)"
    sleep 2
    if [ $i -eq 30 ]; then
        echo "ERROR: Server not responding after 60 seconds"
        exit 1
    fi
done

# Get model info from server
echo ">>> Fetching model info from server..."
MODEL_INFO=$(curl -s "${SERVER_URL}/v1/models" 2>/dev/null || echo '{"data":[{"id":"unknown"}]}')
SERVER_MODEL=$(echo "$MODEL_INFO" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "unknown")
echo ">>> Server reports model: $SERVER_MODEL"

echo ">>> Running Finnish benchmark..."
echo ">>> Command: lm_eval --model local-chat-completions --model_args base_url=${SERVER_URL}/v1/chat/completions,model=$SERVER_MODEL,max_gen_toks=2048 --apply_chat_template --batch_size 1 --output_path $OUTPUT_FILE $EXTRA_ARGS"
echo ">>> Note: max_gen_toks=2048 allows thinking models to complete reasoning + answer"
echo ">>> If you see 'null content' errors, increase max_gen_toks further"
echo "=========================================="

# Run the benchmark using local-chat-completions model
# Note: max_gen_toks needs to be high enough for reasoning models to complete reasoning + answer
# Using 2048 to allow for longer reasoning chains in thinking models

# Create log file path
LOG_FILE="${OUTPUT_FILE%.json}_full.log"

echo ">>> Full logs will be saved to: $LOG_FILE"

# Run with output redirected to log file
lm_eval \
    --model local-chat-completions \
    --model_args base_url=${SERVER_URL}/v1/chat/completions,model=$SERVER_MODEL,max_gen_toks=2048 \
    --apply_chat_template \
    --batch_size 1 \
    --output_path "$OUTPUT_FILE" \
    $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"

echo "=========================================="
echo ">>> Benchmark completed!"
echo ">>> Results saved to: $OUTPUT_FILE"
echo ">>> Full logs saved to: $LOG_FILE"
echo "=========================================="

# Create a summary file with metadata
SUMMARY_FILE="${OUTPUT_FILE%.json}_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Finnish Benchmark Evaluation Summary
=====================================
Model Name: $MODEL_NAME
Server Model: $SERVER_MODEL
Timestamp: $TIMESTAMP
Date: $(date)
Server URL: $SERVER_URL
Output File: $OUTPUT_FILE

Command:
lm_eval --model local-chat-completions --model_args base_url=${SERVER_URL}/v1/chat/completions,model=$SERVER_MODEL,max_gen_toks=2048 --apply_chat_template --batch_size 1 --output_path $OUTPUT_FILE $EXTRA_ARGS

Results:
--------
EOF

# Extract key metrics from results if possible
if [ -f "$OUTPUT_FILE" ]; then
    echo ">>> Extracting metrics..."
    python3 -c "
import json
import sys
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for task, metrics in results.items():
        print(f'{task}:')
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f'  {metric}: {value:.4f}')
        print()
except Exception as e:
    print(f'Could not parse results: {e}', file=sys.stderr)
" >> "$SUMMARY_FILE" 2>/dev/null || echo "See $OUTPUT_FILE for full results" >> "$SUMMARY_FILE"
else
    echo "See $OUTPUT_FILE for full results" >> "$SUMMARY_FILE"
fi

echo ">>> Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
echo "=========================================="
