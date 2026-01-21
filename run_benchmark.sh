#!/bin/bash

# Finnish Benchmark Runner with Container
# Usage: 
#   ./run_benchmark.sh gpt-oss-120b                          # Quick v1 (3 tasks, 5 samples)
#   BENCHMARK=v1 SUBSET=full ./run_benchmark.sh gpt-oss-120b # Full v1 (12 tasks)
#   BENCHMARK=v2 ./run_benchmark.sh gemma3-27b               # v2 (2 tasks: squad, truthfulqa)
#   LIMIT=50 API_ADDRESS=localhost:7999 ./run_benchmark.sh model-name

set -e

if [ -z "$1" ]; then
    echo "Error: Model name required"
    echo ""
    echo "Usage examples:"
    echo "  ./run_benchmark.sh <model-name>                          # Quick v1 (3 tasks)"
    echo "  BENCHMARK=v1 SUBSET=full ./run_benchmark.sh <model-name> # Full v1 (12 tasks)"
    echo "  BENCHMARK=v2 ./run_benchmark.sh <model-name>             # v2 (squad + truthfulqa)"
    echo "  LIMIT=100 ./run_benchmark.sh <model-name>                # Custom sample limit"
    exit 1
fi

MODEL_NAME="$1"

# Configuration with defaults
RESULTS_DIR="./results"
API_ADDRESS=${API_ADDRESS:-"localhost:7999"}
LIMIT=${LIMIT:-"5"}  # Samples per task (e.g., 5 means 5 samples from each task)
BENCHMARK=${BENCHMARK:-"v1"}  # v1 or v2
SUBSET=${SUBSET:-"quick"}  # quick (3 tasks) or full (12 tasks) - only for v1
IMAGE_NAME="finbench-eval:latest"

# SUBSET controls WHICH tasks to run (3 vs 12 tasks for v1)
# LIMIT controls HOW MANY samples per task (default 5)
# Example: SUBSET=full LIMIT=10 → runs all 12 tasks with 10 samples each = 120 total samples

# Parse host and port from API_ADDRESS
if [[ $API_ADDRESS == *":"* ]]; then
    SERVER_URL="http://${API_ADDRESS}"
else
    SERVER_URL="http://${API_ADDRESS}:7999"
fi

# Define task lists
# FIN-bench v1: 12 generate_until tasks
V1_QUICK_TASKS="FIN-bench_analogies_generate_until,FIN-bench_general_knowledge_generate_until,FIN-bench_emotions_generate_until"
V1_FULL_TASKS="FIN-bench_analogies_generate_until,FIN-bench_arithmetic_generate_until,FIN-bench_cause_and_effect_generate_until,FIN-bench_emotions_generate_until,FIN-bench_empirical_judgments_generate_until,FIN-bench_general_knowledge_generate_until,FIN-bench_hhh_alignment_generate_until,FIN-bench_intent_recognition_generate_until,FIN-bench_misconceptions_generate_until,FIN-bench_paraphrase_generate_until,FIN-bench_sentence_ambiguity_generate_until,FIN-bench_similarities_abstraction_generate_until"

# FIN-bench v2: generate_until tasks (all in finbench_v2/gen/)
# Each category has CF and MCF variants with 5 prompts each (p0-p4)
V2_GEN_TASKS_ARC="arc_challenge_fi_gen_cf_fbv2_p0,arc_challenge_fi_gen_cf_fbv2_p1,arc_challenge_fi_gen_cf_fbv2_p2,arc_challenge_fi_gen_cf_fbv2_p3,arc_challenge_fi_gen_cf_fbv2_p4,arc_challenge_fi_gen_mcf_fbv2_p0,arc_challenge_fi_gen_mcf_fbv2_p1,arc_challenge_fi_gen_mcf_fbv2_p2,arc_challenge_fi_gen_mcf_fbv2_p3,arc_challenge_fi_gen_mcf_fbv2_p4"
V2_GEN_TASKS_BELEBELE="belebele_fin_gen_cf_fbv2_p0,belebele_fin_gen_cf_fbv2_p1,belebele_fin_gen_cf_fbv2_p2,belebele_fin_gen_cf_fbv2_p3,belebele_fin_gen_cf_fbv2_p4,belebele_fin_gen_mcf_fbv2_p0,belebele_fin_gen_mcf_fbv2_p1,belebele_fin_gen_mcf_fbv2_p2,belebele_fin_gen_mcf_fbv2_p3,belebele_fin_gen_mcf_fbv2_p4"
V2_GEN_TASKS_GOLDENSWAG="goldenswag_fi_gen_cf_fbv2_p0,goldenswag_fi_gen_cf_fbv2_p1,goldenswag_fi_gen_cf_fbv2_p2,goldenswag_fi_gen_cf_fbv2_p3,goldenswag_fi_gen_cf_fbv2_p4,goldenswag_fi_gen_mcf_fbv2_p0,goldenswag_fi_gen_mcf_fbv2_p1,goldenswag_fi_gen_mcf_fbv2_p2,goldenswag_fi_gen_mcf_fbv2_p3,goldenswag_fi_gen_mcf_fbv2_p4"
V2_GEN_TASKS_SCANDISENT="scandisent_fi_gen_cf_fbv2_p0,scandisent_fi_gen_cf_fbv2_p1,scandisent_fi_gen_cf_fbv2_p2,scandisent_fi_gen_cf_fbv2_p3,scandisent_fi_gen_cf_fbv2_p4,scandisent_fi_gen_mcf_fbv2_p0,scandisent_fi_gen_mcf_fbv2_p1,scandisent_fi_gen_mcf_fbv2_p2,scandisent_fi_gen_mcf_fbv2_p3,scandisent_fi_gen_mcf_fbv2_p4"
V2_GEN_TASKS_SIB200="sib200_fi_gen_cf_fbv2_p0,sib200_fi_gen_cf_fbv2_p1,sib200_fi_gen_cf_fbv2_p2,sib200_fi_gen_cf_fbv2_p3,sib200_fi_gen_cf_fbv2_p4,sib200_fi_gen_mcf_fbv2_p0,sib200_fi_gen_mcf_fbv2_p1,sib200_fi_gen_mcf_fbv2_p2,sib200_fi_gen_mcf_fbv2_p3,sib200_fi_gen_mcf_fbv2_p4"

# All v2 generation tasks
V2_TASKS="${V2_GEN_TASKS_ARC},${V2_GEN_TASKS_BELEBELE},${V2_GEN_TASKS_GOLDENSWAG},${V2_GEN_TASKS_SCANDISENT},${V2_GEN_TASKS_SIB200}"

# Quick test (just one task from each category)
V2_TASKS_QUICK="arc_challenge_fi_gen_cf_fbv2_p0,belebele_fin_gen_cf_fbv2_p0,goldenswag_fi_gen_cf_fbv2_p0,scandisent_fi_gen_cf_fbv2_p0,sib200_fi_gen_cf_fbv2_p0"

# Select tasks based on benchmark version and subset
if [ "$BENCHMARK" = "v2" ]; then
    TASKS="$V2_TASKS"
    TASK_COUNT=50
    DESCRIPTION="FIN-bench v2 generate_until tasks (arc_c, belebele, goldenswag, scandisent, sib200 - CF+MCF, 5 prompts each)"
elif [ "$BENCHMARK" = "v2-quick" ]; then
    TASKS="$V2_TASKS_QUICK"
    TASK_COUNT=5
    DESCRIPTION="FIN-bench v2 quick test (1 task per category)"
elif [ "$SUBSET" = "full" ]; then
    TASKS="$V1_FULL_TASKS"
    TASK_COUNT=12
    DESCRIPTION="All 12 FIN-bench v1 generate_until tasks"
else
    TASKS="$V1_QUICK_TASKS"
    TASK_COUNT=3
    DESCRIPTION="Quick subset: 3 FIN-bench v1 tasks (analogies, general_knowledge, emotions)"
fi

TOTAL_SAMPLES=$((TASK_COUNT * LIMIT))

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Finnish Benchmark Evaluation"
echo "=========================================="
echo "Model name: $MODEL_NAME"
echo "Benchmark: $BENCHMARK"
echo "Subset: $SUBSET"
echo "API address: $API_ADDRESS"
echo "Server URL: $SERVER_URL"
echo "Tasks: $TASK_COUNT"
echo "Samples per task: $LIMIT"
echo "Total samples: $TOTAL_SAMPLES"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Description: $DESCRIPTION"
echo ""
echo "Step 1: Building container image..."
echo "=========================================="

# Build the container image
podman build -t "$IMAGE_NAME" -f Containerfile .

echo ""
echo "=========================================="
echo "Step 2: Running benchmark..."
echo "=========================================="
echo ""
echo "$DESCRIPTION"
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
echo "=========================================="

sleep 3

# Run the benchmark container
podman run --rm -it \
    --network host \
    -v "$(pwd)/results:/results:Z" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e SERVER_URL="$SERVER_URL" \
    -e NO_LIMIT="" \
    "$IMAGE_NAME" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --log_samples

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo "Results saved in: $RESULTS_DIR/${MODEL_NAME}_*.json"
echo ""
echo "Next steps:"
echo "  - Compare results: ./aggregate_results.sh"
if [ "$BENCHMARK" = "v1" ] && [ "$SUBSET" = "quick" ]; then
echo "  - Run full v1: BENCHMARK=v1 SUBSET=full ./run_benchmark.sh $MODEL_NAME"
fi
if [ "$BENCHMARK" = "v1" ]; then
echo "  - Try v2: BENCHMARK=v2 ./run_benchmark.sh $MODEL_NAME"
fi
if [ "$BENCHMARK" = "v2" ]; then
echo "  - Try v1: BENCHMARK=v1 ./run_benchmark.sh $MODEL_NAME"
fi
echo "  - More samples: LIMIT=50 ./run_benchmark.sh $MODEL_NAME"
echo "=========================================="
