#!/bin/bash

# Finnish Benchmark Runner
# Simple usage:
#   BENCHMARK=v1 LIMIT=10 ./run_benchmark.sh gpt-oss-120b        # v1 legacy (12 tasks)
#   BENCHMARK=v2 LIMIT=10 ./run_benchmark.sh gpt-oss-120b        # v2 all (100 tasks)
#   BENCHMARK=v2_new LIMIT=10 ./run_benchmark.sh gpt-oss-120b    # v2 new datasets only (50 tasks)

set -e

if [ -z "$1" ]; then
    echo "Usage: BENCHMARK=<v1|v2|v2_new> LIMIT=<samples> ./run_benchmark.sh <model-name>"
    echo ""
    echo "Examples:"
    echo "  BENCHMARK=v1 LIMIT=10 ./run_benchmark.sh gpt-oss-120b"
    echo "  BENCHMARK=v2 LIMIT=10 ./run_benchmark.sh gpt-oss-120b"
    echo "  BENCHMARK=v2_new LIMIT=10 ./run_benchmark.sh gpt-oss-120b"
    exit 1
fi

MODEL_NAME="$1"
RESULTS_DIR="./results"
API_ADDRESS=${API_ADDRESS:-"localhost:7999"}
BENCHMARK=${BENCHMARK:-"v2"}
LIMIT=${LIMIT:-"10"}
IMAGE_NAME="finbench-eval:latest"

# Parse server URL
if [[ $API_ADDRESS == *":"* ]]; then
    SERVER_URL="http://${API_ADDRESS}"
else
    SERVER_URL="http://${API_ADDRESS}:7999"
fi

# V1 legacy tasks (12 tasks)
V1_TASKS="FIN-bench_analogies_generate_until,FIN-bench_arithmetic_generate_until,FIN-bench_cause_and_effect_generate_until,FIN-bench_emotions_generate_until,FIN-bench_empirical_judgments_generate_until,FIN-bench_general_knowledge_generate_until,FIN-bench_hhh_alignment_generate_until,FIN-bench_intent_recognition_generate_until,FIN-bench_misconceptions_generate_until,FIN-bench_paraphrase_generate_until,FIN-bench_sentence_ambiguity_generate_until,FIN-bench_similarities_abstraction_generate_until"

# V2 new dataset tasks (25 tasks: arc, belebele, goldenswag, scandisent, sib200 - MCF only)
V2_NEW_ARC="arc_challenge_fi_gen_mcf_fbv2_p0,arc_challenge_fi_gen_mcf_fbv2_p1,arc_challenge_fi_gen_mcf_fbv2_p2,arc_challenge_fi_gen_mcf_fbv2_p3,arc_challenge_fi_gen_mcf_fbv2_p4"
V2_NEW_BELEBELE="belebele_fin_gen_mcf_fbv2_p0,belebele_fin_gen_mcf_fbv2_p1,belebele_fin_gen_mcf_fbv2_p2,belebele_fin_gen_mcf_fbv2_p3,belebele_fin_gen_mcf_fbv2_p4"
V2_NEW_GOLDENSWAG="goldenswag_fi_gen_mcf_fbv2_p0,goldenswag_fi_gen_mcf_fbv2_p1,goldenswag_fi_gen_mcf_fbv2_p2,goldenswag_fi_gen_mcf_fbv2_p3,goldenswag_fi_gen_mcf_fbv2_p4"
V2_NEW_SCANDISENT="scandisent_fi_gen_mcf_fbv2_p0,scandisent_fi_gen_mcf_fbv2_p1,scandisent_fi_gen_mcf_fbv2_p2,scandisent_fi_gen_mcf_fbv2_p3,scandisent_fi_gen_mcf_fbv2_p4"
V2_NEW_SIB200="sib200_fi_gen_mcf_fbv2_p0,sib200_fi_gen_mcf_fbv2_p1,sib200_fi_gen_mcf_fbv2_p2,sib200_fi_gen_mcf_fbv2_p3,sib200_fi_gen_mcf_fbv2_p4"
V2_NEW_TASKS="${V2_NEW_ARC},${V2_NEW_BELEBELE},${V2_NEW_GOLDENSWAG},${V2_NEW_SCANDISENT},${V2_NEW_SIB200}"

# V2 updated v1 tasks (25 tasks: analogies, emotions, knowledge, hhh, similarities - MCF only)
V2_V1_ANALOGIES="finbench_analogies_gen_mcf_fbv2_p0,finbench_analogies_gen_mcf_fbv2_p1,finbench_analogies_gen_mcf_fbv2_p2,finbench_analogies_gen_mcf_fbv2_p3,finbench_analogies_gen_mcf_fbv2_p4"
V2_V1_EMOTIONS="finbench_emotions_1k_gen_mcf_fbv2_p0,finbench_emotions_1k_gen_mcf_fbv2_p1,finbench_emotions_1k_gen_mcf_fbv2_p2,finbench_emotions_1k_gen_mcf_fbv2_p3,finbench_emotions_1k_gen_mcf_fbv2_p4"
V2_V1_KNOWLEDGE="finbench_general_knowledge_gen_mcf_fbv2_p0,finbench_general_knowledge_gen_mcf_fbv2_p1,finbench_general_knowledge_gen_mcf_fbv2_p2,finbench_general_knowledge_gen_mcf_fbv2_p3,finbench_general_knowledge_gen_mcf_fbv2_p4"
V2_V1_HHH="finbench_hhh_alignment_gen_mcf_fbv2_p0,finbench_hhh_alignment_gen_mcf_fbv2_p1,finbench_hhh_alignment_gen_mcf_fbv2_p2,finbench_hhh_alignment_gen_mcf_fbv2_p3,finbench_hhh_alignment_gen_mcf_fbv2_p4"
V2_V1_SIMILARITIES="finbench_similarities_abstraction_gen_mcf_fbv2_p0,finbench_similarities_abstraction_gen_mcf_fbv2_p1,finbench_similarities_abstraction_gen_mcf_fbv2_p2,finbench_similarities_abstraction_gen_mcf_fbv2_p3,finbench_similarities_abstraction_gen_mcf_fbv2_p4"
V2_V1_TASKS="${V2_V1_ANALOGIES},${V2_V1_EMOTIONS},${V2_V1_KNOWLEDGE},${V2_V1_HHH},${V2_V1_SIMILARITIES}"

# V2 all (50 tasks: MCF only, CF removed as redundant)
V2_ALL_TASKS="${V2_NEW_TASKS},${V2_V1_TASKS}"

# Select tasks based on BENCHMARK
case "$BENCHMARK" in
    v1)
        TASKS="$V1_TASKS"
        TASK_COUNT=12
        DESCRIPTION="FINBench v1 legacy (12 tasks)"
        ;;
    v2)
        TASKS="$V2_ALL_TASKS"
        TASK_COUNT=50
        DESCRIPTION="FINBench v2 all (50 tasks: 25 new + 25 updated v1, MCF only)"
        ;;
    v2_new)
        TASKS="$V2_NEW_TASKS"
        TASK_COUNT=25
        DESCRIPTION="FINBench v2 new datasets only (25 tasks, MCF only)"
        ;;
    v2_v1)
        TASKS="$V2_V1_TASKS"
        TASK_COUNT=25
        DESCRIPTION="FINBench v2 updated v1 only (25 tasks, MCF only)"
        ;;
    *)
        echo "Error: Unknown BENCHMARK='$BENCHMARK'"
        echo "Valid options: v1, v2, v2_new, v2_v1"
        exit 1
        ;;
esac

TOTAL_SAMPLES=$((TASK_COUNT * LIMIT))

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Finnish Benchmark Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Benchmark: $BENCHMARK"
echo "Server: $SERVER_URL"
echo "Tasks: $TASK_COUNT"
echo "Samples per task: $LIMIT"
echo "Total samples: $TOTAL_SAMPLES"
echo ""
echo "$DESCRIPTION"
echo ""
echo "Building container..."
echo "=========================================="

podman build -t "$IMAGE_NAME" -f Containerfile .

echo ""
echo "=========================================="
echo "Running benchmark..."
echo "=========================================="
echo ""

podman run --rm -it \
    --network host \
    -v "$(pwd)/results:/results:Z" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e SERVER_URL="$SERVER_URL" \
    -e ADD_CHAT_TYPE_FIELD="${ADD_CHAT_TYPE_FIELD:-true}" \
    -e NO_LIMIT="" \
    "$IMAGE_NAME" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --log_samples

echo ""
echo "=========================================="
echo "Complete! Results in: $RESULTS_DIR/"
echo "=========================================="
