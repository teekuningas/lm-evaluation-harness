#!/usr/bin/env bash

# Aggregate FIN-bench results across multiple models
# Works with both v1 and v2 results
# Usage: ./aggregate_results.sh [results_dir]

RESULTS_DIR=${1:-"./results"}

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "=========================================="
echo "FIN-bench Results Aggregation"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Find all result JSON files (exclude summary files and old timestamps)
JSON_FILES=$(find "$RESULTS_DIR" -name "*.json" -not -path "*summary*" | sort)

if [ -z "$JSON_FILES" ]; then
    echo "No result files found in $RESULTS_DIR"
    exit 1
fi

NUM_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found $NUM_FILES model result(s)"
echo ""

# Check if Python is available for JSON parsing
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed"
    exit 1
fi

# Create timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# CSV output
CSV_FILE="$RESULTS_DIR/aggregated_results_$TIMESTAMP.csv"
echo "Model,Task,Exact_Match,Samples,Limit,Date" > "$CSV_FILE"

echo "=========================================="
echo "Model Comparison Summary"
echo "=========================================="
printf "%-40s %10s %10s %8s\n" "Model (timestamp)" "Avg_Acc" "Samples" "Limit"
echo "--------------------------------------------------------------------------------"

# Process each JSON file
for json_file in $JSON_FILES; do
    # Extract model name and timestamp from filename
    filename=$(basename "$json_file")
    # Extract just the model name and date (not the full timestamp)
    model_name=$(echo "$filename" | sed 's/_[0-9]\{8\}_[0-9]\{6\}_.*\.json$//')
    timestamp=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | head -1)
    model_display="${model_name} (${timestamp})"
    
    # Extract date from filename
    date_str=$(echo "$timestamp" || echo "unknown")
    
    # Parse results using Python and output summary line
    read avg_acc total_samples limit <<< $(python3 << EOF
import json
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    config = data.get('config', {})
    n_samples = data.get('n-samples', {})
    
    limit = config.get('limit', 'none')
    
    # Calculate overall metrics and write to CSV
    total_acc = 0
    total_samples = 0
    count = 0
    
    for task, task_results in results.items():
        # Handle different metric naming conventions
        acc = task_results.get('exact_match,finbench_v2_answer_extraction')
        if acc is None:
            acc = task_results.get('exact_match,finbench_answer_extraction')
        if acc is None:
            acc = task_results.get('exact_match,none')
        if acc is None:
            acc = task_results.get('exact_match')
        if acc is None:
            acc = 0
        samples = n_samples.get(task, {}).get('effective', 0)
        
        total_acc += acc
        total_samples += samples
        count += 1
        
        # Write to CSV
        with open('$CSV_FILE', 'a') as f:
            f.write(f'$model_name,{task},{acc},{samples},{limit},$date_str\n')
    
    # Calculate average and print for bash to read
    if count > 0:
        avg_acc = total_acc / count
        print(f'{avg_acc:.4f} {total_samples} {limit}')
    else:
        print('N/A 0 none')
        
except Exception as e:
    print(f'N/A 0 error', file=sys.stderr)
    print('N/A 0 error')
EOF
)
    
    printf "%-40s %10s %10s %8s\n" "$model_display" "$avg_acc" "$total_samples" "$limit"
done

echo "--------------------------------------------------------------------------------"
echo ""

# Detailed breakdown by task
echo "=========================================="
echo "Detailed Breakdown by Task"
echo "=========================================="
echo ""

# Extract all unique tasks from all JSON files
ALL_TASKS=$(python3 << EOF
import json
import sys

tasks = set()
json_files = '''$JSON_FILES'''.strip().split('\n')

for json_file in json_files:
    json_file = json_file.strip()
    if not json_file:
        continue
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        tasks.update(data.get('results', {}).keys())
    except Exception as e:
        print(f'Error reading {json_file}: {e}', file=sys.stderr)
        pass

# Sort tasks for consistent output
for task in sorted(tasks):
    print(task)
EOF
)

if [ -z "$ALL_TASKS" ]; then
    echo "No tasks found in result files"
    exit 0
fi

# Convert to array
TASKS_ARRAY=()
while IFS= read -r task; do
    TASKS_ARRAY+=("$task")
done <<< "$ALL_TASKS"

printf "%-50s" "Task"
for json_file in $JSON_FILES; do
    filename=$(basename "$json_file")
    model_name=$(echo "$filename" | sed 's/_[0-9]\{8\}_[0-9]\{6\}_.*\.json$//' | sed 's/_[0-9]\{8\}_[0-9]\{6\}.*\.json$//')
    printf " %15s" "$(echo $model_name | head -c 15)"
done
echo ""
echo "--------------------------------------------------------------------------------"

for task in "${TASKS_ARRAY[@]}"; do
    # Shorten task name for display (keep p0-p4 variants visible)
    task_short=$(echo "$task" | sed 's/FIN-bench_//' | sed 's/_generate_until//' | sed 's/_gen_mcf_fbv2_/\//' | sed 's/_fi//')
    printf "%-50s" "$task_short"
    
    for json_file in $JSON_FILES; do
        acc=$(python3 << EOF
import json
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    task_data = data.get('results', {}).get('$task', {})
    # Handle different metric naming conventions  
    result = task_data.get('exact_match,finbench_v2_answer_extraction')
    if result is None:
        result = task_data.get('exact_match,finbench_answer_extraction')
    if result is None:
        result = task_data.get('exact_match,none')
    if result is None:
        result = task_data.get('exact_match')
    print(f'{result:.4f}' if result is not None else 'N/A')
except:
    print('N/A')
EOF
)
        printf " %15s" "$acc"
    done
    echo ""
done

echo ""
echo "=========================================="
echo "Results saved to: $CSV_FILE"
echo "=========================================="
