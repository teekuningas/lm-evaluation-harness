#!/usr/bin/env bash

# Aggregate FIN-bench Results v2
# Properly handles invalid responses (filtered=empty) by reading sample logs
# Invalid responses are EXCLUDED from accuracy calculation

RESULTS_DIR=${1:-"./results"}

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "=========================================="
echo "FIN-bench Results Aggregation v2"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Methodology: Invalid responses (filtered=empty) EXCLUDED from accuracy"
echo ""

# Find all result JSON files
JSON_FILES=$(find "$RESULTS_DIR" -name "*.json" -not -path "*summary*" | sort)

if [ -z "$JSON_FILES" ]; then
    echo "No result files found in $RESULTS_DIR"
    exit 1
fi

NUM_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found $NUM_FILES model result(s)"
echo ""

# Create timestamp for output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$RESULTS_DIR/aggregated_results_v2_$TIMESTAMP.csv"
echo "Model,Task,Raw_Acc,Valid_Acc,Correct,Wrong,Invalid,Total,Invalid_Pct" > "$CSV_FILE"

echo "=========================================="
echo "Model Comparison Summary"
echo "=========================================="
printf "%-40s %10s %10s %10s %10s\n" "Model (timestamp)" "Valid_Acc" "Invalid%" "Raw_Acc" "Samples"
echo "--------------------------------------------------------------------------------"

# Process each JSON file  
for json_file in $JSON_FILES; do
    filename=$(basename "$json_file")
    
    # Extract model name and full timestamp from filename
    # Format: v2_model-name_YYYYMMDD_HHMMSS_YYYY-MM-DDTHH-MM-SS.ffffff.json
    model_name=$(echo "$filename" | sed 's/_[0-9]\{8\}_[0-9]\{6\}_.*\.json$//')
    short_ts=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | head -1)
    full_ts=$(echo "$filename" | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}\.[0-9]\{6\}')
    
    model_display="${model_name} (${short_ts})"
    
    # Parse results using Python
    result=$(python3 - "$json_file" "$model_name" "$full_ts" "$RESULTS_DIR" "$CSV_FILE" << 'EOFPYTHON'
import json
import glob
import sys

json_file = sys.argv[1]
model_name = sys.argv[2]
full_timestamp = sys.argv[3]  # Format: YYYY-MM-DDTHH-MM-SS.ffffff
results_dir = sys.argv[4]
csv_file = sys.argv[5]

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    task_stats = {}
    
    for task in results.keys():
        correct = 0
        wrong = 0
        invalid = 0
        
        # Try to find sample files for this task
        # Pattern: samples_{task}_YYYY-MM-DDTHH-MM-SS.ffffff.jsonl
        pattern = f"{results_dir}/samples_{task}_{full_timestamp}.jsonl"
        sample_files = glob.glob(pattern)
        
        if sample_files:
            # Read sample file to count correct/wrong/invalid
            with open(sample_files[0], 'r') as sf:
                for line in sf:
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                        filtered = sample.get('filtered_resps', [[]])[0]
                        exact_match = sample.get('exact_match', 0.0)
                        
                        # Check if filtered is empty -> INVALID
                        is_invalid = (filtered == '' or filtered == [''] or 
                                    filtered is None or filtered == [])
                        
                        if is_invalid:
                            invalid += 1
                        elif exact_match == 1.0:
                            correct += 1
                        else:
                            wrong += 1
                    except Exception as e:
                        pass
        else:
            # No sample file found, use aggregate data (no invalid count available)
            acc = results[task].get('exact_match,finbench_v2_answer_extraction')
            if acc is None:
                acc = results[task].get('exact_match,finbench_answer_extraction', 0)
            
            n_samples = data.get('n-samples', {}).get(task, {}).get('effective', 0)
            correct = int(acc * n_samples)
            wrong = n_samples - correct
            invalid = 0
        
        task_stats[task] = {
            'correct': correct,
            'wrong': wrong,
            'invalid': invalid,
            'total': correct + wrong + invalid
        }
    
    # Calculate totals
    total_correct = sum(s['correct'] for s in task_stats.values())
    total_wrong = sum(s['wrong'] for s in task_stats.values())
    total_invalid = sum(s['invalid'] for s in task_stats.values())
    total_samples = sum(s['total'] for s in task_stats.values())
    
    # Calculate accuracies
    # CRITICAL: Valid accuracy EXCLUDES invalids from denominator
    if (total_correct + total_wrong) > 0:
        valid_acc = total_correct / (total_correct + total_wrong)
    else:
        valid_acc = 0.0
    
    if total_samples > 0:
        raw_acc = total_correct / total_samples  # Includes invalids as wrong
        invalid_pct = total_invalid / total_samples
    else:
        raw_acc = 0.0
        invalid_pct = 0.0
    
    # Write task-level data to CSV
    with open(csv_file, 'a') as f:
        for task, stats in task_stats.items():
            t_raw = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            t_valid = stats['correct'] / (stats['correct'] + stats['wrong']) if (stats['correct'] + stats['wrong']) > 0 else 0
            t_inv_pct = stats['invalid'] / stats['total'] if stats['total'] > 0 else 0
            
            f.write(f"{model_name},{task},{t_raw:.4f},{t_valid:.4f},"
                   f"{stats['correct']},{stats['wrong']},{stats['invalid']},"
                   f"{stats['total']},{t_inv_pct:.4f}\n")
    
    # Output for bash
    print(f"{valid_acc:.4f} {invalid_pct:.4f} {raw_acc:.4f} {total_samples}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("N/A N/A N/A 0")

EOFPYTHON
)
    
    read valid_acc invalid_pct raw_acc total_samples <<< "$result"
    printf "%-40s %10s %10s %10s %10s\n" "$model_display" "$valid_acc" "$invalid_pct" "$raw_acc" "$total_samples"
done

echo "--------------------------------------------------------------------------------"
echo ""
echo "LEGEND:"
echo "  Valid_Acc:  Accuracy on valid responses only = Correct / (Correct + Wrong)"
echo "             This EXCLUDES invalid responses from calculation"
echo "  Invalid%:   Percentage of responses that couldn't be parsed/extracted"
echo "  Raw_Acc:    Original lm-eval accuracy (treats invalid as wrong)"
echo ""
echo "Valid_Acc is the TRUE CAPABILITY score (unbiased by system prompt)"
echo "Invalid% shows how well models follow formatting instructions"
echo ""

# Print detailed task-specific results
echo "=========================================="
echo "Task-Specific Results"
echo "=========================================="
echo ""

# Group results by model - process CSV with Python for reliability
python3 - "$CSV_FILE" "$JSON_FILES" << 'EOFPYTHON2'
import csv
import sys

csv_file = sys.argv[1]
json_files = sys.argv[2].split()

# Read CSV data
tasks_by_model = {}
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row['Model']
        if model not in tasks_by_model:
            tasks_by_model[model] = []
        tasks_by_model[model].append(row)

# Extract model names and timestamps from JSON filenames
import os
model_info = []
for jf in json_files:
    filename = os.path.basename(jf)
    model_name = filename.split('_202')[0]  # Everything before _YYYYMMDD
    short_ts = filename.split('_')[1] + '_' + filename.split('_')[2] if len(filename.split('_')) > 2 else 'unknown'
    model_info.append((model_name, short_ts))

# Print results grouped by model
for model_name, short_ts in sorted(set(model_info)):
    if model_name not in tasks_by_model:
        continue
    
    print(f"Model: {model_name} ({short_ts})")
    print("-" * 100)
    print(f"{'Task':<60} {'Valid_Acc':>10} {'Invalid%':>10} {'Samples':>8}")
    print("-" * 100)
    
    for row in sorted(tasks_by_model[model_name], key=lambda x: x['Task']):
        task = row['Task']
        valid_acc = float(row['Valid_Acc'])
        inv_pct = float(row['Invalid_Pct'])
        total = row['Total']
        print(f"{task:<60} {valid_acc:>10.4f} {inv_pct:>10.4f} {total:>8}")
    
    print()

EOFPYTHON2

echo "=========================================="
echo "Results saved to: $CSV_FILE"
echo "=========================================="
