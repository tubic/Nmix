#!/bin/bash

# Fixed parameters
GPUS=(1 2 3)
BASES=(A C G U all)
WEIGHT_FACTORS=($(seq 0.1 0.1 2.0))
RATIOS=(10)

# Log file
LOG_FILE="./experiment_log.txt"

# Clear the log file
> "$LOG_FILE"

# Function: Read completed combinations from CSV file
get_completed_combinations() {
    tail -n +2 ../train_metrics.csv | \
    awk -F',' '$4 == "avg" {printf "%s,%s,%.2f,%s\n", $1, $2, $3, $4}' | \
    sort | uniq
}

# Get combinations to run
generate_combinations() {
    for ratio in "${RATIOS[@]}"; do
        for base in "${BASES[@]}"; do
            for weight_factor in "${WEIGHT_FACTORS[@]}"; do
                printf "%s,%s,%.2f,avg\n" "$ratio" "$base" "$weight_factor"
            done
        done
    done
}

# Get combinations to run
COMBINATIONS=$(comm -23 <(generate_combinations | sort) <(get_completed_combinations | sort))

# Convert combinations to array
readarray -t COMBO_ARRAY <<< "$COMBINATIONS"

# Function: Run a single experiment
run_experiment() {
    local gpu=$1
    local combo=$2
    IFS=',' read -r ratio base weight_factor fold <<< "$combo"
    if [[ -n "$ratio" && -n "$base" && -n "$weight_factor" ]]; then
        echo "Running with GPU=$gpu, RATIO=$ratio, BASE=$base, WEIGHT_FACTOR=$weight_factor" | tee -a "$LOG_FILE"
        python train.py --gpu "$gpu" --ratio "$ratio" --base "$base" --weight_factor "$weight_factor" &>> "$LOG_FILE"
    else
        echo "Error: Empty values detected. GPU=$gpu, RATIO=$ratio, BASE=$base, WEIGHT_FACTOR=$weight_factor" | tee -a "$LOG_FILE"
    fi
}

# Initialize task counter and PID array
declare -A running_tasks
declare -A gpu_pids

# Function: Update the number of tasks running on GPU
update_running_tasks() {
    local gpu=$1
    local pids=(${gpu_pids[$gpu]})
    running_tasks[$gpu]=0
    for pid in "${pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            running_tasks[$gpu]=$((running_tasks[$gpu] + 1))
        else
            gpu_pids[$gpu]=${gpu_pids[$gpu]/$pid/}
        fi
    done
}

# Main loop
for combo in "${COMBO_ARRAY[@]}"; do
    while true; do
        for gpu in "${GPUS[@]}"; do
            update_running_tasks $gpu
            if [ "${running_tasks[$gpu]:-0}" -lt 2 ]; then
                run_experiment "$gpu" "$combo" &
                pid=$!
                gpu_pids[$gpu]+="$pid "
                break 2  # Found available GPU, break out of two loops
            fi
        done
        sleep 5  # If all GPUs are busy, wait 5 seconds before checking again
    done
done

# Wait for all background processes to complete
for gpu in "${GPUS[@]}"; do
    for pid in ${gpu_pids[$gpu]}; do
        wait $pid
    done
done

echo "All experiments completed." | tee -a "$LOG_FILE"