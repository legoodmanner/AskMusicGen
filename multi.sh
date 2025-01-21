#!/bin/bash

# Function to handle SIGINT
cleanup() {
    echo "Terminating all background processes..."
    pkill -P $$  # Kill all child processes of this script
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Total number of layers
TOTAL_LAYERS=24  # Change this to the total number of layers you have
THREAD_COUNT=2   # Number of threads
STEP=1          # Step size for layers

# Calculate total ranges and distribute ranges among threads
RANGES=()
for i in $(seq 0 $STEP $((TOTAL_LAYERS - 1))); do
    RANGES+=("$i-$((i + STEP - 1))")
done

# Ensure no range exceeds the total number of layers
for i in "${!RANGES[@]}"; do
    IFS="-" read -r START END <<<"${RANGES[$i]}"
    if [ $END -ge $TOTAL_LAYERS ]; then
        RANGES[$i]="$START-$((TOTAL_LAYERS - 1))"
    fi
done

# Assign ranges to threads
RANGES_PER_THREAD=$(( (${#RANGES[@]} + THREAD_COUNT - 1) / THREAD_COUNT ))

# Run each thread
for thread in $(seq 0 $((THREAD_COUNT - 1))); do
    # Calculate the start and end index for this thread's ranges
    START_INDEX=$((thread * RANGES_PER_THREAD))
    END_INDEX=$((START_INDEX + RANGES_PER_THREAD - 1))
    
    # Ensure we do not exceed available ranges
    if [ $END_INDEX -ge ${#RANGES[@]} ]; then
        END_INDEX=$((${#RANGES[@]} - 1))
    fi

    # Run the thread in background
    (
        for index in $(seq $START_INDEX $END_INDEX); do
            IFS="-" read -r START END <<<"${RANGES[$index]}"
            for i in $(seq $START $END); do
                echo "Running layer $i on thread $thread"
                ##### THE COMMAND YOU WANT TO RUN #####
                python3 experiment.py configs/MusicGenS_MTG_genre_feature.yaml --layer $i
            done
        done
    ) &
done

# Wait for all threads to finish
wait

echo "All experiments completed!"
