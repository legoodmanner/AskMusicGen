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
THREAD_COUNT=4   # Number of threads

# Calculate range of layers for each thread
LAYERS_PER_THREAD=$((TOTAL_LAYERS / THREAD_COUNT))
REMAINDER=$((TOTAL_LAYERS % THREAD_COUNT))

# Run each thread
for thread in $(seq 0 $((THREAD_COUNT - 1))); do
    START=$((thread * LAYERS_PER_THREAD))
    END=$((START + LAYERS_PER_THREAD - 1))
    if [ $thread -eq $((THREAD_COUNT - 1)) ]; then
        END=$((END + REMAINDER))  # Add the remainder to the last thread
    fi

    # Run the thread in background
    (
        for i in $(seq $START $END); do
            echo "Running layer $i on thread $thread"
            ##### THE COMMAND YOU WANT TO RUN #####
            python experiment.py configs/MusicGenS_GS_tempo_feature.yaml --layer $i
        done
    ) &
done

# Wait for all threads to finish
wait

echo "All experiments completed!"