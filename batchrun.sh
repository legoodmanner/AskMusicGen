#!/bin/bash

for i in {7..9}  #include last number
do
    python3 experiment.py configs/musicgen_GS_tempo_feature.yaml  --layer $i
done
