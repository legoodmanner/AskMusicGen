#!/bin/bash

for i in {0..19}
do
    python3 experiment.py configs/musicgen_GS_tempo_feature.yaml  --layer $i
done
