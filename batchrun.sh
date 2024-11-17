#!/bin/bash

for i in {1..20}
do
    python3 experiment.py configs/VampC_MTG_genre_feature.yaml --layer $i
done
