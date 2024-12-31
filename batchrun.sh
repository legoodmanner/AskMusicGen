#!/bin/bash

for i in {15..35..5}
do
    python3 experiment.py configs/musgenMed_MTG_genre_feature.yaml  --layer $i
done
