#!/bin/bash

for i in {7..9}  #include last number
do
    python3 experiment.py configs/musgenMed_MTG_genre_feature.yaml  --layer $i
done
