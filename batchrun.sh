#!/bin/bash

for i in {0..50}  #include last number
do
    python3 experiment.py configs/MusicGenL_MTG_genre_feature.yaml  --layer $i
done
