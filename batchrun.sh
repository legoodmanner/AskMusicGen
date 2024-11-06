#!/bin/bash

for i in {7..23}
do
    python3 experiment.py configs/musicgenMTG_genre.yaml --layer $i
done
