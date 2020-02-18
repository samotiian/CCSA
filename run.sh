#!/usr/bin/env bash

for SAMPLE_PER_CLASS in 1 3 5 7
do
    for REPETITION in 0 1 2 3 4 5 6 8 9
    do
        python main.py --sample_per_class $SAMPLE_PER_CLASS --repetition $REPETITION --gpu_id 1
    done
done