#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

## Create dev_r
python create_dev_r.py

## Create dev_s
python create_dev_s.py

## Generate numbers
python compute_average_scores.py
