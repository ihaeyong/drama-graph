#!/bin/bash
export PYTHONPATH=/testdata/HSE/scripts

python models/train_place.py --img_path ../data/AnotherMissOh_images --json_path ../data/AnotherMissOh_Visual_full.json
