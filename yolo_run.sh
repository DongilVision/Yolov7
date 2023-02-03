#!/bin/bash
python3 /content/parse_config.py
bash /data/train.sh 
chmod -R 777 result
