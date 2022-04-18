#!/usr/bin/python3


CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/eval_example.py --manualSeed 0 --method PARSE --n-labeled 1  --batch-size 8 --alpha 0.25&
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/eval_example.py --manualSeed 0 --method PARSE --n-labeled 3  --batch-size 8 --alpha 0.25;
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/eval_example.py --manualSeed 0 --method PARSE --n-labeled 5  --batch-size 8 --alpha 0.25&
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/eval_example.py --manualSeed 0 --method PARSE --n-labeled 7  --batch-size 8 --alpha 0.25;
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/eval_example.py --manualSeed 0 --method PARSE --n-labeled 10 --batch-size 8 --alpha 0.25&
CUDA_VISIBLE_DEVICES=0 python3 ./PARSE/eval_example.py --manualSeed 0 --method PARSE --n-labeled 25 --batch-size 8 --alpha 0.25;


