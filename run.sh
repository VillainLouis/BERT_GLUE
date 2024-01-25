#!/bin/bash

# eg: python train.py --model_checkpoint /data0/jliu/Models/LLM/bert-base-uncased --lr 2e-5 --task 20news --batch_size 8 --optimizer AdamW --device 3 --test_interval 150 --epoch 10 --logger_name my_logger --log_name test.log

python train.py --model_checkpoint /data0/jliu/Models/LLM/bert-base-uncased --lr 2e-4 --task 20news --batch_size 8 --optimizer AdamW --device 3 --test_interval 150 --epoch 19

python train.py --model_checkpoint /data0/jliu/Models/LLM/bert-base-uncased --lr 2e-4 --task qnli --batch_size 4 --optimizer AdamW --device 3 --test_interval 150 --epoch 10 --logger_name my_logger --log_name test.log
