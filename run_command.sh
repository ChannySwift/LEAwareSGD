#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=7 \
#python train_model.py \
#    --dataset OfficeHome \
#    --num_class 65 \
#    --seen_index 3 \
#    --network resnet18 \
#    --algorithm LEAware \
#    --train_epochs 50 \
#    --save_path ./Experiment/Domain

#CUDA_VISIBLE_DEVICES=7 \
#python train_model.py \
#    --dataset VLCS \
#    --num_class 5 \
#    --seen_index 1 \
#    --network resnet18 \
#    --algorithm LEAware \
#    --train_epochs 50 \
#    --save_path ./Experiment/Domain

#CUDA_VISIBLE_DEVICES=7 \
#python train_model.py \
#    --dataset DomainNet \
#    --num_class 345 \
#    --seen_index 0 \
#    --batch_size 128 \
#    --network resnet18 \
#    --algorithm LEAware \
#    --train_epochs 2 \
#    --save_path ./Experiment/Domain

CUDA_VISIBLE_DEVICES=3 \
python train_model.py \
    --dataset TerraIncognita \
    --num_class 10 \
    --seen_index 0 \
    --batch_size 32 \
    --network resnet18 \
    --algorithm LEAware \
    --train_epochs 50 \
    --save_path ./Experiment/Domain

#CUDA_VISIBLE_DEVICES=7 \
#python train_model.py \
#    --dataset Fundus \
#    --num_class 5 \
#    --seen_index 3 \
#    --batch_size 32 \
#    --network resnet50 \
#    --algorithm LEAware \
#    --train_epochs 100 \
#    --save_path ./Experiment/Domain

#CUDA_VISIBLE_DEVICES=7 \
#python train_model.py \
#    --dataset PACS \
#    --num_class 7 \
#    --seen_index 2 \
#    --batch_size 32 \
#    --network resnet18 \
#    --algorithm LEAware \
#    --train_epochs 50 \
#    --save_path ./Experiment/Domain