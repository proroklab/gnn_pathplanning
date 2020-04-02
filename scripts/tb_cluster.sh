#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1

#MODEL_DIR = /local/scratch/ql295/Data/MultiAgentDataset/Tensorboard
#let ipnport=($UID-6025)%65274
#echo ipnport=$ipnport
#ipnip=$(hostname -i)
#echo ipnip=$ipnip
#tensorboard --logdir="${MODEL_DIR}" --port=$ipnport


tensorboard --logdir /local/scratch/ql295/Data/MultiAgentDataset/Tensorboard_CG50 --port 16050