#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES= 0, 1, 2

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


#####################################################################
#                                                                   #
#                  Control Group 20 - CG 20                         #
#                   (relative coordination)                         #
#                                                                   #
#####################################################################


################## K = 1

python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525   --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 6

################## K = 2
python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028194    --nGraphFilterTaps 2  --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 6

################## K = 2
#python main.py configs/dcp_onlineExpert_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635   --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 6


################## K = 3
python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876   --nGraphFilterTaps  3 --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 6

python main.py configs/dcp_onlineExpert_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757   --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 6


################## K = 1

python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525   --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 20

################## K = 2
python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028194    --nGraphFilterTaps 2  --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 20

################## K = 2
#python main.py configs/dcp_onlineExpert_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635   --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 20


################## K = 3
python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876   --nGraphFilterTaps  3 --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 20

python main.py configs/dcp_onlineExpert_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757   --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --map_type demo --trained_num_agents 10 --trained_map_w 20 --num_testset 5 --rate_maxstep 3 --commR 20
