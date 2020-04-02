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




###############################################################################################################

######################################               Best K          		      #############################

###############################################################################################################

#
################## trained on 10 agents

# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028876     --nGraphFilterTaps 3 --map_w 20  --num_agents 10  --trained_num_agents 10 --trained_map_w 20

# testing
# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028876    --nGraphFilterTaps 3   --map_w 20 --num_agents 12  --trained_num_agents 10 --trained_map_w 20

# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028876    --nGraphFilterTaps 3   --map_w 28 --num_agents 20  --trained_num_agents 10 --trained_map_w 20 --num_testset 500


# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028876    --nGraphFilterTaps 3   --map_w 40 --num_agents 40  --trained_num_agents 10 --trained_map_w 20 --num_testset 200 --rate_maxstep 3

#### test on BMAP

#python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876   --nGraphFilterTaps 3   --map_w 32 --num_agents 40 --map_type random --trained_num_agents 10 --trained_map_w 20 --num_testset 200 --rate_maxstep 3



################## trained on 12 agents

# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028411     --nGraphFilterTaps 3   --map_w 20 --num_agents 12  --trained_num_agents 12 --trained_map_w 20


#####  larger scale
# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028411     --nGraphFilterTaps 3   --map_w 28 --num_agents 20  --trained_num_agents 12 --trained_map_w 20 --num_testset 500


# python main.py configs/dcp_ECBS.json --mode test --log_anime  --best_epoch --test_general --log_time_trained 1582028411     --nGraphFilterTaps 3   --map_w 40 --num_agents 40  --trained_num_agents 12 --trained_map_w 20 --num_testset 200

#### test on BMAP

#python main.py configs/dcp_ECBS_BMAP.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028411    --nGraphFilterTaps 3   --map_w 32 --num_agents 40 --map_type random --trained_num_agents 12 --trained_map_w 20 --num_testset 200

