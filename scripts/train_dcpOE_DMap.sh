#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
#python3 main.py PATH_OF_THE_CONFIG_FILE

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES= 0, 1, 2

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


#####################################################################
#                                                                   #
#                  Control Group 40 - CG 40                         #
#               (relative coordination + DMAP )                     #
#                                                                   #
#####################################################################

################## K = 1

# ## 10 agents - training
# python main.py configs/dcp_onlineExpert.json --mode train  --map_w 20 --nGraphFilterTaps 1  --num_agents 10  --trained_num_agents 10


# ## 12 agents - training
# python main.py configs/dcp_onlineExpert.json --mode train  --map_w 20 --nGraphFilterTaps 1  --num_agents 12  --trained_num_agents 12

################## K = 2

#
# ## 10 agents - training
# python main.py configs/dcp_onlineExpert.json --mode train  --map_w 20 --nGraphFilterTaps 2  --num_agents 10  --trained_num_agents 10


# ## 12 agents - training
# python main.py configs/dcp_onlineExpert.json --mode train  --map_w 20 --nGraphFilterTaps 2  --num_agents 12  --trained_num_agents 12



# ################## K = 3


# ## 10 agents - training
# python main.py configs/dcp_onlineExpert.json --mode train  --map_w 20 --nGraphFilterTaps 3  --num_agents 10  --trained_num_agents 10

# ## 12 agents - training
# python main.py configs/dcp_onlineExpert.json --mode train  --map_w 20 --nGraphFilterTaps 3  --num_agents 12  --trained_num_agents 12

