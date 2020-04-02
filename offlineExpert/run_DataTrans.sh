##!/usr/bin/env bash
#
#
## density 0.0
##--use_savedpair
##--init_id
##making
#
#
#
##------------------------------------- density 0.1
#
## ECBS - suboptimal 1.3

#
python DataGen_Transformer.py  --num_agents 12 --map_w 20 --map_density 0.0  --div_train 21000 --div_valid 200 --div_test 4500
##
#python DataGen_Transformer.py  --num_agents 12 --map_w 20 --map_density 0.1  --div_train 21000 --div_valid 200 --div_test 4500


## ECBS - load benchmarking map
#
python DataGen_Transformer.py --loadmap_TYPE random  --num_agents 20 --map_w 32 --map_density 0.1  --div_train 0 --div_valid 0 --div_test 2 --solCases_dir /local/scratch/ql295/Data/MultiAgentDataset/Solution_BMap --dir_SaveData  /local/scratch/ql295/Data/MultiAgentDataset/DataSource_BMap
#

python DataGen_Transformer.py --loadmap_TYPE demo  --num_agents 10 --map_w 20 --map_density 0.1  --div_train 0 --div_valid 0 --div_test 5 --solCases_dir /local/scratch/ql295/Data/MultiAgentDataset/Solution_BMap --dir_SaveData  /local/scratch/ql295/Data/MultiAgentDataset/DataSource_BMap

