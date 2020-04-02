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
#
## 20 x 20 world density 0.1 with 10 agents

python CasesSolver.py  --random_map --gen_CasePool --chosen_solver ECBS --map_width 20 --map_density 0.0  --num_agents 10 --num_dataset 30000 --num_caseSetup_pEnv 50
#

python CasesSolver.py  --random_map --gen_CasePool --chosen_solver ECBS --map_width 20 --map_density 0.0  --num_agents 10 --num_dataset 30000 --num_caseSetup_pEnv 50


#### load benchmarking map
#

python CasesSolver.py   --gen_CasePool --chosen_solver ECBS  --num_agents 10 --num_caseSetup_pEnv 5000  --loadmap_TYPE random --path_save /local/scratch/ql295/Data/MultiAgentDataset/Solution_BMap

