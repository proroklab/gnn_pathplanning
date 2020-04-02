#!/usr/bin/env python3
import yaml
import matplotlib
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import animation
from matplotlib import lines
import matplotlib.animation as manimation
import argparse
import math
import gc
import seaborn as sns
import time
import scipy.io as sio
import sys
from easydict import EasyDict
np.set_printoptions(threshold=np.inf)
import os

from utils.visualize import Animation




if __name__ == "__main__":
    DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/AnimeDemo/'

    ##################
    #### Case 4 - maze from Liu - 20A

    map_setup = 'map20x20_rho1_10Agent'

    # map_setup = 'demo20x20_rho1_10Agent'
    exp_setup =[('dcp','K1_HS0','1582029525'),
                ('dcp', 'K2_HS0', '1582028194'),
                ('dcp', 'K3_HS0', '1582028876'),
                ('dcpOE', 'K2_HS0', '1582314635'),
                ('dcpOE','K3_HS0','1582034757'),
                ]

    # selected_case = [([1, 1, 1, 1, 1], [4099, 31]), # 1015
    #                  ([0, 1, 1, 1, 1], [1, 23]), # 1546
    #                  ([0, 0, 1, 0, 1], [935, 3206]), # 58
    #                  ([0, 0, 0, 1, 1], [4097, 2052]), # 175
    #                  ([0, 0, 0, 0, 1], [3093, 4388]),#74
    #                  ]
    selected_case = [
                     ([0, 0, 0, 0, 1], [3093, 4388]),  # 74
                     ]
    Setup_comR = 'commR_5'
    Id_agent = 0
    num_exp = len(exp_setup)

    for id_mod in range(len(selected_case)):
        list_record = selected_case[id_mod][0]
        list_id_case = selected_case[id_mod][1]
        for id_exp in range(num_exp):
            Setup = '{}/{}/{}/TR_M20p1_10Agent/{}/'.format(exp_setup[id_exp][0],map_setup,exp_setup[id_exp][1],exp_setup[id_exp][2])

            network_label = str(exp_setup[id_exp][1])
            print(network_label)
            K = int(network_label.split('K')[-1].split('_')[0])


            Data_path = os.path.join(DATA_FOLDER, Setup, Setup_comR)

            for Id_case in list_id_case:
                print(id_exp, list_record, list_record[id_exp])
                if list_record[id_exp]:
                    File_name = 'successCases_ID{:05d}'.format(Id_case)
                    Path_sol = os.path.join(Data_path, 'predict_success', '{}.yaml'.format(File_name))
                else:
                    File_name = 'failureCases_ID{:05d}'.format(Id_case)
                    Path_sol = os.path.join(Data_path, 'predict_failure', '{}.yaml'.format(File_name))

                Path_map = os.path.join(Data_path, 'input', '{}.yaml'.format(File_name))
                Path_GSO = os.path.join(Data_path, 'GSO','{}.mat'.format(File_name))

                Path_video = os.path.join(DATA_FOLDER, 'video',map_setup, 'Case{}'.format(Id_case))
                # print(Path_map)
                # print(Path_sol)
                # print()
                # print()
                try:
                    # Create target Directory
                    os.makedirs(Path_video)
                except FileExistsError:
                    pass
                # print(Path_video)
                if list_record[id_exp]:
                    Name_video = '{}/{}_K{}_{}_IDcase{}_{}_success.mp4'.format(Path_video, exp_setup[id_exp][0], K, Setup_comR, Id_case, Id_agent)
                else:
                    Name_video = '{}/{}_K{}_{}_IDcase{}_{}_failure.mp4'.format(Path_video, exp_setup[id_exp][0], K, Setup_comR, Id_case, Id_agent)
                # print(Name_video)
                config = {'map': Path_map,
                          'schedule': Path_sol,
                          'GSO': Path_GSO,
                          'nGraphFilterTaps': K,
                          'id_chosenAgent': Id_agent,
                          'video': Name_video,
                          'speed': 2,
                          }
                config_setup = EasyDict(config)

                animation = Animation(config_setup)

                # animation.show()
                if config_setup.video:
                    print(config_setup.video)
                    animation.save(config_setup.video, config_setup.speed)
                    print('Movie generation finished.')
                else:
                    animation.show()

            time.sleep(60)