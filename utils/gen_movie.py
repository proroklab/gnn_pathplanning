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

    ############
    #### Case 1 - random - 20A
    # Setup = 'dcpOE/random32x32_rho1_20Agent/K3_HS0/TR_M20p1_10Agent/1582034757/'
    # list_id_case = [7,19,26,27,29,42,48,50,56,61,66,91,102,106,115,117,130,137,156,166,171,172,186,193,197,198]
    # list_id_case = [5]
    # Id_agent = 9
    # K = 1

    ############
    #### Case 2 - random - 40A
    # Setup = 'dcpOE/random32x32_rho1_40Agent/K3_HS0/TR_M20p1_10Agent/1582034757/'
    # Id_case = 132
    # Id_agent = 20

    ############
    #### Case 3 - DMap - 40A
    # Setup = 'dcpOE/map40x40_rho1_40Agent/K3_HS0/TR_M20p1_10Agent/1582034757/'
    # Id_case = 1
    # Id_agent = 26

    ##################
    #### Case 4 - maze from Liu - 20A

    # Setup = 'dcpOE/maze40x40_rho1_32Agent/K3_HS0/TR_M20p1_10Agent/1582034757/'
    # list_id_case = [1]
    # Id_agent = 0
    # K = 3
    # # Setup_comR = 'commR_5'
    # # Setup_comR = 'commR_11'
    # choose_sucess = False

    # Setup = 'dcpOE/random40x40_rho1_64Agent/K3_HS0/TR_M20p1_10Agent/1582034757/'
    # list_id_case = [0]
    # Id_agent = 0
    # K = 3
    # # Setup_comR = 'commR_5'
    # Setup_comR = 'commR_11'
    # choose_sucess = False

    ########################
    ########################
    #### Case 5-1 - Dmap - 20A

    Setup = 'dcp/map20x20_rho1_10Agent/K1_HS0/TR_M20p1_10Agent/1582029525'

    Id_agent = 0
    K = 1
    Setup_comR = 'commR_5'
    list_id_case = [11]
    choose_sucess = True
    # list_id_case = [1]
    # choose_sucess = False


    # Setup = 'dcp/map20x20_rho1_10Agent/K2_HS0/TR_M20p1_10Agent/1582028194'
    #
    # Id_agent = 0
    # K = 2
    #
    # Setup_comR = 'commR_5'

    # choose_sucess = True
    # list_id_case = [1]

    # choose_sucess = True
    # list_id_case = [11]

    # Setup = 'dcp/map20x20_rho1_10Agent/K3_HS0/TR_M20p1_10Agent/1582028876'
    # Id_agent = 0
    # K = 3
    # Setup_comR = 'commR_5'

    #
    # choose_sucess = True
    # list_id_case = [1]

    # choose_sucess = True
    # list_id_case = [11]

    # #### Case 5-1 - Dmap - 20A
    #
    # Setup = 'dcpOE/map20x20_rho1_10Agent/K2_HS0/TR_M20p1_10Agent/1582314635'
    #
    # Id_agent = 0
    # K = 2
    # Setup_comR = 'commR_5'
    # choose_sucess = False
    # list_id_case = [1]

    # choose_sucess = True
    # list_id_case = [11]
    #
    # #### Case 5-1 - Dmap - 20A
    #
    # Setup = 'dcpOE/map20x20_rho1_10Agent/K3_HS0/TR_M20p1_10Agent/1582034757'
    # K = 3
    # Id_agent = 0
    # Setup_comR = 'commR_5'
    # # choose_sucess = True
    # # list_id_case = [1]
    #
    # choose_sucess = True
    # list_id_case = [11]



    Data_path = os.path.join(DATA_FOLDER, Setup, Setup_comR)

    for Id_case in list_id_case:

        if choose_sucess:
            File_name = 'successCases_ID{:05d}'.format(Id_case)
            Path_sol = os.path.join(Data_path, 'predict_success', '{}.yaml'.format(File_name))
        else:
            File_name = 'failureCases_ID{:05d}'.format(Id_case)
            Path_sol = os.path.join(Data_path, 'predict_failure', '{}.yaml'.format(File_name))

        Path_map = os.path.join(Data_path, 'input', '{}.yaml'.format(File_name))
        Path_GSO = os.path.join(Data_path, 'GSO','{}.mat'.format(File_name))
        Path_video = os.path.join(Data_path, 'video')
        print(Path_map)
        print(Path_sol)
        # print()
        # print()
        try:
            # Create target Directory
            os.makedirs(Path_video)
        except FileExistsError:
            pass
        # print(Path_video)
        if choose_sucess:
            Name_video = '{}/{}_Case{}_K{}_{}_success.mp4'.format(Path_video, Setup_comR, Id_case, K, Id_agent)
        else:
            Name_video = '{}/{}_Case{}_K{}_{}_failure.mp4'.format(Path_video, Setup_comR, Id_case, K, Id_agent)
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

