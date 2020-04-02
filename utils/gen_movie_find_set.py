from scipy.io import loadmat
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = "serif"

import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 22})
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class StatisticAnalysis:
    def __init__(self, data_root, SAVEDATA_FOLDER,map_setup, exp_setup, trained_num_agent, list_testing_num_agent):
        self.DATA_FOLDER = data_root
        self.SAVEDATA_FOLDER = SAVEDATA_FOLDER
        self.map_setup = map_setup
        self.exp_setup = exp_setup
        self.trained_num_agent = trained_num_agent
        self.list_testing_num_agent = list_testing_num_agent

        self.load_data()

    def load_data(self):
        data = {
            'dcp': {},
            'dcpOE': {},
            'rdcp': {},
            'rdcpOE': {},
        }
        data_list = []

        for data_type in data.keys():

            for subdir, dirs, files in os.walk(os.path.join(self.DATA_FOLDER, data_type,self.map_setup)):
                for file in files:
                    # print os.path.join(subdir, file)
                    filepath = subdir + os.sep + file

                    if filepath.endswith(".mat"):
                        # print(subdir, file)
                        mat_data = loadmat(filepath)


                        hidden_state = mat_data['hidden_state'][0][0]
                        num_agents_trained = mat_data['num_agents_trained'][0][0]
                        num_agents_testing = mat_data['num_agents_testing'][0][0]

                        K = mat_data['K'][0][0]

                        cleaned_data = {
                            'filename': file,

                            'type': data_type,
                            'exp_stamps': mat_data['exp_stamps'][0],
                            'rate_ReachGoal':mat_data['rate_ReachGoal'][0][0],

                            'map_density_trained': mat_data['map_density_trained'][0][0],
                            'num_agents_trained': mat_data['num_agents_trained'][0][0],

                            'map_size_testing': mat_data['map_size_testing'][0],
                            'map_density_testing': mat_data['map_density_testing'][0][0],
                            'num_agents_testing': mat_data['num_agents_testing'][0][0],

                            'K': K,
                            'hidden_state': hidden_state,

                            'map_size_trained': mat_data['map_size_trained'][0],
                            'list_reachGoal': mat_data['list_reachGoal'][0],
                            'list_noReachGoalSH': mat_data['list_noReachGoalSH'][0],

                        }
                        data_list.append(cleaned_data)
                        data[data_type].setdefault(num_agents_trained, {}).setdefault(num_agents_testing, []).append(
                            cleaned_data)

        self.data_list = data_list
        self.data = data


    def summary_data(self):
        summary_ReachGoal = {}
        summary_noReachGoal = {}
        for index, testing_num_agent in enumerate(self.list_testing_num_agent):
            for id_exp in range(len(self.exp_setup)):
                # print(testing_num_agent)

                label_set1 = self.exp_setup[id_exp][0]
                label_set1_type = label_set1.split(' ')[0].lower()
                label_exp = self.exp_setup[id_exp][1]


                label_set1_K = int(label_exp.split('K')[1].split('_HS')[0])
                label_set1_HS = int(label_exp.split('_HS')[1])

                data_label = '{}_{}'.format(label_set1_type,label_exp)

                label_stamp = str(self.exp_setup[id_exp][2])


                searched_results_set1 = [item for item in self.data_list
                                    if item['num_agents_trained'] == self.trained_num_agent
                                    and item['num_agents_testing'] == testing_num_agent
                                    and item['exp_stamps'] == label_stamp
                                    and item['type'].lower() == label_set1_type
                                    and item['K'] == label_set1_K
                                    and item['hidden_state'] == label_set1_HS
                                    ]


                if len(searched_results_set1) == 0:
                    pass
                else:
                    summary_list_reachGoal = np.array(searched_results_set1[0]['list_reachGoal'])
                    summary_list_reachGoal_index = summary_list_reachGoal.nonzero()[0].tolist()
                    summary_list_noreachGoal_index = np.where(summary_list_reachGoal==0)[0].tolist()
                    # print(self.exp_setup[id_exp],'\n', summary_list_reachGoal)
                    summary_ReachGoal.update({data_label: set(sorted(summary_list_reachGoal_index))})
                    summary_noReachGoal.update({data_label: set(sorted(summary_list_noreachGoal_index))})

        return summary_ReachGoal, summary_noReachGoal



if __name__ == '__main__':

    trained_num_agent = 10
    list_testing_num_agent = [10]


    label_exp_setup = "ImpactK"
    label_exp = 'GNNOE'


    map_setup = 'map20x20_rho1_10Agent'
    exp_setup = [('dcp', 'K1_HS0', '1582029525'),
                 ('dcp', 'K2_HS0', '1582028194'),
                 ('dcp', 'K3_HS0', '1582028876'),
                 ('dcpOE', 'K2_HS0', '1582314635'),
                 ('dcpOE', 'K3_HS0', '1582034757'),
                 ]

    #####################################################################################
    #####################################################################################

    title_text = "{}_{}".format(label_exp, label_exp_setup)

    # DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics_BMAP/'
    DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics_generalization/'


    title_text = "{}_TR_{}".format(title_text, trained_num_agent)

    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)
    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    ResultAnalysis = StatisticAnalysis(DATA_FOLDER, SAVEDATA_FOLDER, map_setup, exp_setup, trained_num_agent, list_testing_num_agent)
    Statistic_ReachGoal, Statistic_notReachGoal  = ResultAnalysis.summary_data()

    # print(Statistic_ReachGoal)


    # index_summary = Statistic_ReachGoal['dcp_K1_HS0'] & Statistic_ReachGoal['dcp_K2_HS0'] & Statistic_ReachGoal['dcp_K3_HS0'] & Statistic_ReachGoal['dcpoe_K2_HS0'] & Statistic_ReachGoal['dcpoe_K3_HS0']

    # index_summary = Statistic_notReachGoal['dcp_K1_HS0'] & Statistic_ReachGoal['dcp_K2_HS0'] & Statistic_ReachGoal['dcp_K3_HS0'] \
    #           & Statistic_ReachGoal['dcpoe_K2_HS0'] & Statistic_ReachGoal['dcpoe_K3_HS0']

    # index_summary = Statistic_notReachGoal['dcp_K1_HS0'] & Statistic_notReachGoal['dcp_K2_HS0'] & Statistic_ReachGoal[
    #             'dcp_K3_HS0'] \
    #           & Statistic_notReachGoal['dcpoe_K2_HS0'] & Statistic_ReachGoal['dcpoe_K3_HS0']
    #
    # index_summary = Statistic_notReachGoal['dcp_K1_HS0'] & Statistic_notReachGoal['dcp_K2_HS0'] & Statistic_notReachGoal['dcp_K3_HS0'] \
    #           & Statistic_ReachGoal['dcpoe_K2_HS0'] & Statistic_ReachGoal['dcpoe_K3_HS0']

    index_summary = Statistic_notReachGoal['dcp_K1_HS0'] & Statistic_notReachGoal['dcp_K2_HS0'] & Statistic_notReachGoal[
        'dcp_K3_HS0'] \
              & Statistic_notReachGoal['dcpoe_K2_HS0'] & Statistic_ReachGoal['dcpoe_K3_HS0']
    print(index_summary)
    print(len(index_summary))
