from scipy.io import loadmat
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = "serif"

import matplotlib.ticker as ticker
import plotly.graph_objects as go
plt.rcParams.update({'font.size': 22})
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class StatisticAnalysis:
    def __init__(self, data_root, SAVEDATA_FOLDER, exp_setup,  trained_num_agent, list_testing_num_agent):
        self.DATA_FOLDER = data_root
        self.SAVEDATA_FOLDER = SAVEDATA_FOLDER
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

            for subdir, dirs, files in os.walk(os.path.join(self.DATA_FOLDER, data_type)):
                for file in files:
                    # print os.path.join(subdir, file)
                    filepath = subdir + os.sep + file

                    if filepath.endswith(".mat"):
                        # print(subdir, file)
                        mat_data = loadmat(filepath)

                        rate_ReachGoal = mat_data['rate_ReachGoal'][0][0]
                        mean_deltaFT = mat_data['mean_deltaFT'][0][0]
                        mean_deltaMP = mat_data['mean_deltaMP'][0][0]
                        hidden_state = mat_data['hidden_state'][0][0]
                        num_agents_trained = mat_data['num_agents_trained'][0][0]
                        num_agents_testing = mat_data['num_agents_testing'][0][0]

                        K = mat_data['K'][0][0]

                        cleaned_data = {
                            'filename': file,

                            'type': data_type,
                            'exp_stamps': mat_data['exp_stamps'][0],

                            'map_size_trained': mat_data['map_size_trained'][0],
                            'map_density_trained': mat_data['map_density_trained'][0][0],
                            'num_agents_trained': mat_data['num_agents_trained'][0][0],

                            'map_size_testing': mat_data['map_size_testing'][0],
                            'map_density_testing': mat_data['map_density_testing'][0][0],
                            'num_agents_testing': mat_data['num_agents_testing'][0][0],

                            'K': K,
                            'hidden_state': hidden_state,
                            'rate_ReachGoal': rate_ReachGoal,
                            'mean_deltaFT': mean_deltaFT,
                            'std_deltaMP': mat_data['std_deltaMP'][0][0],
                            'mean_deltaMP': mean_deltaMP,
                            'std_deltaFT': mat_data['std_deltaFT'][0][0],

                            'list_numAgentReachGoal': mat_data['list_numAgentReachGoal'][0],
                            'hist_numAgentReachGoal': mat_data['hist_numAgentReachGoal'][0],
                        }
                        data_list.append(cleaned_data)
                        data[data_type].setdefault(num_agents_trained, {}).setdefault(num_agents_testing, []).append(
                            cleaned_data)

        self.data_list = data_list
        self.data = data
        # print(len(data_list))
        # return data

    def plot_hist_data(self, title_setup, text_legend):
        for index, testing_num_agent in enumerate(self.list_testing_num_agent):
            print(testing_num_agent)
            title_text = "{}_TE{}".format(title_setup, testing_num_agent)

            label_set1 = self.exp_setup[0]
            label_set1_type = label_set1.split(' ')[0].lower()
            label_set1_K = int(label_set1.split('K')[1].split('-HS')[0])
            label_set1_HS = int(label_set1.split('-HS')[1])



            searched_results_set1 = [item for item in self.data_list
                                if item['num_agents_trained'] == self.trained_num_agent
                                and item['num_agents_testing'] == testing_num_agent
                                and item['type'].lower() == label_set1_type
                                and item['K'] == label_set1_K
                                and item['hidden_state'] == label_set1_HS
                                ]

            label_set2 = self.exp_setup[1]
            label_set2_type = label_set2.split(' ')[0].lower()
            label_set2_K = int(label_set2.split('K')[1].split('-HS')[0])
            label_set2_HS = int(label_set2.split('-HS')[1])


            searched_results_set2 = [item for item in self.data_list
                                     if item['num_agents_trained'] == self.trained_num_agent
                                     and item['num_agents_testing'] == testing_num_agent
                                     and item['type'].lower() == label_set2_type
                                     and item['K'] == label_set2_K
                                     and item['hidden_state'] == label_set2_HS
                                     ]

            if len(searched_results_set1) == 0:
                pass
            else:
                hist_numAgentReachGoal_set1 = searched_results_set1[0]['hist_numAgentReachGoal']

                print(label_set1, hist_numAgentReachGoal_set1)
                hist_numAgentReachGoal_set2 = searched_results_set2[0]['hist_numAgentReachGoal']
                print(label_set2, hist_numAgentReachGoal_set2)

                total_num_cases = sum(hist_numAgentReachGoal_set1)
                hist_numAgentReachGoal_norm_set1 = []
                hist_numAgentReachGoal_norm_set2 = []
                list_numAgents = []

                for index in range(len(hist_numAgentReachGoal_set1)):
                    list_numAgents.append(str(index))
                    hist_numAgentReachGoal_norm_set1.append(hist_numAgentReachGoal_set1[index]/total_num_cases)
                    hist_numAgentReachGoal_norm_set2.append(hist_numAgentReachGoal_set2[index]/total_num_cases)

                self.plot_figure(testing_num_agent, list_numAgents, total_num_cases, hist_numAgentReachGoal_norm_set1, hist_numAgentReachGoal_norm_set2, label_set1_K, title_text, text_legend)
                pass



    def plot_figure(self, testing_num_agent, list_numAgents, total_num_cases, hist_data_set1, hist_data_set2, label_set1_K, title_text, text_legend, use_log_scale=False):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 6)
        # title_exp_setup = ('trained on {} agents and tested on {} agents'.format(self.trained_num_agent, testing_num_agent))
        # self.title_text = 'Histogram of percentage (# agents reach goal among {} cases) \n in network is {}.'.format(total_num_cases, title_exp_setup)
        #
        # self.ax.set_title(self.title_text)
        self.ax.set_xlabel('# robots')

        width = 0.35  # the width of the bars
        label_width = 1.05
        if len(list_numAgents)<20 and label_set1_K == 2:
            step_size = 2
            self.ax.set_ylabel('Proportion of cases'.format(total_num_cases))
        else:
            step_size = 4

        label_pos = np.arange(len(list_numAgents))
        # rects1 = self.ax.bar(x - label_width / 2 + width * 1, hist_numAgentReachGoal, width, label=text_legend)

        hist_set1 = self.ax.bar(label_pos, hist_data_set1, align='center', label='{}'.format(text_legend[0]), ls='dotted', lw=3, fc=(0, 0, 1, 0.5))
        hist_set2 = self.ax.bar(label_pos, hist_data_set2, align='center', label='{}'.format(text_legend[1]),lw=3, fc=(1, 0, 0, 0.5))

        start, end = self.ax.get_xlim()
        self.ax.xaxis.set_ticks(np.arange(0,len(list_numAgents), step_size))
        # plt.xticks(label_pos)
        # self.ax.set_xticklabels(label_pos)
        # self.autolabel(rects1)

        if use_log_scale:
            self.ax.set_yscale('log')

        self.ax.legend()
        # plt.grid()
        plt.show()

        self.save_fig(title_text)

    def show(self):
        plt.show()

    def save_fig(self, title):
        # name_save_fig = os.path.join(self.SAVEDATA_FOLDER, "{}_{}.pdf".format(self.title_text, title))
        name_save_fig = os.path.join(self.SAVEDATA_FOLDER, "{}.jpg".format(title))
        name_save_fig_pdf = os.path.join(self.SAVEDATA_FOLDER, "{}.pdf".format(title))
        self.fig.savefig(name_save_fig, bbox_inches='tight', pad_inches=0)
        self.fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)


    def autolabel(self, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height in [0.7558, 0.7596]:
                self.ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(-6, 15),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=0, fontweight='bold')
                continue
            self.ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(-6, 15),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)

if __name__ == '__main__':
    #
    # trained_num_agent = 8
    # list_testing_num_agent = [8, 12, 16, 32]

    trained_num_agent = 10
    list_testing_num_agent = [40]
    # # list_testing_num_agent = [10, 40]
    # list_testing_num_agent = [14, 20, 40]

    #
    # trained_num_agent = 12
    # list_testing_num_agent = [12, 14, 20, 40]

    #####################################################################################
    #####################################################################################
    # label_exp_setup = "ImpactK"
    # label_exp = 'GNN'
    # select_label = ["DCP - K2-HS0", "DCP - K3-HS0"]
    # text_legend = [
    #      "GNN - K=2", "GNN - K=3"
    # ]

    # label_exp_setup = "ImpactK"
    # label_exp = 'GNNOE'
    # select_label = ["DCPOE - K2-HS0", "DCPOE - K3-HS0"]
    # text_legend = [
    #     "GNN(OE) - K=2", "GNN(OE) - K=3"
    # ]

    label_exp_setup = "ImpactK"
    label_exp = 'GNNOE'
    select_label = ["DCPOE - K2-HS0", "DCPOE - K3-HS0"]
    text_legend = [
        "GNN - K=2", "GNN - K=3"
    ]


    #####################################################################################

    # label_exp_setup = "ImpactOE"
    # label_exp = 'K2'
    # select_label = ["DCP - K2-HS0", "DCPOE - K2-HS0"]
    # text_legend = [
    #      "GNN - K=2",  "GNN(OE) - K=2"
    # ]

    #
    # label_exp_setup = "ImpactOE"
    # label_exp = 'K3'
    # select_label = ["DCP - K3-HS0", "DCPOE - K3-HS0"]
    # text_legend = [
    #     "GNN - K=3", "GNN(OE) - K=3"
    # ]

    #####################################################################################
    #####################################################################################

    title_text = "{}_{}".format(label_exp, label_exp_setup)

    # DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics_BMAP/'
    DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics_generalization/'
    epoch_text = "IROS"


    title_text = "{}_TR_{}".format(title_text, trained_num_agent)

    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)
    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    ResultAnalysis = StatisticAnalysis(DATA_FOLDER, SAVEDATA_FOLDER, select_label, trained_num_agent, list_testing_num_agent)
    ResultAnalysis.plot_hist_data(title_text, text_legend)
