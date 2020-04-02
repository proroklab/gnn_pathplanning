from scipy.io import loadmat
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
# plt.rcParams.update({'font.size': 16})
import pandas as pd
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = "serif"

class StatisticAnalysis:
    def __init__(self, data_root, SAVEDATA_FOLDER, exp_setup, id_stamp,  trained_num_agent, list_testing_num_agent):
        self.DATA_FOLDER = data_root
        self.SAVEDATA_FOLDER = SAVEDATA_FOLDER
        self.exp_setup = exp_setup
        self.id_stamp = id_stamp[0]
        self.trained_num_agent = trained_num_agent
        self.list_testing_num_agent = list_testing_num_agent
        self.load_data()

    def load_data(self):
        data = {
            'dcp': {},
            'rdcp': {},
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

    def plot_hist_data(self, title_setup):
        for index, testing_num_agent in enumerate(self.list_testing_num_agent):
            title_text = "{}_TE{}".format(title_setup, testing_num_agent)

            searched_results = [item for item in self.data_list
                                if item['num_agents_trained'] == self.trained_num_agent
                                and item['num_agents_testing'] == testing_num_agent
                                and item['exp_stamps'] == str(self.id_stamp)
                                ]
            if len(searched_results) == 0:
                pass
            else:
                hist_numAgentReachGoal = searched_results[0]['hist_numAgentReachGoal']
                total_num_cases = sum(hist_numAgentReachGoal)
                hist_numAgentReachGoal_norm = []
                list_numAgents = []
                for index in range(len(hist_numAgentReachGoal)):
                    hist_numAgentReachGoal_norm.append(hist_numAgentReachGoal[index]/total_num_cases)
                    list_numAgents.append(str(index))

                self.plot_figure(testing_num_agent, list_numAgents, total_num_cases, hist_numAgentReachGoal_norm, title_text)
                pass



    def plot_figure(self, testing_num_agent, list_numAgents, total_num_cases, hist_numAgentReachGoal, title_text, use_log_scale=False):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 5.5)
        self.title_text = 'Histogram of number of agents reach its goal'

        self.ax.set_title(self.title_text)
        self.ax.set_xlabel('# agents')
        self.ax.set_ylabel('Percentage (#agent reach goal/{})'.format(total_num_cases))
        text_legend = ('Train on {} agents and tested on {} agents'.format(self.trained_num_agent, testing_num_agent))
        width = 0.35  # the width of the bars
        label_width = 1.05
        label_pos = np.arange(len(list_numAgents))
        # rects1 = self.ax.bar(x - label_width / 2 + width * 1, hist_numAgentReachGoal, width, label=text_legend)

        rects1 = self.ax.bar(label_pos, hist_numAgentReachGoal, align='center', label=text_legend)
        plt.xticks(label_pos)
        self.ax.set_xticklabels(label_pos)
        # self.autolabel(rects1)

        if use_log_scale:
            self.ax.set_yscale('log')

        self.ax.legend()
        plt.show()
        self.save_fig(title_text)

    def show(self):
        plt.show()

    def save_fig(self, title):
        # name_save_fig = os.path.join(self.SAVEDATA_FOLDER, "{}_{}.pdf".format(self.title_text, title))
        name_save_fig = os.path.join(self.SAVEDATA_FOLDER, "{}.jpg".format(title))
        self.fig.savefig(name_save_fig)


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

    select_label = "DCP - K2-HS0"
    trained_num_agent = 10
    list_testing_num_agent = [10, 12, 20, 40]
    id_stamp = [1573174674]


    DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics_generalization/'
    epoch_text = "AAMAS"
    title_text = "{}_{}_TR_{}".format(select_label, epoch_text, trained_num_agent)

    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)
    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    ResultAnalysis = StatisticAnalysis(DATA_FOLDER, SAVEDATA_FOLDER, select_label, id_stamp, trained_num_agent, list_testing_num_agent)
    ResultAnalysis.plot_hist_data(title_text)
