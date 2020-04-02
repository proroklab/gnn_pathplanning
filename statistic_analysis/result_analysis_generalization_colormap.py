from scipy.io import loadmat
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['font.family'] = "serif"
plt.rcParams.update({'font.size': 20})

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class StatisticAnalysis:
    def __init__(self, data_root, list_metrics, labels, list_trained_agents,list_testing_agents,with_onlineExpert):
        self.DATA_FOLDER = data_root
        self.list_metrics = list_metrics
        self.labels = labels
        self.list_trained_agents = list_trained_agents
        self.list_testing_agents = list_testing_agents
        self.with_onlineExpert = with_onlineExpert
        self.load_data(labels[0].split(' ')[0].lower())

    def load_data(self, target_model):
        if target_model == "dcp":
            data = {
                'dcp': {},
            }
        elif target_model == "dcpoe":
            data = {
                'dcpOE': {},
            }
        elif target_model == "rdcpoe":
            data = {
                'rdcpOE': {},
            }
        else:
            data = {
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
                        # tmp = mat_data['num_agents_testing'][0][0]
                        # print(tmp)
                        K = mat_data['K'][0][0]

                        cleaned_data = {
                            'filename': file,
                            'type': data_type,

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

                            'list_MP_predict': mat_data['list_MP_predict'][0],
                            'list_MP_target': mat_data['list_MP_target'][0],
                            'list_FT_predict': mat_data['list_FT_predict'][0],
                            'list_FT_target': mat_data['list_FT_target'][0],

                            'list_numAgentReachGoal': mat_data['list_numAgentReachGoal'][0],
                            'hist_numAgentReachGoal': mat_data['hist_numAgentReachGoal'][0],
                        }
                        data_list.append(cleaned_data)
                        data[data_type].setdefault(num_agents_trained, {}).setdefault(num_agents_testing,[]).append(cleaned_data)
        self.data_list = data_list
        self.data = data
        # print(len(data_list))
        # return data

    def collect_data(self, metrics):

        summary_label_data_text = []

        for index, label in enumerate(self.labels):
            for id_trained_agents in self.list_trained_agents:
                # Read type and K from label
                label_data_text = []
                #
                # print(id_trained_agents)
                label_type = label.split(' ')[0].lower()

                label_K = int(label.split('K')[1].split('-HS')[0])
                label_HS = int(label.split('-HS')[1])

                standard_data = [item for item in self.data_list
                                        if item['K'] == label_K and item['hidden_state'] == label_HS
                                        and item['type'].lower() == label_type
                                        and item['num_agents_trained'] == id_trained_agents
                                        and item['num_agents_testing'] == id_trained_agents
                                        ]
                if len(standard_data) == 0:
                    # print('data missing')
                    # label_data.append(0)
                    label_data_text.append("/")
                    pass
                else:
                    standard_data_metrics = standard_data[0][metrics]
                for id_testing_agents in self.list_testing_agents:
                    # print('\t {}'.format(id_testing_agents))
                    # print(label_type, label_K)
                    searched_results = [item for item in self.data_list
                                        if item['K'] == label_K and item['hidden_state'] == label_HS
                                        and item['type'].lower() == label_type
                                        and item['num_agents_trained'] == id_trained_agents
                                        and item['num_agents_testing'] == id_testing_agents
                                        ]


                    # Report missing data
                    if len(searched_results) == 0:
                        # print('data missing')
                        # label_data.append(0)
                        label_data_text.append("/")
                        pass
                    else:
                        if metrics == 'rate_ReachGoal':
                            label_data_text.append(np.around(searched_results[0][metrics], decimals=4))
                        elif metrics == 'mean_deltaFT':
                            # v1

                            # label_data_text.append(np.around(searched_results[0][metrics], decimals=4))
                            #v1
                            # label_data_text.append(np.around(searched_results[0][metrics]/id_testing_agents, decimals=4))

                            # v2.1
                            # label_data_text.append(np.around(searched_results[0][metrics] + 1, decimals=4))
                            # v2.1
                            # mean_predict_metrics = np.mean(searched_results[0]['list_FT_predict'])
                            # mean_target_metrics = np.mean(searched_results[0]['list_FT_target'])
                            #
                            # label_data_text.append(np.around(mean_predict_metrics/mean_target_metrics, decimals=4))

                            # v2.3
                            mean_predict_metrics = searched_results[0]['list_FT_predict'] #np.mean(searched_results[0]['list_FT_predict'])
                            mean_target_metrics = searched_results[0]['list_FT_target'] #np.mean(searched_results[0]['list_FT_target'])
                            div_FT = np.divide(np.subtract(mean_predict_metrics,mean_target_metrics),mean_target_metrics)
                            label_data_text.append(np.around(np.mean(div_FT), decimals=4))


                        #########################
                        ##### Save data as string
                        #########################
                        # if id_trained_agents == id_testing_agents:
                        #     label_data_text.append("{0:.4f}".format((searched_results[0][metrics])))
                        #
                        # else:
                        #     # TODO origin value - color map - min-max
                        #     # https://seaborn.pydata.org/generated/seaborn.heatmap.html
                        #     ########### Extract the value
                        #     multi_factor = "{0:.4f}".format((searched_results[0][metrics])/standard_data_metrics)
                        #     value_currentmetric = "{0:.4f}".format(searched_results[0][metrics])
                        #     value_standard = "{0:.4f}".format(standard_data_metrics)
                        #     ########### Store the value
                        #     # label_data_text.append("{}\n({}/{})".format(multi_factor,value_currentmetric,value_standard))
                        #     # label_data_text.append("{}".format(multi_factor))
                        #     # label_data_text.append("{}".format(value_currentmetric))

                summary_label_data_text.append(label_data_text)

        return summary_label_data_text

    def summary_result(self, title, save_table=True):
        summary_df = []
        for metrics in self.list_metrics:
            label_data_text = self.collect_data(metrics)
            # print(label_data_text)
            df = self.save_as_table(title, metrics, label_data_text)
            summary_df.append(df)

        fulldata_df = pd.concat(summary_df, keys=self.list_metrics)
        if save_table:
            self.save_data_as_table(fulldata_df, title)
        print(fulldata_df)
        return fulldata_df

    def _color_red_or_green(self, val):
        color = 'red' if val < 0.5 else 'green'
        return 'color: %s' % color

    def highlight_max(self, val):
        '''
        highlight the maximum in a Series yellow.
        '''
        # is_max = s == s.max()
        color = 'red' if val < 0.5 else 'green'
        return ['background-color: {}'.format(color)]


    def save_as_table(self, title, metrics, label_data_text):

        df = pd.DataFrame(np.asarray(label_data_text), index=self.list_trained_agents, columns=self.list_testing_agents)
        df.style.applymap(self.highlight_max)
        self.plot_heatmap(df, title, metrics)
        df.style.apply(self.highlight_max)
        return df

    def plot_heatmap(self, df, title, metrics):
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 4)
        plt.rcParams['font.family'] = "serif"
        plt.tick_params(axis='both', which='major', labelsize=24, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        sns.set(font='serif')
        # df.pivot(index=df.index, columns=df.columns, values=df.values)
        # df.index.name = "# Trained Robots"
        # df.columns.name = "# Testing Robots"
        #YlGnBu
        #rainbow
        #Paired
        #cool
        if metrics == 'rate_ReachGoal':
            ax = sns.heatmap(df, annot=True, fmt=".4f", linewidths=.5, cmap='rainbow_r', square=False, annot_kws={"size": 24})#, "font.family":"serif"})
        else:
            ax = sns.heatmap(df, annot=True, fmt=".4f", linewidths=.5, cmap='rainbow', square=False, annot_kws={"size": 24})#,"font.family":"serif"})
        ax.set_title = "# Testing Robots"
        # ax.figure.axes[-1].yaxis.label.set_size(20)
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=24)
        plt.show()
        name_save_fig = os.path.join(self.DATA_FOLDER, 'Generalization_{}_{}.jpg'.format(title, metrics))
        name_save_fig_pdf = os.path.join(self.DATA_FOLDER, 'Generalization_{}_{}.pdf'.format(title, metrics))

        fig.savefig(name_save_fig, bbox_inches='tight', pad_inches=0)
        fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)
        return df

    def save_data_as_table(self, df, title):
        file_name_table = os.path.join(self.DATA_FOLDER, 'Generalization_{}.csv'.format(title))
        # print(df)
        df.to_csv(file_name_table)

        # file_name_table = os.path.join(self.DATA_FOLDER, 'Generalization Table of {}.xlsx'.format(title))
        # df.to_excel(file_name_table)


if __name__ == '__main__':

    with_onlineExpert = False
    # labels = ['DCP - K2-HS0',] ## completed
    # labels = ['DCP - K3-HS0',]  ## completed

    # with_onlineExpert = True
    # labels = ['dcpOE - K2-HS0', ]  # completed
    labels = ['dcpOE - K3-HS0', ]  # completed


    DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics_generalization/'
    epoch_text = "IROS"

    label = labels[0]
    label_type = label.split(' ')[0].lower()
    label_K = int(label.split('K')[1].split('-HS')[0])

    title_text = "{}_K{}".format(label_type, label_K)

    # list_trained_agents = [6, 8, 10, 12]
    # list_testing_agents = [4, 6, 8, 10, 12, 14]

    list_trained_agents = [4, 6, 8, 10, 12]
    list_testing_agents = [4, 6, 8, 10, 12, 14]

    list_metrics = ['rate_ReachGoal', 'mean_deltaFT']#, 'mean_deltaMP']
    # list_metrics = [ 'mean_deltaFT']
    ResultAnalysis = StatisticAnalysis(DATA_FOLDER, list_metrics, labels, list_trained_agents, list_testing_agents,with_onlineExpert=with_onlineExpert)
    ResultAnalysis.summary_result(title_text, save_table=True)

