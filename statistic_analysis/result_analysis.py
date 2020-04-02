from scipy.io import loadmat
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager
import plotly.graph_objects as go
import fnmatch
import pandas as pd
from cycler import cycler
from matplotlib import rc
matplotlib.font_manager._rebuild()
plt.rcParams['font.family'] = "serif"
plt.rcParams.update({'font.size': 20})


# rc('text', usetex=True)
# Say, "the default sans-serif font is COMIC SANS"
# plt.rcParams['font.serif'] = "Palatino"
# Then, "ALWAYS use sans-serif fonts"
# rc('font',**{'family':'serif','serif':['Palatino']})

# fm = matplotlib.font_manager.json_load(os.path.expanduser(" ~/.cache/matplotlib/fontlist-v300.json"))
# fm.findfont("serif", rebuild_if_missing=False)
# fm.findfont("serif", fontext="afm", rebuild_if_missing=False)

# rc('text', usetex=True)
# print(plt.style.available)

class StatisticAnalysis:
    def __init__(self, data_root, SAVEDATA_FOLDER, list_metrics, labels, list_num_agents):
        self.DATA_FOLDER = data_root
        self.SAVEDATA_FOLDER = SAVEDATA_FOLDER
        self.list_metrics = list_metrics
        self.labels = labels
        self.label_num_agents = list_num_agents
        self.load_data()

    def load_data(self):
        data = {
            'dcp': {},
            'dcpOE': {},
        }
        data_list = []

        for data_type in data.keys():
        
            for subdir, dirs, files in os.walk(os.path.join(self.DATA_FOLDER, data_type)):
                for file in files:
                    # print os.path.join(subdir, file)
                    filepath = subdir + os.sep + file
        
                    if filepath.endswith(".mat"):
                        # print(subdir, file)
                        num_agents = subdir.split('_')[-1].split('Agent')[0]
                        num_agents = int(num_agents)
                        mat_data = loadmat(filepath)
                        rate_ReachGoal = mat_data['rate_ReachGoal'][0][0]
                        mean_deltaFT = mat_data['mean_deltaFT'][0][0]
                        mean_deltaMP = mat_data['mean_deltaMP'][0][0]
                        hidden_state = mat_data['hidden_state'][0][0]
                        K = mat_data['K'][0][0]
        
                        cleaned_data = {
                            'filename': file,
                            'num_agents': num_agents,
                            'type': data_type,

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
                        data[data_type].setdefault(num_agents, []).append(cleaned_data)
        self.data_list = data_list
        self.data = data
        # print(len(data_list))
        # return data

    def collect_data(self, metrics):

        summary_label_data = []
        summary_label_error = []
        summary_label_data_text = []
        summary_list_num_agents = []
        for index, label in enumerate(self.labels):
            label_data = []
            label_error = []
            label_data_text = []
            list_num_agents = []
            for num_agents in self.label_num_agents:
                # Read type and K from label

                label_type = label.split(' ')[0].lower()
                # label_type = label.split(' ')[0]
                label_K = int(label.split('K')[1].split('-HS')[0])
                label_HS = int(label.split('-HS')[1])


                # print(label_type, label_K)
                searched_results = [item for item in self.data_list
                                    if item['num_agents'] == num_agents
                                    and item['type'].lower() == label_type
                                    and item['K'] == label_K and item['hidden_state'] == label_HS]
                # Report missing data
                if len(searched_results) == 0:
                    # print('data missing')
                    # label_data.append(0)
                    label_data_text.append("/")
                    pass
                else:
                    label_data.append(searched_results[0][metrics])
                    list_num_agents.append(num_agents)
                    label_data_text.append("{0:.4f}".format(searched_results[0][metrics]))
                    # if metrics == "mean_deltaMP":
                    #     label_error.append(searched_results[0]["std_deltaMP"])
                    #     print(searched_results[0][metrics], searched_results[0]["std_deltaMP"])
                    #     data_test = "{0:.4f} \u00B1 {0:.4f}".format(searched_results[0][metrics], searched_results[0]["std_deltaMP"])
                    #     print(data_test)
                    #     label_data_text.append(data_test)
                    # elif metrics == "mean_deltaFT":
                    #     label_error.append(searched_results[0]["std_deltaFT"])
                    #     label_data_text.append("{0:.4f} \u00B1 {0:.4f}".format(searched_results[0][metrics], searched_results[0]["std_deltaFT"]))
                    # else:
                    #     label_data_text.append("{0:.4f}".format(searched_results[0][metrics]))

            summary_label_data.append(label_data)
            summary_label_error.append(label_error)
            summary_label_data_text.append(label_data_text)
            summary_list_num_agents.append(list_num_agents)
        return summary_label_data, summary_label_error, summary_label_data_text, summary_list_num_agents


    def summary_result(self, title, text_legend, save_fig=True, use_log_scale=True, save_table=True):
        summary_df = []
        summary_df_deltaPerformance = []
        for metrics in self.list_metrics:
            label_data, label_error, label_data_text, list_num_agents = self.collect_data(metrics)

            # summary result for data at each label
            self.plot_figure(metrics, label_data, label_error, list_num_agents, title, text_legend, use_log_scale, save_fig=True)
            summary_df.append(self.save_as_table(label_data_text, self.labels, self.label_num_agents))


        fulldata_df = pd.concat(summary_df, keys=self.list_metrics)
        # fulldata_deltaPerformance = pd.concat(summary_df_deltaPerformance, keys=self.list_metrics)
        if save_table:
            self.save_data_as_table(fulldata_df, title)
        print(fulldata_df)
        # print(fulldata_deltaPerformance)
        return fulldata_df

    def plot_figure(self, metrics, label_data, label_error, list_num_agents, title_end, text_legend, use_log_scale=True, save_fig=True):

        self.fig, ax = plt.subplots()
        # self.fig.set_size_inches(9, 6)
        self.fig.set_size_inches(8, 6)
        title_text = 'Impact of K in {}'.format(metrics)

        # ax.set_title(title_text)
        ax.set_xlabel('# robots')

        if metrics == "rate_ReachGoal":
            label_text_y = "Success Rate"
            ax.set_ylabel(r'$\alpha$')
        elif metrics == "mean_deltaFT":
            label_text_y = "Flowtime Increase"
            ax.set_ylabel(r'$\delta_{FT}$')
        elif metrics == "mean_deltaMP":
            label_text_y = "Makespan Increase"
            ax.set_ylabel(label_text_y)

        self.label_data_text_combine = []

        # custom_cycler = (cycler(color=['b', 'g', 'r', 'm']) +
        #                  cycler(linestyle=['-', '--', ':', '-.']))

        ## GNN
        # custom_cycler = (cycler(color=['b', 'orange','g', 'r', 'm', ''])
        #                  )

        ## GNN with OE
        custom_cycler = (cycler(color=['k', 'brown', 'blue', 'orange', 'green', 'red',  'cyan']))


        ax.set_prop_cycle(custom_cycler)
        for index, label in enumerate(self.labels):
            ax.plot(list_num_agents[index], label_data[index], label=text_legend[index],  linewidth=3)


        # ax.yticks(np.arange())
        # ax.ticklabel_format(useOffset=False)
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        # plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)

        # plt.grid()
        # ax.yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda y, _: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
        # ax.yaxis.set_minor_formatter()

        if use_log_scale:
            ax.set_yscale('log')
        plt.grid()

        # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        # ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

        # ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation())
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        # ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


        if metrics == "rate_ReachGoal":
            ax.legend(loc='lower left', borderaxespad=0.1)
            start, end = ax.get_ylim()
            # ax.set_ylim([start * 0.6, end])
        elif metrics == "mean_deltaFT":
            # ## GNN
            ax.legend(loc='upper left', ncol=1, borderaxespad=0.1,handleheight=0.1, columnspacing=0.2, labelspacing=0.05)
            start, end = ax.get_ylim()
            # ax.set_ylim([start, end*3])
            # ax.set_ylim([0.02, 3])

            # ## GNN OE - log
            # ax.legend(loc='lower right', ncol=2, borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
            #           labelspacing=0.05)
            # start, end = ax.get_ylim()
            # ax.set_ylim([start * 0.4, end])

            ## GNN OE - origin
            # ax.legend(loc='upper left',  borderaxespad=0.1, handleheight=0.1, columnspacing=0.2,
            #           labelspacing=0.05)
            # start, end = ax.get_ylim()
            # ax.set_ylim([start * 0.4, end])

        # ax.legend()

        self.show()
        if save_fig:
            self.save_fig("{}_{}".format(metrics,title_end))


    def plot_deltaPerformance(self, metrics, list_label, label_data, list_num_agents, title_end, use_log_scale=True, save_fig=True, save_table=True):

        self.fig, ax = plt.subplots()
        self.fig.set_size_inches(6, 5.5)
        title_text = 'Deterioration of {} across K in DCP and RDCP'.format(metrics)

        ax.set_title(title_text)
        ax.set_xlabel('# robots')
        ax.set_ylabel(metrics)

        self.label_data_text_combine = []
        for index, label in enumerate(list_label):
            ax.plot(list_num_agents[index], label_data[index], label=label)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        if use_log_scale:
            ax.set_yscale('log')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.legend()
        # self.show()
        if save_fig:
            self.save_fig("{}_{}".format(title_text, title_end))
        if save_table:
            df = pd.DataFrame(label_data, columns=list_num_agents[0], index=list_label)
            self.save_data_as_table(df, title_text)

    def save_as_table(self, label_data_text, labels, list_num_agents):
        df = pd.DataFrame(np.asarray(label_data_text), index=labels, columns=list_num_agents)
        return df

    def show(self):
        plt.show()

    def save_fig(self, title):
        name_save_fig_pdf = os.path.join(self.SAVEDATA_FOLDER, "{}.pdf".format(title))
        name_save_fig = os.path.join(self.SAVEDATA_FOLDER, "{}.jpg".format(title))
        self.fig.savefig(name_save_fig, bbox_inches='tight', pad_inches = 0)
        self.fig.savefig(name_save_fig_pdf, bbox_inches='tight', pad_inches=0)

    def save_data_as_table(self, df, title_text):
        file_name_table = os.path.join(self.SAVEDATA_FOLDER, 'SummaryData_{}.csv'.format(title_text))
        # print(df)
        df.to_csv(file_name_table)


if __name__ == '__main__':


    ###############################################################
    ####################   IROS - arXiv ##########################
    ###############################################################
    select_label = 'DCP with K = 1 and DCP OE (K2)'
    labels = [
        'dcp - K1-HS0',
        'dcp - K2-HS0', 'dcp - K3-HS0',
        'dcpOE - K2-HS0', 'dcpOE - K3-HS0',
    ]
    title_text = 'GNN_GNN_OE'
    text_legend = [
        'GNN - K1',
        'GNN - K2', 'GNN - K3',
        'GNN(OE) - K2', 'GNN(OE) - K3',
    ]
    # select_label = 'DCP with K = 1 and DCP OE (K2)'
    # labels = [
    #
    #     'dcp - K2-HS0'
    # ]
    # title_text = 'GNN_GNN_OE'
    # text_legend = [
    #
    #     'GNN - K2',
    # ]

    ###############################################################
    DATA_FOLDER = '/local/scratch/ql295/Data/MultiAgentDataset/Results_best/Statistics/'

    epoch_text = "IROS"

    # use_log = True
    use_log = False

    if use_log:
        title_text = "{}_logscale".format(title_text)
    else:
        title_text = "{}".format(title_text)
    list_num_agents = [4, 6, 8, 10, 12]
    # list_metrics = ['rate_ReachGoal', 'mean_deltaFT']#, 'mean_deltaMP']
    list_metrics = ['rate_ReachGoal',]

    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)
    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    ResultAnalysis = StatisticAnalysis(DATA_FOLDER,SAVEDATA_FOLDER, list_metrics, labels, list_num_agents)
    ResultAnalysis.summary_result(title_text, text_legend, save_fig=True, use_log_scale=use_log, save_table=True)




