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
import seaborn as sns
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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class StatisticAnalysis:
    def __init__(self, data_root, SAVEDATA_FOLDER, list_metrics, labels, list_num_agents,text_legend):
        self.DATA_FOLDER = data_root
        self.SAVEDATA_FOLDER = SAVEDATA_FOLDER
        self.list_metrics = list_metrics
        self.labels = labels
        self.label_num_agents = list_num_agents
        self.text_legend = text_legend
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
                        # list_deltaFT = mat_data['list_deltaFT'][0]
                        # list_data = list_deltaFT.tolist()
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

                            'list_deltaFT': mat_data['list_deltaFT'][0],
                            'list_FT_predict': mat_data['list_FT_predict'][0],
                            'list_FT_target': mat_data['list_FT_target'][0],

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

        summary_label_setup = []
        summary_label_trained_agents = []
        summary_label_data_metric = []

        for index, label in enumerate(self.labels):

            for num_agents in self.label_num_agents:
                # Read type and K from label
                data_label = self.text_legend[index]




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
                    # label_data_text.append("/")
                    pass
                else:
                    # label_data_text.append("{0:.4f}".format(searched_results[0][metrics]))

                    if metrics == "mean_deltaFT":
                        load_data = searched_results[0]["list_deltaFT"]
                        size_data = load_data.shape[0]
                        list_data = load_data.tolist()
                        summary_label_setup.extend([data_label]*size_data)
                        summary_label_trained_agents.extend([num_agents]*size_data)
                        summary_label_data_metric.extend(list_data)
                        pass
                    else:
                        summary_label_setup.append(data_label)
                        summary_label_trained_agents.append(num_agents)
                        summary_label_data_metric.append(searched_results[0][metrics])


        return summary_label_setup, summary_label_trained_agents, summary_label_data_metric


    def summary_result(self, title, text_legend, save_fig=True, use_log_scale=True, save_table=True):
        summary_df = []
        summary_df_deltaPerformance = []
        for metrics in self.list_metrics:
            list_setup, list_trained_agents, list_data_metric = self.collect_data(metrics)

            # summary result for data at each label
            df_metrics = self.save_as_table(metrics, list_setup, list_trained_agents, list_data_metric)
            self.plot_sns_figure(metrics, df_metrics, title, text_legend, use_log_scale, save_fig=True)
            # self.plot_figure(metrics, list_setup, list_trained_agents, list_data_metric, title, text_legend, use_log_scale, save_fig=True)
            summary_df.append(df_metrics)


        fulldata_df = pd.concat(summary_df, keys=self.list_metrics)
        # fulldata_deltaPerformance = pd.concat(summary_df_deltaPerformance, keys=self.list_metrics)
        if save_table:
            self.save_data_as_table(fulldata_df, title)
        print(fulldata_df)
        # print(fulldata_deltaPerformance)
        return fulldata_df

    def plot_sns_figure(self, metrics, df_metrics, title_end, text_legend, use_log_scale=True, save_fig=True):

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

        self.label_data_text_combine = []

        # custom_cycler = (cycler(color=['b', 'g', 'r', 'm']) +
        #                  cycler(linestyle=['-', '--', ':', '-.']))


        ## GNN with OE
        # custom_cycler = (cycler(color=['k', 'brown', 'blue', 'orange', 'green', 'red',  'cyan']))
        list_color = ['k', 'brown', 'blue', 'orange', 'green', 'red',  'cyan']

        # ax.set_prop_cycle(custom_cycler)
        # sns.palplot(sns.color_palette("Paired"))
        with plt.rc_context({'lines.linewidth':3}):
            ax = sns.lineplot(x='# robots', y=label_text_y,data=df_metrics,hue='Exp setup',style='Exp setup', markers=True)


        if use_log_scale:
            ax.set_yscale('log')
        plt.grid()


        if metrics == "rate_ReachGoal":
            ax.legend(loc='lower left',ncol=1, borderaxespad=0.1,handleheight=0.1, columnspacing=0.2, labelspacing=0.05)
            start, end = ax.get_ylim()
            ax.set_ylim([0, 1])
        elif metrics == "mean_deltaFT":
            # ## GNN
            ax.legend(loc='upper left', ncol=1, borderaxespad=0.1,handleheight=0.1, columnspacing=0.2, labelspacing=0.05)
            start, end = ax.get_ylim()
            # ax.set_ylim([start, end*3])
            # ax.set_ylim([0.02, 3])


        # ax.legend()

        self.show()
        if save_fig:
            self.save_fig("{}_{}".format(metrics,title_end))


    def save_as_table(self, metrics, list_setup, list_trained_agents, list_data_metric):
        if metrics == "rate_ReachGoal":
            label_text_y = "Success Rate"
            label_text_y_latex = r'$\alpha$'
        elif metrics == "mean_deltaFT":
            label_text_y = "Flowtime Increase"
            label_text_y_latex = r'$\delta_{FT}$'

        df = pd.DataFrame({'Exp setup':list_setup, '# robots':list_trained_agents, label_text_y:list_data_metric})
        # df_tidy = pd.melt(df,id_vars='Exp setup',var_name='trained_agents', value_name=metrics)
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
    ####################   AAMAS - Extended Abstract ##############
    ###############################################################
    select_label = 'DCP with K = 1 and DCP OE (K2)'
    labels = [
        'dcp - K1-HS0',
        'dcpOE - K2-HS0', 'dcpOE - K3-HS0',
    ]
    title_text = 'GNN_GNN_OE'
    text_legend = [
        'GNN - K=1',
        'GNN - K=2', 'GNN - K=3',
    ]

    ###############################################################
    ####################   IROS - arXiv ##########################
    ###############################################################
    # select_label = 'DCP with K = 1 and DCP OE (K2)'
    # labels = [
    #     'dcp - K1-HS0',
    #     'dcp - K2-HS0', 'dcp - K3-HS0',
    #     'dcpOE - K2-HS0', 'dcpOE - K3-HS0',
    # ]
    # title_text = 'GNN_GNN_OE'
    # text_legend = [
    #     'GNN - K=1',
    #     'GNN - K=2', 'GNN - K=3',
    #     'GNN(OE) - K=2', 'GNN(OE) - K=3',
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
    list_metrics = ['rate_ReachGoal', 'mean_deltaFT']#, 'mean_deltaMP']
    # list_metrics = ['rate_ReachGoal',]
    # list_metrics = [ 'mean_deltaFT']
    SAVEDATA_FOLDER = os.path.join(DATA_FOLDER, 'Summary', title_text)
    try:
        # Create target Directory
        os.makedirs(SAVEDATA_FOLDER)
        print("Directory ", SAVEDATA_FOLDER, " Created ")
    except FileExistsError:
        pass

    ResultAnalysis = StatisticAnalysis(DATA_FOLDER,SAVEDATA_FOLDER, list_metrics, labels, list_num_agents, text_legend)
    ResultAnalysis.summary_result(title_text, text_legend, save_fig=True, use_log_scale=use_log, save_table=True)




