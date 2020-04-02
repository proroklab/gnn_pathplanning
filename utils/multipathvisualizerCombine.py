import numpy as np
import torch


from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import gc

class DrawpathCombine:
    def __init__(self, config, map, list_obstacle, status_MultiAgent):
        self.config = config
        self.map = map
        self.status_MultiAgent = status_MultiAgent
        self.list_obstacle = list_obstacle
        # self.makespan = makespan
        # self.flowtime = flowtime
        self.delta_name_modified = ['< ', 'v ', '> ', '^ ', "o"]
        self.delta_list =[[-1, 0],  # go up
                         [0, -1],  # go left
                         [1, 0],  # go down
                         [0, 1],  # go right
                         [0, 0]]  # stop
        self.delta = np.asarray(self.delta_list)
        #https://www.rapidtables.com/web/color/RGB_Color.html
        # self.list_agent_color = [(0.19607843137254902, 0.803921568627451, 0.19607843137254902), # lime green
        #                          (0.9607843137254902, 0.8705882352941177, 0.7019607843137254),  # wheat
        #                         (0.803921568627451, 0.5215686274509804, 0.24705882352941178), # peru
        #                         (0.5803921568627451, 0.0, 0.8274509803921568), # dark violet
        #                         (1.0, 0.4117647058823529, 0.7058823529411765), # hot pink
        #                         (1.0, 0.27058823529411763, 0.0),# orange red
        #                          (1.0, 0.2117647058823529, 0.7058823529411765),  # hot pink
        #                          (1.0, 0.47058823529411763, 0.0),
        #                          ]

        self.list_agent_color = self.get_cmap(self.config.num_agents)
        # self.list_step_color = [	(0.13333333333333333 , 0.5450980392156862 , 0.13333333333333333),
        #                             (0.9568627450980393 , 0.6431372549019608 , 0.3764705882352941),
        #                             (0.5450980392156862 , 0.27058823529411763 , 0.07450980392156863),
        #                             (0.8470588235294118 , 0.7490196078431373 , 0.8470588235294118),
        #                             (0.7803921568627451 , 0.08235294117647059 , 0.5215686274509804),
        #                             (0.5019607843137255 , 0.0 , 0.0),
        #                             (0.4803921568627451, 0.08235294117647059, 0.5215686274509804),
        #                             (0.3019607843137255, 0.0, 0.0),
        #                             ]
        # for (r, g, b) in list_agent_color:
        #     r_n = r / 255
        #     g_n = g / 255
        #     b_n = b / 255
        #     print(r_n, ',', g_n, ',', b_n)
        self.size_map = map.shape

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def draw(self, ID_dataset):
        # todo make the position of test scalable
        self.id = ID_dataset
        aspect = self.size_map[0] / self.size_map[1]
        xmin = -0.5
        ymin = -0.5
        xmax = self.size_map[0] - 0.5
        ymax = self.size_map[1] - 0.5

        # setup figure
        num_subplot = 2
        # self.fig = plt.figure(frameon=False, figsize=(8* aspect, 4))
        self.fig = plt.figure(frameon=False, figsize=(2* self.size_map[0], self.size_map[1]))

        label = ['target','predict']
        for i in range(num_subplot):

            self.ax = self.fig.add_subplot(1, num_subplot, i+1, aspect='equal')
            self.ax.set_axis_off()

            self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

            # self.ax.set_frame_on(True)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            self.patches = []
            self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='black'))

            for ID_obs in range(self.list_obstacle.shape[0]):
                obstacleIndexX = self.list_obstacle[ID_obs][0]
                obstacleIndexY = self.list_obstacle[ID_obs][1]
                self.patches.append(
                    Rectangle((obstacleIndexX - 0.5, obstacleIndexY - 0.5), 1, 1, facecolor='black', edgecolor='black'))

            self.drawMultiAgentPath(self.status_MultiAgent[label[i]])
            self.drawCollision(self.status_MultiAgent[label[i]])
            # plt.xlabel(text_result)
            for p in self.patches:
              self.ax.add_patch(p)

            self.ax.title.set_text(label[i])
            text_result = 'MS:{}, FT:{}'.format(self.status_MultiAgent[label[i]]['makespan'],
                                                self.status_MultiAgent[label[i]]['flowtime'])

            test_pos = [(xmin+xmax-2)/2, ymin-0.5]
            self.ax.text(test_pos[0],test_pos[1], text_result)



    def drawMultiAgentPath(self, status_MultiAgent):

        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)

            path = status_MultiAgent[name_agent]["path"]
            list_pathIndexX = []
            list_pathIndexY = []
            list_action = status_MultiAgent[name_agent]["action"]

            len_path = len(path)  # self.status_MultiAgent[name_agent]["len_action"]
            start = status_MultiAgent[name_agent]["start"]
            goal = status_MultiAgent[name_agent]["goal"]

            start_np = start.cpu().detach().numpy()
            goal_np = goal.cpu().detach().numpy()

            agent_color = self.list_agent_color(id_agent)
            # step_color = self.list_step_color[id_agent]
            if len_path != 0:

                color_gradient = 1 / len_path
            else:
                color_gradient = 0.1

            if color_gradient < 0.1:
                step_color_gradient = 1 / len_path
            else:
                step_color_gradient = 0.1

            # start symbol
            self.patches.append(
                Circle((start_np[0][0], start_np[0][1]), 0.3, facecolor=(1,1,1), edgecolor=agent_color, linewidth=3, alpha=1))

            for step in range(len_path):

                pathIndexX = path[step][0][0].cpu().detach().numpy()
                pathIndexY = path[step][0][1].cpu().detach().numpy()

                list_pathIndexX.append(pathIndexX)
                list_pathIndexY.append(pathIndexY)

                step_color = tuple([x * step * step_color_gradient for x in agent_color])

                # self.patches.append(Circle((pathIndexX , pathIndexY), 0.3, facecolor=agent_color, edgecolor=agent_color))

                if step < len_path - 1:
                    targetSymbol = self.delta_name_modified[list_action[step]]
                    plt.plot(pathIndexX, pathIndexY, targetSymbol, markerfacecolor=agent_color,
                             markeredgecolor=agent_color, markersize=16)
                    self.ax.annotate(step, xy = (pathIndexX, pathIndexY), color = step_color)
            plt.plot(list_pathIndexX, list_pathIndexY, linewidth=3, color=agent_color)
            # goal symbol
            plt.plot(goal_np[0][0], goal_np[0][1], '*', markerfacecolor='red', markeredgecolor='red', markersize=20)




    def drawCollision(self, status_MultiAgent):
        id_prevCollidedAgent = []
        pos_prevCollsion = {}

        id_currentCollidedAgent = []
        pos_currentCollsion = {}



        makespan = status_MultiAgent['makespan']
        for step in range(makespan):

            list_pos = []
            for id_agent in range(self.config.num_agents):
                name_agent = "agent{}".format(id_agent)
               
                path = status_MultiAgent[name_agent]["path"][step][0].tolist()
                list_pos.append(path)

            for pos in list_pos:

                count_collision = list_pos.count(pos)
                if count_collision > 1:
                    plt.plot(pos[0], pos[1], 'x', markerfacecolor='red', markeredgecolor='red', markersize=16)
                    print('Collision happens in #{} test set.'.format(self.id))
                    id_collidedAgent = list_pos.index(pos)

                    if id_collidedAgent in id_prevCollidedAgent:
                        # pos append
                        posX = [pos_prevCollsion[id_collidedAgent][0], pos[0]]
                        posY = [pos_prevCollsion[id_collidedAgent][1], pos[1]]
                        plt.plot(posX, posY, linewidth=3, color='red')

                    id_currentCollidedAgent.append(id_collidedAgent)
                    pos_currentCollsion[id_collidedAgent] = pos

            id_prevCollidedAgent = id_currentCollidedAgent
            pos_prevCollsion = pos_currentCollsion

            list_nextpos = []
            list_currentpos = []
            for id_agent in range(self.config.num_agents):
                name_agent = "agent{}".format(id_agent)

                currentstate_currrentAgent = np.asarray(status_MultiAgent[name_agent]["path"][step][0].tolist())
                action_currentAgent = status_MultiAgent[name_agent]["action"][step]
                nextstate_currrentAgent = np.add(currentstate_currrentAgent,self.delta[action_currentAgent]).tolist()
                list_currentpos.append(currentstate_currrentAgent.tolist())
                list_nextpos.append(nextstate_currrentAgent)

            for id_agent in range(self.config.num_agents):
                name_agent = "agent{}".format(id_agent)
                currentstate_currrentAgent = list_currentpos[id_agent]

                if currentstate_currrentAgent in list_nextpos:
                    id_agent_swap = list_nextpos.index(currentstate_currrentAgent)
                    name_agent_swap = "agent{}".format(id_agent_swap)
                    if name_agent_swap != name_agent:
                        if list_currentpos[id_agent_swap] == list_nextpos[id_agent]:
                            print("In #{}case (visual), {} and {} swap position happens.".format(self.id , name_agent,
                                                                                        name_agent_swap))
                            posX = [list_currentpos[id_agent_swap][0], list_currentpos[id_agent][0]]
                            posY = [list_currentpos[id_agent_swap][1], list_currentpos[id_agent][1]]
                            plt.plot(posX, posY, linewidth=3, color='red')



    def show(self):
        plt.show()

    def save(self):

        if self.config.mode == "train":
            dataset = "valid"
        else:
            dataset = "test"

        dirName = self.config.result_demo_dir + dataset

        try:
            # Create target Directory
            os.makedirs(dirName)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass

        file_name = os.path.join(dirName, '{}{:02d}x{:02d}_ID{:02d}_{}set_{:05d}_path'.format(self.config.map_type,
                                                    self.config.map_w, self.config.map_h,
                                                    self.config.id_map, dataset,
                                                    self.id))
        self.fig.savefig(file_name, bbox_inches='tight', pad_inches=0)

        self.fig.clf()
        plt.close()
        gc.collect()

