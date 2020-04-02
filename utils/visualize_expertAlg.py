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
np.set_printoptions(threshold=np.inf)
class Animation:
    def __init__(self, config):

        self.config = config
        with open(config.map) as map_file:
            self.data_map = yaml.load(map_file)

        with open(config.schedule) as states_file:
            self.schedule = yaml.load(states_file)

        self.num_agents = len(self.data_map["agents"])
        self.K = self.config .nGraphFilterTaps
        self.ID_agent = self.config.id_chosenAgent
        # data_contents = sio.loadmat(args.GSO)
        # self.GSO = np.transpose(data_contents["gso"], (2, 3, 0, 1)).squeeze(3)
        # self.commRadius = data_contents["commRadius"]

        aspect = self.data_map["map"]["dimensions"][0] / self.data_map["map"]["dimensions"][1]

        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()

        # self.list_color = self.get_cmap(self.num_agents)
        self.list_color = sns.color_palette("hls", self.num_agents)
        self.list_color_commLink = sns.color_palette("hls", 8) # self.K)
        self.list_commLinkStyle = list(lines.lineStyles.keys())
        # self.list_commLinkStyle = ['-', '-.', ':']  #

        # create boundary patch
        xmin = -0.5
        ymin = -0.5
        xmax = self.data_map["map"]["dimensions"][0] - 0.5
        ymax = self.data_map["map"]["dimensions"][1] - 0.5

        # self.ax.relim()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # plt.axis('off')
        # self.ax.axis('tight')
        # self.ax.axis('off')

        self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='black'))
        for o in self.data_map["map"]["obstacles"]:
            x, y = o[0], o[1]
            self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='black', edgecolor='black'))

        # create agents:
        self.T = 0
        # draw goals first
        for d, i in zip(self.data_map["agents"], range(0, self.num_agents)):
            self.patches.append(
                Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.6, 0.6, facecolor=self.list_color[i],
                          edgecolor=self.list_color[i], alpha=0.5))
        for d, i in zip(self.data_map["agents"], range(0, self.num_agents)):
            name = d["name"]
            self.agents[name] = Circle((d["start"][0], d["start"][1]), 0.4, facecolor=self.list_color[i],
                                       edgecolor=self.list_color[i])
            self.agents[name].original_face_color = self.list_color[i]
            self.patches.append(self.agents[name])
            self.T = max(self.T, self.schedule["schedule"][name][-1]["t"])

            # set floating ID
            self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
            self.agent_names[name].set_horizontalalignment('center')
            self.agent_names[name].set_verticalalignment('center')
            self.artists.append(self.agent_names[name])

        # self.ax.set_axis_off()
        # self.fig.axes[0].set_visible(False)
        # self.fig.axes.get_yaxis().set_visible(False)

        # self.fig.tight_layout()

        self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                                            init_func=self.init_func,
                                            frames=int(self.T + 1) * 10,
                                            interval=100,
                                            blit=True)

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def save(self, file_name, speed):

        self.anim.save(
            file_name,
            "ffmpeg",
            fps=10 * speed,
            dpi=200),
        # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

    def show(self):
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:

            self.ax.add_artist(a)
        return self.patches + self.artists


    def animate_func(self, i):

        for agent_name in self.schedule["schedule"]:
            agent = self.schedule["schedule"][agent_name]
            # print(agent)
            pos = self.getState(i / 10, agent)
            p = (pos[0], pos[1])
            self.agents[agent_name].center = p
            self.agent_names[agent_name].set_position(p)

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for id_m in range(0, len(agents_array)):
            for id_n in range(id_m + 1, len(agents_array)):
                # print(i,j)
                d1 = agents_array[id_m]
                d2 = agents_array[id_n]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                # plt.plot(pos1, pos2, 'ro-')
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print("COLLISION! (agent-agent) ({}, {})".format(id_m, id_n))

        return self.patches + self.artists

    def getState(self, t, d):
        idx = 0
        while idx < len(d) and d[idx]["t"] < t:
            idx += 1
        if idx == 0:
            return np.array([float(d[0]["x"]), float(d[0]["y"])])
        elif idx < len(d):
            posLast = np.array([float(d[idx - 1]["x"]), float(d[idx - 1]["y"])])
            posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
        else:
            return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
        dt = d[idx]["t"] - d[idx - 1]["t"]
        t = (t - d[idx - 1]["t"]) / dt
        pos = (posNext - posLast) * t + posLast
        return pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", help="input file containing map")
    parser.add_argument("--schedule", help="schedule for agents")
    parser.add_argument("--GSO", help="record of adjacency matrix for agents")
    parser.add_argument('--nGraphFilterTaps', type=int, default=3)
    parser.add_argument('--id_chosenAgent', type=int, default=0)
    parser.add_argument('--video', dest='video', default=None,
                        help="output video file (or leave empty to show on screen)")
    parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
    args = parser.parse_args()


    animation = Animation(args)

    if args.video:
        animation.save(args.video, args.speed)
    else:
        animation.show()
