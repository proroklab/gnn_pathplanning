import os
import sys
import time
import yaml
import random
import signal
import argparse
import itertools
import subprocess

import numpy as np
import cv2
import scipy.io as sio
from multiprocessing import Queue, Pool, Lock, Manager, Process
from os.path import dirname, realpath, pardir

os.system("taskset -p -c 0 %d" % (os.getpid()))
# os.system("taskset -p 0xFFFFFFFF %d" % (os.getpid()))
os.system("taskset -p -c 8-15,24-31 %d" % (os.getpid()))

parser = argparse.ArgumentParser("Input width and #Agent")
parser.add_argument('--map_width', type=int, default=10)
parser.add_argument('--map_density', type=float, default=0.1)
parser.add_argument('--map_complexity', type=float, default=0.01)
parser.add_argument('--num_agents', type=int, default=4)
parser.add_argument('--num_dataset', type=int, default=30000)

args = parser.parse_args()

# set random seed
np.random.seed(1337)

def tf_index2xy(num_col, index):
    Id_row = index // num_col
    Id_col = np.remainder(index, num_col)
    return [Id_row, Id_col]
    # return Id_col, Id_row

def tf_xy2index(num_col, i, j):
    return i * num_col + j

def handler(signum, frame):
    raise Exception("Solution computed by CBS is timeout.")

class CasesGen:
    def __init__(self, path_save, path_loadmap, size_map, map_density, map_complexity, num_agents, num_dataset):
        self.size_load_map = size_map
        self.path_loadmap = path_loadmap
        self.map_density = map_density
        self.label_density = str(map_density).split('.')[-1]
        self.num_agents = num_agents
        self.num_data = num_dataset
        self.map_complexity = map_complexity
        self.path_save = path_save
        self.pair_CasesPool = []
        self.createFolder()
        self.PROCESS_NUMBER = 4
        self.timeout = 300
        self.task_queue = Queue()

    def createFolder(self):
        self.dirName_root = self.path_save + 'map{:02d}x{:02d}_density_p{}/{}_Agent/'.format(self.size_load_map[0],self.size_load_map[1],
                                                                                             self.label_density, self.num_agents)

        self.dirName_input = self.dirName_root + 'input/'
        self.dirName_output = self.dirName_root + 'output/'

        try:
            # Create target Directory
            os.makedirs(self.dirName_root)
            os.makedirs(self.dirName_input)
            os.makedirs(self.dirName_output)
            print("Directory ", self.dirName_root, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass
        try:
            # Create target Directory
            os.makedirs(self.dirName_input)
            os.makedirs(self.dirName_output)
            print("Directory ", self.dirName_root, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass

    def search_Cases(self, dir):
        # make a list of file name of input yaml
        list_path = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    path = os.path.join(root, fname)
                    list_path.append(path)

        return list_path

    def is_target_file(self, filename):
        DATA_EXTENSIONS = ['.yaml']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)

    def mapGen(self, width=10, height=10, complexity=0.01, density=0.1):
        # Only odd shapes
        # world_size = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # world_size = ((height // 2) * 2 , (width // 2) * 2 )
        world_size = (height, width)
        # Adjust complexity and density relative to maze size

        # number of components
        complexity = int(complexity * (5 * (world_size[0] + world_size[1])))
        # size of components
        density = int(density * ((world_size[0] // 2) * (world_size[1] // 2)))

        # density = int(density * world_size[0] * world_size[1])
        # Build actual maze
        maze = np.zeros(world_size, dtype=np.int64)

        # Make aisles
        for i in range(density):
            # x, y = np.random.randint(0, world_size[1]), np.random.randint(0, world_size[0])

            # pick a random position
            x, y = np.random.randint(0, world_size[1] // 2) * 2, np.random.randint(0, world_size[0] // 2) * 2

            maze[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < world_size[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < world_size[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.randint(0, len(neighbours) - 1)]
                    if maze[y_, x_] == 0:
                        maze[y_, x_] = 1
                        maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
            # print(np.count_nonzero(maze))
        return maze

    def img_fill(self, im_in, n):  # n = binary image threshold
        th, im_th = cv2.threshold(im_in, n, 1, cv2.THRESH_BINARY)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # print(im_floodfill_inv)
        # Combine the two images to get the foreground.
        fill_image = im_th | im_floodfill_inv

        return fill_image

    def setup_cases(self, id_random_case):
        # Generate only one random unique cases in unique map

        map_env_raw = self.mapGen(width=self.size_load_map[0], height=self.size_load_map[1], complexity=self.map_complexity, density=self.map_density)

        self.size_load_map = np.shape(map_env_raw)

        # use flood-fill to ensure the connectivity of node in maze
        map_env = self.img_fill(map_env_raw.astype(np.uint8), 0.5)


        array_freespace = np.argwhere(map_env == 0)
        num_freespace = array_freespace.shape[0]
        array_obstacle = np.transpose(np.nonzero(map_env))
        num_obstacle = array_obstacle.shape[0]


        print("###### Check Map Size: [{},{}]- density: {} - Actual [{},{}] - #Obstacle: {}".format(self.size_load_map[0], self.size_load_map[1],
                                                                                               self.map_density, self.size_load_map[0],self.size_load_map[1],
                                                                                               num_obstacle))

        list_freespace = []
        list_obstacle = []

        # transfer into list (tuple)
        for id_FS in range(num_freespace):
            list_freespace.append((array_freespace[id_FS, 0], array_freespace[id_FS, 1]))

        for id_Obs in range(num_obstacle):
            list_obstacle.append((array_obstacle[id_Obs, 0], array_obstacle[id_Obs, 1]))

        pair_CaseSet = []
        for id_agents in range(self.num_agents):
            ID_cases_agent = random.sample(list_freespace, 2)
            pair_CaseSet.append(ID_cases_agent)

        pair_agent = list(itertools.combinations(range(self.num_agents), 2))

        check_condition = []
        for id_pairagent in range(len(pair_agent)):
            firstAgent = pair_agent[id_pairagent][0]
            secondAgent = pair_agent[id_pairagent][1]
            # print("pair", pairset)
            if pair_CaseSet[firstAgent][0] == pair_CaseSet[secondAgent][0] or pair_CaseSet[firstAgent][1] == \
                    pair_CaseSet[secondAgent][1]:
                print("Remove pair \t", pair_CaseSet)
                check_condition.append(0)
            else:
                check_condition.append(1)
                # pairStore.append(pairset)

        # todo: generate n-agent pair start-end position - start from single agent CBS
        # todo: non-swap + swap

        if sum(check_condition) == len(pair_agent):
            # print("Add {}-case: {}".format(id_random_case,pair_CaseSet))
            # self.pair_CasesPool.append(pair_CaseSet)
            # return True, pair_CaseSet, map_env

            return True, pair_CaseSet, list_obstacle
        else:
            print("Remove cases ID-{}:\t {}".format(id_random_case, pair_CaseSet))
            return False, [],[]



    def setup_CasePool(self):
        pairStore = []
        mapStore = []

        num_data_exceed = int(self.num_data * 2)

        for id_random_case in range(num_data_exceed):
            Check_add_item, pair_CaseSet, map_env = self.setup_cases(id_random_case)
            if Check_add_item:
                pairStore.append(pair_CaseSet)
                mapStore.append(map_env)

        # [k for k in d if not d[k]]
        for initialCong in pairStore:
            count_repeat = pairStore.count(initialCong)
            if count_repeat > 1:
                id_repeat = pairStore.index(initialCong)
                pairStore.remove(initialCong)
                map_toRemoved = mapStore[id_repeat[0]]
                mapStore.remove(map_toRemoved)
                print('Repeat cases ID:{} \n{} \nObstacle list: \n{} '.format(id_repeat,pairStore[id_repeat],map_toRemoved))

        CasePool = list(zip(pairStore, mapStore))
        random.shuffle(CasePool)
        random.shuffle(CasePool)
        pairPool, mapPool = zip(*CasePool)

        self.save_CasePool(pairPool,mapPool)
        # self.save_Pair(pairStore)

    def save_CasePool(self,pairPool,mapPool):
        for id_case in range(len(pairPool)):
            inputfile_name = self.dirName_input + \
                             'input_map{:02d}x{:02d}_ID{:05d}.yaml'.format(self.size_load_map[0], self.size_load_map[1],id_case)
            self.dump_yaml(self.num_agents, self.size_load_map[0], self.size_load_map[1],
                           pairPool[id_case], mapPool[id_case], inputfile_name)


    def dump_yaml(self, num_agent, map_width, map_height, agents, obstacle_list, filename):
        f = open(filename, 'w')
        f.write("map:\n")
        f.write("    dimensions: {}\n".format([map_width, map_height]))
        f.write("    obstacles:\n")
        for id_Obs in range(len(obstacle_list)):
            f.write("    - [{}, {}]\n".format(obstacle_list[id_Obs][0],obstacle_list[id_Obs][1]))
        f.write("agents:\n")
        for n in range(num_agent):
            # f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(n, agents[n][0], agents[n][1]))
            # f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(n, agents[n]['start'], agents[n]['goal']))
            f.write("  - name: agent{}\n    start: [{}, {}]\n    goal: [{}, {}]\n".format(n, agents[n][0][0], agents[n][0][1],
                                                                              agents[n][1][0], agents[n][1][1]))
        f.close()


    def computeSolution(self):

        self.list_Cases_input = self.search_Cases(self.dirName_input)
        self.len_pair = len(self.list_Cases_input)

        for id_case in range(self.len_pair):
            self.task_queue.put(id_case)

        time.sleep(0.3)
        processes = []
        for i in range(self.PROCESS_NUMBER):
            # Run Multiprocesses
            p = Process(target=self.compute_thread, args=(str(i)))

            processes.append(p)
        [x.start() for x in processes]


    def compute_thread(self, thread_id):
        while True:
            try:
                # print(thread_id)
                id_case = self.task_queue.get(block=False)
                print('thread {} get task:{}'.format(thread_id,id_case))
                self.runExpertSolver(id_case)
                # print('thread {} finish task:{}'.format(thread_id, id_case))
            except:
                # print('thread {} no task, exit'.format(thread_id))
                return

    def runExpertSolver(self, id_case):

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)
        try:
            # load
            name_inputfile = self.list_Cases_input[id_case]
            id_input_case = name_inputfile.split('_ID')[-1]

            name_outputfile = self.dirName_output + 'output_map{:02d}x{:02d}_ID{}.yaml'.format(self.size_load_map[0], self.size_load_map[1], id_input_case)
            command_dir = dirname(realpath(__file__))
            # print(command_dir)
            # command_dir = '/local/scratch/ql295/Data/Project/GraphNeural_Planner/onlineExpert'
            command_file = os.path.join(command_dir, "ecbs")
            # run ECBS
            subprocess.call(
                [command_file,
                 "-i", name_inputfile,
                 "-o", name_outputfile,
                 "-w", str(1.1)],
                cwd=command_dir)

            log_str = 'map{:02d}x{:02d}_{}Agents_#{}'.format(self.size_load_map[0], self.size_load_map[1], self.num_agents, id_input_case)
            print('############## Find solution for {} generated  ###############'.format(log_str))
            with open(name_outputfile) as output_file:
                return yaml.safe_load(output_file)
        except Exception as e:
            print(e)


if __name__ == '__main__':


    # path_loadmap = '/homes/ql295/PycharmProjects/GraphNeural_Planner/ExpertPlanner/MapDataset'
    # path_loadmap = '/homes/ql295/Documents/Graph_mapf_dataset/setup/map/'

    # path_savedata = '/homes/ql295/Documents/Graph_mapf_dataset/solution/'
    # path_savedata = '/local/scratch/ql295/Data/MultiAgentDataset/SolutionTri_ECBS/'


    path_loadmap = ''
    path_savedata = '/local/scratch/ql295/Data/MultiAgentDataset/Solution_DMap/'


    # num_dataset = 10 #16**2
    # size_map = (5, 5)

    size_map = (args.map_width, args.map_width)
    map_density = args.map_density
    map_complexity = args.map_complexity
    num_agents = args.num_agents
    num_dataset = args.num_dataset

    dataset = CasesGen(path_savedata, path_loadmap, size_map, map_density, map_complexity, num_agents, num_dataset)
    timeout = 300

    dataset.setup_CasePool()

    time.sleep(10)
    dataset.computeSolution()





