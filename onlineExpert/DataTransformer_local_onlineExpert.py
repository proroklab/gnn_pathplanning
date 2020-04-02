
import csv
import os
import sys
import shutil
import time
import numpy as np
import scipy.io as sio
import yaml

from easydict import EasyDict
from os.path import dirname, realpath, pardir
from hashids import Hashids
import hashlib
sys.path.append(os.path.join(dirname(realpath(__file__)), pardir))

import utils.graphUtils.graphTools as graph
# from utils.graphUtils.graphTools import isConnected

from dataloader.statetransformer import AgentState
from scipy.spatial.distance import squareform, pdist
from multiprocessing import Queue, Process


class DataTransformer:
    def __init__(self, config):
        self.config = config
        self.PROCESS_NUMBER = 4
        self.num_agents = self.config.num_agents
        self.size_map = [self.config.map_w, self.config.map_h]
        self.AgentState = AgentState(self.num_agents)
        self.communicationRadius = 5 # communicationRadius
        self.zeroTolerance = 1e-9
        self.delta = [[-1, 0],  # go up
                 [0, -1],  # go left
                 [1, 0],  # go down
                 [0, 1],  # go right
                 [0, 0]]  # stop
        self.num_actions = 5
        self.root_path_save = self.config.failCases_dir
        self.list_seqtrain_file = []
        self.list_train_file = []
        self.pathtransformer = self.pathtransformer_RelativeCoordinate

    def set_up(self, epoch):
        self.dir_input = os.path.join(self.config.failCases_dir, "input/")
        self.dir_sol = os.path.join(self.config.failCases_dir, "output_ECBS/")
        self.list_failureCases_solution = self.search_failureCases(self.dir_sol)
        self.list_failureCases_input = self.search_failureCases(self.dir_input)
        self.nameprefix_input = self.list_failureCases_input[0].split('input/')[-1].split('ID')[0]
        self.list_failureCases_solution = sorted(self.list_failureCases_solution)
        self.len_failureCases_solution = len(self.list_failureCases_solution)
        self.current_epoch = epoch
        self.task_queue = Queue()

        self.path_save_solDATA = os.path.join(self.root_path_save, "Cache_data", "Epoch_{}".format(epoch))
        try:
            # Create target Directory
            os.makedirs(self.path_save_solDATA)
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass


    def solutionTransformer(self):



        for id_sol in range(self.len_failureCases_solution):
        # for id_sol in range(21000):
            self.task_queue.put(id_sol)

        time.sleep(0.3)
        processes = []
        for i in range(self.PROCESS_NUMBER):
            # Run Multiprocesses
            p = Process(target=self.compute_thread, args=(str(i)))

            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]


    def compute_thread(self,thread_id):
        while True:
            try:
                id_sol = self.task_queue.get(block=False)
                print('thread {} get task:{}'.format(thread_id, id_sol))
                self.pipeline(id_sol)

            except:
                # print('thread {} no task, exit'.format(thread_id))
                return

    def pipeline(self,id_sol):
        agents_schedule, agents_goal, makespan, map_data, id_case = self.load_ExpertSolution(id_sol)
        log_str = 'Transform_failureCases_ID_#{} in Epoch{}'.format(id_case[1],id_case[0])
        print('############## {} ###############'.format(log_str))
        self.pathtransformer(map_data, agents_schedule, agents_goal, makespan + 1, id_case)
        

    def load_ExpertSolution(self, ID_case):

        name_solution_file = self.list_failureCases_solution[ID_case]
        id_sol_case = name_solution_file.split('_ID')[-1].split('.yaml')[0]
        name_inputfile = self.dir_input + self.nameprefix_input + 'ID{}.yaml'.format(id_sol_case)

        with open(name_inputfile, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(name_solution_file, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        agentsConfig = data_config['agents']
        num_agent = len(agentsConfig)
        list_posObstacle = data_config['map']['obstacles']

        if list_posObstacle == None:
            map_data = np.zeros(self.size_map, dtype=np.int64)
        else:
            map_data = self.setup_map(list_posObstacle)
        schedule = data_output['schedule']
        makespan = data_output['statistics']['makespan']


        goal_allagents = np.zeros([num_agent, 2])
        schedule_agentsState = np.zeros([makespan + 1, num_agent, 2])
        schedule_agentsActions = np.zeros([makespan + 1, num_agent, self.num_actions])
        schedule_agents = [schedule_agentsState, schedule_agentsActions]
        hash_ids = np.zeros(self.num_agents)
        for id_agent in range(num_agent):
            goalX = agentsConfig[id_agent]['goal'][0]
            goalY = agentsConfig[id_agent]['goal'][1]
            goal_allagents[id_agent][:] = [goalX, goalY]

            schedule_agents = self.obtainSchedule(id_agent, schedule, schedule_agents, goal_allagents, makespan + 1)

            str_id = '{}_{}_{}'.format(self.current_epoch,id_sol_case,id_agent)
            int_id = int(hashlib.sha256(str_id.encode('utf-8')).hexdigest(), 16) % (10 ** 5)
            # hash_ids[id_agent]=np.divide(int_id,10**5)
            hash_ids[id_agent] = int_id

        # print(id_sol_map, id_sol_case, hash_ids)
        return schedule_agents, goal_allagents, makespan, map_data, (self.current_epoch, id_sol_case, hash_ids)

    def obtainSchedule(self, id_agent, agentplan, schedule_agents, goal_allagents, teamMakeSpan):

        name_agent = "agent{}".format(id_agent)
        [schedule_agentsState, schedule_agentsActions] = schedule_agents
        
        planCurrentAgent = agentplan[name_agent]
        pathLengthCurrentAgent = len(planCurrentAgent)

        actionKeyListAgent = []

        for step in range(teamMakeSpan):
            if step < pathLengthCurrentAgent:
                currentX = planCurrentAgent[step]['x']
                currentY = planCurrentAgent[step]['y']
            else:
                currentX = goal_allagents[id_agent][0]
                currentY = goal_allagents[id_agent][1]
                
            schedule_agentsState[step][id_agent][:] = [currentX, currentY]
            # up left down right stop
            actionVectorTarget = [0, 0, 0, 0, 0]

            # map action with respect to the change of position of agent
            if step < (pathLengthCurrentAgent - 1):
                nextX = planCurrentAgent[step + 1]['x']
                nextY = planCurrentAgent[step + 1]['y']
                # actionCurrent = [nextX - currentX, nextY - currentY]

            elif step >= (pathLengthCurrentAgent - 1):
                nextX = goal_allagents[id_agent][0]
                nextY = goal_allagents[id_agent][1]

            actionCurrent = [nextX - currentX, nextY - currentY]


            actionKeyIndex = self.delta.index(actionCurrent)
            actionKeyListAgent.append(actionKeyIndex)

            actionVectorTarget[actionKeyIndex] = 1
            schedule_agentsActions[step][id_agent][:] = actionVectorTarget


        return [schedule_agentsState,schedule_agentsActions]

    def setup_map(self, list_posObstacle):
        num_obstacle = len(list_posObstacle)
        map_data = np.zeros(self.size_map)
        for ID_obs in range(num_obstacle):
            obstacleIndexX = list_posObstacle[ID_obs][0]
            obstacleIndexY = list_posObstacle[ID_obs][1]
            map_data[obstacleIndexX][obstacleIndexY] = 1

        return map_data



    def pathtransformer_RelativeCoordinate(self, map_data, agents_schedule, agents_goal, makespan, ID_case):
        # input: start and goal position,
        # output: a set of file,
        #         each file consist of state (map. goal, state) and target (action for current state)
        mode = 'train'
        [schedule_agentsState, schedule_agentsActions] = agents_schedule
        save_PairredData = {}

        # compute AdjacencyMatrix
        GSO, communicationRadius = self.computeAdjacencyMatrix(schedule_agentsState, self.communicationRadius)

        # transform into relative Coordinate, loop "makespan" times
        self.AgentState.setmap(map_data)
        input_seq_tensor = self.AgentState.toSeqInputTensor(agents_goal, schedule_agentsState, makespan)

        list_input = input_seq_tensor.cpu().detach().numpy()
        save_PairredData.update({'map': map_data, 'goal': agents_goal, 'inputState': schedule_agentsState,
                                 'inputTensor': list_input, 'target': schedule_agentsActions,
                                 'GSO': GSO,'makespan':makespan, 'HashIDs':ID_case[2]})

        self.save(mode, save_PairredData, ID_case, makespan)
        print("Save as Relative Coordination - {}set_#{} from Epoch {}.".format(mode, ID_case[1], ID_case[0]))

    def save(self, mode, save_PairredData, ID_case, makespan):

        file_name = os.path.join(self.path_save_solDATA, '{}_ID{}_MP{}.mat'.format(mode, ID_case[1], makespan))
        sio.savemat(file_name, save_PairredData)


    def search_failureCases(self, dir):
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

    def computeAdjacencyMatrix(self, pos, CommunicationRadius, connected=True):

        # First, transpose the axis of pos so that the rest of the code follows
        # through as legible as possible (i.e. convert the last two dimensions
        # from 2 x nNodes to nNodes x 2)
        # pos: TimeSteps x nAgents x 2 (X, Y)

        # Get the appropriate dimensions
        nSamples = pos.shape[0]
        len_TimeSteps = pos.shape[0]  # length of timesteps
        nNodes = pos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices
        W = np.zeros([len_TimeSteps, nNodes, nNodes])
        threshold = CommunicationRadius  # We compute a different
        # threshold for each sample, because otherwise one bad trajectory
        # will ruin all the adjacency matrices

        for t in range(len_TimeSteps):
            # Compute the distances
            distances = squareform(pdist(pos[t]))  # nNodes x nNodes
            # Threshold them
            W[t] = (distances < threshold).astype(pos.dtype)
            # And get rid of the self-loops
            W[t] = W[t] - np.diag(np.diag(W[t]))
            # Now, check if it is connected, if not, let's make the
            # threshold bigger
            while (not graph.isConnected(W[t])) and (connected):
                # while (not graph.isConnected(W[t])) and (connected):
                # Increase threshold
                threshold = threshold * 1.1  # Increase 10%
                # Compute adjacency matrix
                W[t] = (distances < threshold).astype(pos.dtype)
                W[t] = W[t] - np.diag(np.diag(W[t]))

        # And since the threshold has probably changed, and we want the same
        # threshold for all nodes, we repeat:
        W = np.zeros([len_TimeSteps, nNodes, nNodes])
        for t in range(len_TimeSteps):
            distances = squareform(pdist(pos[t]))
            W[t] = (distances < threshold).astype(pos.dtype)
            W[t] = W[t] - np.diag(np.diag(W[t]))
            # And, when we compute the adjacency matrix, we normalize it by
            # the degree
            deg = np.sum(W[t], axis=1)  # nNodes (degree vector)
            # Build the degree matrix powered to the -1/2
            Deg = np.diag(np.sqrt(1. / deg))
            # And finally get the correct adjacency
            W[t] = Deg @ W[t] @ Deg

        return W, threshold

    def pathtransformer_GlobalCoordinate(self, map_data, agents_schedule, agents_goal, makespan, ID_case):
        # input: start and goal position,
        # output: a set of file,
        #         each file consist of state (map. goal, state) and target (action for current state)

        mode = 'train'
        [schedule_agentsState, schedule_agentsActions] = agents_schedule
        save_PairredData = {}
        save_PairredData.update({'map': map_data, 'goal': agents_goal,
                                 'inputState': schedule_agentsState,
                                 'target': schedule_agentsActions,
                                 'makespan': makespan})

        self.save(mode, save_PairredData, ID_case)
        # print("Save as Global Coordination - {}set_#{}.".format(mode, ID_case))

if __name__ == '__main__':

    config = {'num_agents': 12,
              'map_w': 20,
              'map_h': 20,
              'failCases_dir': '/local/scratch/ql295/Data/MultiAgentDataset/test',
              'exp_net': 'dcp'
              }
    config_setup = EasyDict(config)
    DataTransformer = DataTransformer(config_setup)
    DataTransformer.set_up('1')
    DataTransformer.solutionTransformer()
