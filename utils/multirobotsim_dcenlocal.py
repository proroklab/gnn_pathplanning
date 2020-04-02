import numpy as np
import torch

import os
import sys
from utils.multipathvisualizerCombine import DrawpathCombine
from torch import nn
import random
import time
random.seed(1337)
import utils.graphUtils.graphTools as graph
from scipy.spatial.distance import squareform, pdist
from dataloader.statetransformer import AgentState
import scipy.io as sio
# from onlineExpert.ECBS_onlineExpert import ComputeCBSSolution

class multiRobotSim:
    def __init__(self, config):
        self.config = config

        self.AgentState = AgentState(self.config.num_agents)
        self.delta_list =[[-1, 0],  # go up
                         [0, -1],  # go left
                         [1, 0],  # go down
                         [0, 1],  # go right
                         [0, 0]]  # stop
        self.delta = torch.FloatTensor(self.delta_list).to(self.config.device)

        self.List_MultiAgent_ActionVec_target = None
        self.store_MultiAgent = None
        self.channel_map = None

        self.size_map = None
        self.maxstep = None

        self.posObstacle = None
        self.numObstacle = None
        self.posStart = None
        self.posGoal = None

        self.currentState_predict = None

        self.makespanTarget = None
        self.flowtimeTarget = None
        self.makespanPredict = None
        self.flowtimePredict = None

        self.count_reachgoal = None
        self.count_reachgoalTarget = None
        self.fun_Softmax = None
        self.zeroTolerance = 1e-9
        print("run on multirobotsim with collision shielding")

    def setup(self, loadInput, loadTarget, makespanTarget, tensor_map, ID_dataset):

        # self.fun_Softmax = nn.Softmax(dim=-1)
        self.fun_Softmax = nn.LogSoftmax(dim=-1)
        self.ID_dataset = ID_dataset

        self.store_GSO = []
        self.store_communication_radius = []
        self.status_MultiAgent = {}
        # setupState = loadInput.permute(3, 4, 2, 1, 0)
        target = loadTarget.permute(1, 2, 3, 0)
        self.List_MultiAgent_ActionVec_target = target[:, :, :,0]
        # self.List_MultiAgent_ActionVec_target = target[:, :, 0]

        self.channel_map = tensor_map[0] # setupState[:, :, 0, 0, 0]
        self.AgentState.setmap(self.channel_map)
        self.posObstacle = self.findpos(self.channel_map).to(self.config.device)
        self.numObstacle = self.posObstacle.shape[0]
        self.size_map = self.channel_map.shape

        # self.communicationRadius = 5 #self.size_map[0] * 0.5
        # self.maxstep = self.size_map[0] * self.size_map[1]
        if self.config.num_agents >=20:
            self.rate_maxstep = 3
        else:
            self.rate_maxstep = self.config.rate_maxstep

        self.maxstep = int(makespanTarget.type(torch.int32) * self.rate_maxstep)

        self.check_predictCollsion = False
        self.check_moveCollision = True
        self.check_predictEdgeCollsion = [False] * self.config.num_agents
        self.count_reachgoal = [False] * self.config.num_agents
        self.count_reachgoalTarget = [False] * self.config.num_agents
        self.allReachGoal_Target = False
        self.makespanTarget = 0
        self.flowtimeTarget = 0
        self.makespanPredict = self.maxstep
        self.flowtimePredict = self.maxstep * self.config.num_agents #0

        self.stopKeyValue = torch.tensor(4).to(self.config.device)
        self.reset_disabled_action = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).float().to(self.config.device)

        self.store_goalAgents = loadInput[0, 0, :,:]
        self.store_stateAgents = loadInput[0, 1, :, :]
        for id_agent in range(self.config.num_agents):

            status_CurrentAgent = {}

            posGoal = loadInput[:, 0,id_agent,:] #self.findpos(goal_CurrentAgent)
            posStart = loadInput[:, 1,id_agent,:] #self.findpos(start_CurrentAgent)


            path_predict = {0:posStart}
            path_target = {0:posStart}
            len_action_predict  = 0
            list_actionKey_predict = []
            actionVec_target_CurrentAgents = self.List_MultiAgent_ActionVec_target[id_agent, :, :]
            actionKeyList_target_CurrentAgents = torch.max(actionVec_target_CurrentAgents, 1)[1]

            disabled_action_predict_currentAgent = self.reset_disabled_action
            startStep_action_currentAgent = None
            endStep_action_currentAgent = None


            len_action_target = actionKeyList_target_CurrentAgents.shape[0]

            status_CurrentAgents = {"goal": posGoal,
                                    "start": posStart,#torch.FloatTensor([[0,0]]).to(self.config.device),
                                    "currentState": posStart,
                                    "path_target": path_target,
                                    "action_target": actionKeyList_target_CurrentAgents,
                                    "len_action_target": len_action_target,
                                    "startStep_action_target": startStep_action_currentAgent,
                                    "endStep_action_target": endStep_action_currentAgent,
                                    "path_predict": path_predict,
                                    "nextState_predict": posStart,
                                    "action_predict": list_actionKey_predict,
                                    "disabled_action_predict": disabled_action_predict_currentAgent,
                                    "len_action_predict": len_action_predict,
                                    "startStep_action_predict": startStep_action_currentAgent,
                                    "endStep_action_predict": endStep_action_currentAgent
                                    }
            # print("Agent{} - goal:{} - start:{} - currentState:{}".format(id_agent, posGoal,posStart,posStart))
            name_agent = "agent{}".format(id_agent)
            self.status_MultiAgent.update({name_agent: status_CurrentAgents})

        self.getPathTarget()

        pass

    def findpos(self, channel):
        pos_object = channel.nonzero()
        num_object = pos_object.shape[0]
        pos = torch.zeros(num_object, 2)
        # pos_list = []

        for i in range(num_object):
            pos[i][0] = pos_object[i][0]
            pos[i][1] = pos_object[i][1]
        #     pos_list.append([pos_object[i][0], pos_object[i][1]])
        # pos = torch.FloatTensor(pos_list)
        return pos


    def getPathTarget(self):
        #todo check the length for ground truth, out of index

        list_len_action_target = []
        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)

            len_actionTarget_currentAgent = self.status_MultiAgent[name_agent]["len_action_target"]
            list_len_action_target.append(len_actionTarget_currentAgent)

        maxStep = max(list_len_action_target)

        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)

            pathTarget_currentAgent = self.status_MultiAgent[name_agent]["path_target"]
            currentState_target = self.status_MultiAgent[name_agent]['start']
            goal_currentAgent = self.status_MultiAgent[name_agent]['goal']

            nextState_target = currentState_target
            goalIndexX = int(goal_currentAgent[0][0])
            goalIndexY = int(goal_currentAgent[0][1])


            for step in range(maxStep):

                actionKey_target = self.status_MultiAgent[name_agent]['action_target'][step]

                check_move = (actionKey_target != self.stopKeyValue)
                check_startStep_action = self.status_MultiAgent[name_agent]["startStep_action_target"]

                if check_move == 1 and check_startStep_action is None:
                    self.status_MultiAgent[name_agent]["startStep_action_target"] = step

                else:
                    currentState_target = nextState_target

                action_target = self.delta[actionKey_target]
                nextState_target = torch.add(currentState_target, action_target)

                pathTarget_currentAgent.update({step+1: nextState_target})

                self.status_MultiAgent[name_agent]["path_target"] = pathTarget_currentAgent

                if nextState_target[0][0] == goalIndexX and nextState_target[0][1] == goalIndexY and not self.count_reachgoalTarget[id_agent]:
                    self.count_reachgoalTarget[id_agent] = True
                    self.status_MultiAgent[name_agent]["endStep_action_target"] = step + 1

                self.allReachGoal_Target = all(self.count_reachgoalTarget)

            if self.allReachGoal_Target:
                List_endStep_target = []
                List_startStep_target = []
                self.flowtimeTarget = 0
                for id_agent in range(self.config.num_agents):
                    name_agent = "agent{}".format(id_agent)
                    List_endStep_target.append(self.status_MultiAgent[name_agent]["endStep_action_target"])
                    List_startStep_target.append(self.status_MultiAgent[name_agent]["startStep_action_target"])

                    self.flowtimeTarget += self.status_MultiAgent[name_agent]["endStep_action_target"] - \
                                            self.status_MultiAgent[name_agent]["startStep_action_target"]

                    len_action_predict = self.status_MultiAgent[name_agent]["endStep_action_target"] - \
                                         self.status_MultiAgent[name_agent]["startStep_action_target"]
                    self.status_MultiAgent[name_agent]["len_action_target"] = len_action_predict

                self.makespanTarget = max(List_endStep_target) - min(List_startStep_target)

                # print("Makespane(target):{} \n Flowtime(target): {} \n ").format(self.makespanTarget, self.flowtimeTarget)
                break



    def getOptimalityMetrics(self):
        return [self.makespanPredict, self.makespanTarget], [self.flowtimePredict, self.flowtimeTarget]

    def getMaxstep(self):

        return self.maxstep

    def getMapsize(self):
        return self.size_map

    def initCommunicationRadius(self):
        self.communicationRadius = self.config.commR
        # self.communicationRadius = 5
        # self.communicationRadius = 6
        # self.communicationRadius = 7
        # self.communicationRadius = 8
        # self.communicationRadius = 9
        # self.communicationRadius = 10


    def reachObstacle(self, state):
        reach_obstacle = False

        # name_agent = "agent{}".format(id_agent)
        currentState_predict = state #self.status_MultiAgent[name_agent]["currentState"]
        currentStateIndexX = currentState_predict[0][0]
        currentStateIndexY = currentState_predict[0][1]

        # print(self.channel_map.shape)
        # print(self.channel_map)
        # time.sleep(10)

        if self.channel_map[int(currentStateIndexX)][int(currentStateIndexY)] == 1:
            # print('Reach obstacle.')
            reach_obstacle = True
        else:
            reach_obstacle = False


        # if reach_obstacle:
        #     break
        return reach_obstacle

    def reachEdge(self, state):
        reach_edge = False

        # name_agent = "agent{}".format(id_agent)
        currentState_predict = state #self.status_MultiAgent[name_agent]["currentState"]
        currentStateIndexX = currentState_predict[0][0]
        currentStateIndexY = currentState_predict[0][1]

        if currentStateIndexX >= self.size_map[0] or currentStateIndexX < 0 or currentStateIndexY >= self.size_map[1] or currentStateIndexY < 0:
            # print('Reach edge.')
            reach_edge = True
            # break
        else:
            reach_edge = False
        return reach_edge

    def computeAdjacencyMatrix_fixedCommRadius(self, step, agentPos, CommunicationRadius, graphConnected=False):
        len_TimeSteps = agentPos.shape[0]  # length of timesteps
        nNodes = agentPos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices
        W = np.zeros([len_TimeSteps, nNodes, nNodes])

        # Initial matrix
        distances = squareform(pdist(agentPos[0]))  # nNodes x nNodes

        # I will increase the communication radius by 10% each time,
        # but I have to do it consistently within the while loop,
        # so in order to not affect the first value set of communication radius, I will account for that initial 10% outside


        distances = squareform(pdist(agentPos[0]))  # nNodes x nNodes
        W[0] = (distances < self.communicationRadius).astype(agentPos.dtype)
        W[0] = W[0] - np.diag(np.diag(W[0]))
        graphConnected = graph.isConnected(W[0])
        deg = np.sum(W[0], axis=1)  # nNodes (degree vector)
        zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)[0]
        deg[zeroDeg] = 1.
        invSqrtDeg = np.sqrt(1. / deg)
        invSqrtDeg[zeroDeg] = 0.
        Deg = np.diag(invSqrtDeg)
        W[0] = Deg @ W[0] @ Deg

        return W, self.communicationRadius, graphConnected


    def computeAdjacencyMatrix(self, step, agentPos, CommunicationRadius, graphConnected=False):
        len_TimeSteps = agentPos.shape[0]  # length of timesteps
        nNodes = agentPos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices
        W = np.zeros([len_TimeSteps, nNodes, nNodes])

        # Initial matrix
        distances = squareform(pdist(agentPos[0]))  # nNodes x nNodes


        # I will increase the communication radius by 10% each time,
        # but I have to do it consistently within the while loop,
        # so in order to not affect the first value set of communication radius, I will account for that initial 10% outside

        if step == 0:
            self.communicationRadius = self.communicationRadius / 1.1
            while graphConnected is False:
                self.communicationRadius = self.communicationRadius * 1.1
                W[0] = (distances < self.communicationRadius).astype(agentPos.dtype)
                W[0] = W[0] - np.diag(np.diag(W[0]))
                graphConnected = graph.isConnected(W[0])
            # And once we have found a connected initial position, we normalize it
            deg = np.sum(W[0], axis=1)  # nNodes (degree vector)
            zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)[0]
            deg[zeroDeg] = 1.
            invSqrtDeg = np.sqrt(1. / deg)
            invSqrtDeg[zeroDeg] = 0.
            Deg = np.diag(invSqrtDeg)
            W[0] = Deg @ W[0] @ Deg

        # And once we have found a communication radius that makes the initial graph connected,
        # just follow through with the rest of the times, with that communication radius
        else:
            distances = squareform(pdist(agentPos[0]))  # nNodes x nNodes
            W[0] = (distances < self.communicationRadius).astype(agentPos.dtype)
            W[0] = W[0] - np.diag(np.diag(W[0]))
            graphConnected = graph.isConnected(W[0])
            deg = np.sum(W[0], axis=1)  # nNodes (degree vector)
            zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)[0]
            deg[zeroDeg] = 1.
            invSqrtDeg = np.sqrt(1. / deg)
            invSqrtDeg[zeroDeg] = 0.
            Deg = np.diag(invSqrtDeg)
            W[0] = Deg @ W[0] @ Deg

        return W, self.communicationRadius, graphConnected

    def getGSO(self, step):
        list_PosAgents = []
        action_CurrentAgents=[]
        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)
            currentState_predict = self.status_MultiAgent[name_agent]["currentState"]
            currentPredictIndexX = int(currentState_predict[0][0])
            currentPredictIndexY = int(currentState_predict[0][1])
            action_CurrentAgents.append([currentPredictIndexX, currentPredictIndexY])
        list_PosAgents.append(action_CurrentAgents)
        store_PosAgents = np.asarray(list_PosAgents)

        if step == 0:
            self.initCommunicationRadius()
        # print("{} - Step-{} - initCommunication Radius:{}".format(self.ID_dataset, step, self.communicationRadius))

        # comm radius fixed
        # GSO, communicationRadius, graphConnected = self.computeAdjacencyMatrix_fixedCommRadius(step, store_PosAgents, self.communicationRadius)

        # comm radius that ensure initial graph connected
        GSO, communicationRadius, graphConnected = self.computeAdjacencyMatrix(step, store_PosAgents, self.communicationRadius)
        GSO_tensor = torch.from_numpy(GSO)

        self.store_GSO.append(GSO)
        self.store_communication_radius.append(communicationRadius)

        # print("{} - Step-{} - Communication Radius:{} - graphConnected:{}".format(self.ID_dataset, step, communicationRadius, graphConnected))
        return GSO_tensor

    def getCurrentState__(self):

        tensor_currentState = torch.zeros([1, self.config.num_agents, 3, self.size_map[0], self.size_map[1]])
        # tensor_currentState_all = torch.zeros([1, self.size_map[0], self.size_map[1]])
        for id_agent in range(self.config.num_agents):

            name_agent = "agent{}".format(id_agent)

            goal_CurrentAgent = self.status_MultiAgent[name_agent]["goal"]
            goalIndexX = int(goal_CurrentAgent[0][0])
            goalIndexY = int(goal_CurrentAgent[0][1])
            channel_goal = torch.zeros([self.size_map[0], self.size_map[1]])

            currentState_predict = self.status_MultiAgent[name_agent]["currentState"]
            currentPredictIndexX = int(currentState_predict[0][0])
            currentPredictIndexY = int(currentState_predict[0][1])
            channel_state = torch.zeros([self.size_map[0], self.size_map[1]])

            channel_goal[goalIndexX][goalIndexY] = 1
            channel_state[currentPredictIndexX][currentPredictIndexY] = 1

            tensor_currentState[0, id_agent, 0, :, :] = self.channel_map
            tensor_currentState[0, id_agent, 1, :, :] = channel_goal
            tensor_currentState[0, id_agent, 2, :, :] = channel_state
            # tensor_currentState_allagents = torch.add(tensor_currentState_all, channel_state)
        # print(tensor_currentState_allagents)
        return tensor_currentState


    def getCurrentState(self, return_GPos=False):


        store_goalAgents = torch.zeros([self.config.num_agents, 2])
        store_stateAgents = torch.zeros([self.config.num_agents, 2])

        for id_agent in range(self.config.num_agents):

            name_agent = "agent{}".format(id_agent)

            goal_CurrentAgent = self.status_MultiAgent[name_agent]["goal"]
            goalIndexX = int(goal_CurrentAgent[0][0])
            goalIndexY = int(goal_CurrentAgent[0][1])
            store_goalAgents[id_agent,:] = torch.FloatTensor([goalIndexX,goalIndexY])

            currentState_predict = self.status_MultiAgent[name_agent]["currentState"]
            currentPredictIndexX = int(currentState_predict[0][0])
            currentPredictIndexY = int(currentState_predict[0][1])

            store_stateAgents[id_agent, :] = torch.FloatTensor([currentPredictIndexX, currentPredictIndexY])

        tensor_currentState = self.AgentState.toInputTensor(store_goalAgents, store_stateAgents)
        tensor_currentState = tensor_currentState.unsqueeze(0)
        # print(tensor_currentState_allagents)

        if return_GPos:
            return tensor_currentState, store_stateAgents.unsqueeze(0)
        else:
            return tensor_currentState

    def getCurrentState_(self):

        tensor_currentState = self.AgentState.toInputTensor(self.store_goalAgents, self.store_stateAgents)
        tensor_currentState = tensor_currentState.unsqueeze(0)
        return tensor_currentState


    def interRobotCollision(self):

        # collision = 0
        collision = False

        allagents_pos = {}
        list_pos = []
        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)

            nextstate_currrentAgent = self.status_MultiAgent[name_agent]["nextState_predict"].tolist()
            list_pos.append(nextstate_currrentAgent)
            allagents_pos.update({id_agent: nextstate_currrentAgent})

        for i in range(self.config.num_agents):
            pos = list_pos[i]
            count_collision = list_pos.count(pos)
            if count_collision > 1:
                collision = True
                collided_agents = []

                for id_agent, pos_agent in allagents_pos.items():
                    if pos_agent == pos:
                        name_agent = "agent{}".format(id_agent)
                        collided_agents.append(name_agent)

                # id_agent2move = max(heuristic_agents.items(), key=operator.itemgetter(1))[0]
                id_agent2move = random.choice(collided_agents)
                # print("In {}, {} need to move".format(collided_agents, id_agent2move))
                for name_agent in collided_agents:
                    # name_agent = "agent{}".format(id_agent)
                    # print("The action list of {}:\n{}".format(name_agent,self.status_MultiAgent[name_agent]["action_predict"]))
                    list_actionKey_predict = self.status_MultiAgent[name_agent]["action_predict"]

                    if list_actionKey_predict[-1] == self.stopKeyValue:
                        # print('##### one of the agent has stoppted.#####')
                        for name_agent in collided_agents:
                            list_actionKey_predict = self.status_MultiAgent[name_agent]["action_predict"]
                            list_actionKey_predict[-1] = self.stopKeyValue
                            self.status_MultiAgent[name_agent]["action_predict"] = list_actionKey_predict
                            self.status_MultiAgent[name_agent]["nextState_predict"] = \
                            self.status_MultiAgent[name_agent]["currentState"]
                            # print("All agents {} stops:\n{}\n".format(name_agent,self.status_MultiAgent[name_agent]["action_predict"]))
                            id_agent = int(name_agent.replace("agent", ""))
                            list_pos[id_agent] = self.status_MultiAgent[name_agent]["nextState_predict"].tolist()
                    else:

                        if name_agent != id_agent2move:
                            list_actionKey_predict = self.status_MultiAgent[name_agent]["action_predict"]
                            list_actionKey_predict[-1] = self.stopKeyValue
                            self.status_MultiAgent[name_agent]["action_predict"] = list_actionKey_predict
                            self.status_MultiAgent[name_agent]["nextState_predict"] = \
                            self.status_MultiAgent[name_agent]["currentState"]
                            id_agent = int(name_agent.replace("agent", ""))
                            list_pos[id_agent] = self.status_MultiAgent[name_agent]["nextState_predict"].tolist()
                        #     print('{} stop'.format(name_agent))
                        # print("The action list of {} after changed:\n{}\n".format(name_agent, self.status_MultiAgent[name_agent]["action_predict"]))

        ## position swap
        list_nextpos = []

        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)

            nextstate_currrentAgent = self.status_MultiAgent[name_agent]["nextState_predict"].tolist()
            list_nextpos.append(nextstate_currrentAgent)

        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)
            currentstate_currrentAgent = self.status_MultiAgent[name_agent]["currentState"].tolist()
            if currentstate_currrentAgent in list_nextpos:
                id_agent_swap = list_nextpos.index(currentstate_currrentAgent)
                name_agent_swap = "agent{}".format(id_agent_swap)
                if name_agent_swap != name_agent:
                    if self.status_MultiAgent[name_agent_swap]["currentState"].tolist() == \
                            self.status_MultiAgent[name_agent]["nextState_predict"].tolist():
                        # print("In #{}case(test), {} and {} swap position happens.".format(self.ID_dataset,name_agent,name_agent_swap))
                        self.status_MultiAgent[name_agent]["nextState_predict"] = self.status_MultiAgent[name_agent][
                            "currentState"]
                        self.status_MultiAgent[name_agent_swap]["nextState_predict"] = \
                        self.status_MultiAgent[name_agent_swap]["currentState"]

                        id_agent = int(name_agent.replace("agent", ""))
                        list_pos[id_agent] = self.status_MultiAgent[name_agent]["nextState_predict"].tolist()

                        id_agent_swap = int(name_agent_swap.replace("agent", ""))
                        list_pos[id_agent_swap] = self.status_MultiAgent[name_agent_swap]["nextState_predict"].tolist()

                        self.status_MultiAgent[name_agent]["action_predict"][-1] = self.stopKeyValue
                        self.status_MultiAgent[name_agent_swap]["action_predict"][-1] = self.stopKeyValue

                        collision = True

        return collision

    def heuristic(self, current_pos, goal):

        value = abs(goal[0] - current_pos[0]) + abs(goal[1] - current_pos[1])
        return value

    def move(self, actionVec, currentstep):
        #print("Orignal multirobotsim")
        allReachGoal = all(self.count_reachgoal)
        allReachGoal_withoutcollision = False

        self.check_predictCollsion = False
        self.check_moveCollision = False

        if (not allReachGoal) or (currentstep < self.maxstep):
        # if not allReachGoal and currentstep < self.maxstep:
        #     print("####### Case{} \t Step{}/{}\t- AllReachGoal\t-{}".format(self.ID_dataset, currentstep, self.maxstep, allReachGoal))
            t0_all_agent_move = time.process_time()

            for id_agent in range(self.config.num_agents):
                name_agent = "agent{}".format(id_agent)

                # disabled_actionPredict_currentAgent = self.status_MultiAgent[name_agent]["disabled_action_predict"]
                # if self.config.num_agents == 1:
                #     actionVec_predict_CurrentAgents = torch.mul(self.fun_Softmax(actionVec),
                #                                                 disabled_actionPredict_currentAgent)
                # else:
                #     # actionVec_tmp = actionVec[id_agent]
                #     actionVec_current = self.fun_Softmax(actionVec[id_agent])
                #     actionVec_predict_CurrentAgents = torch.mul(actionVec_current, disabled_actionPredict_currentAgent)

                step_agent_move = time.process_time()

                actionVec_current = self.fun_Softmax(actionVec[id_agent])

                actionKey_predict = torch.max(actionVec_current, 1)[1]

                # set flag of the timestep that agent start to move
                check_move = (actionKey_predict != self.stopKeyValue)

                startStep_action = self.status_MultiAgent[name_agent]["startStep_action_predict"]


                if check_move == 1 and startStep_action is None:
                    self.status_MultiAgent[name_agent]["startStep_action_predict"] = currentstep - 1

                list_actionKey_predict = self.status_MultiAgent[name_agent]["action_predict"]

                currentState_predict = self.status_MultiAgent[name_agent]["currentState"]
                nextState_predict = torch.add(currentState_predict, self.delta[actionKey_predict])

                deltaT_agent_move = time.process_time() - step_agent_move
                #print(" Computation time \t\t-[{}-move]-\t\t :{} ".format(name_agent, deltaT_agent_move))

                # ----- check edge and obstacle
                t0_agent_check_Edge = time.process_time()
                checkEdge = self.reachEdge(nextState_predict)
                deltaT_agent_check_Edge = time.process_time() - t0_agent_check_Edge
                #print(" Computation time \t\t-[{}-checkEdge]-\t\t :{} ".format(name_agent, deltaT_agent_check_Edge))

                t0_agent_check_Obs = time.process_time()
                if not checkEdge:
                    checkObstacle = self.reachObstacle(nextState_predict)
                deltaT_agent_check_Obs = time.process_time() - t0_agent_check_Obs
                #print(" Computation time \t\t-[{}-checkObs]-\t\t :{} ".format(name_agent, deltaT_agent_check_Obs))

                if checkEdge or checkObstacle:
                    # print('Reach obstacle or edge.')
                    # break
                    # todo : remove the collision motion disabled
                    # disabled_actionPredict_currentAgent[actionKey_predict] = 0.0
                    # self.status_MultiAgent[name_agent]["disabled_action_predict"] = disabled_actionPredict_currentAgent
                    # self.move(actionVec, currentstep)
                    self.check_predictCollsion = True

                    list_actionKey_predict.append(self.stopKeyValue)
                    self.status_MultiAgent[name_agent]["action_predict"] = list_actionKey_predict
                    self.status_MultiAgent[name_agent]["nextState_predict"] = currentState_predict
                    # self.check_predictEdgeCollsion
                else:
                    # self.status_MultiAgent[name_agent]["currentState"] = nextState_predict
                    self.status_MultiAgent[name_agent]["nextState_predict"] = nextState_predict
                    # self.status_MultiAgent[name_agent]["disabled_action_predict"] = self.reset_disabled_action

                    list_actionKey_predict.append(actionKey_predict[0])
                    self.status_MultiAgent[name_agent]["action_predict"] = list_actionKey_predict

                # if not self.check_predictCollsion:

            deltaT_allagent_move = time.process_time() - t0_all_agent_move
            # print(" Computation time \t\t-[allMove]-\t\t :{} ".format(deltaT_allagent_move))


            t0_agent_interCollsion = time.process_time()
            detect_interRobotCollision = self.interRobotCollision()
            deltaT_agent_interCollsion = time.process_time() - t0_agent_interCollsion
            # print(" Computation time \t\t-[checkIRCollision]-\t\t :{} ".format(deltaT_agent_interCollsion))


            # while detect_interRobotCollision:
            for _ in range(self.config.num_agents):
                # print('Collision happens.')
                if detect_interRobotCollision:
                    detect_interRobotCollision = self.interRobotCollision()
                    self.check_predictCollsion = True
                    # print("Collision happens")
                else:
                    # print("Collision avoided by collision shielding")
                    break

            self.check_moveCollision = self.interRobotCollision()

            for id_agent in range(self.config.num_agents):
                name_agent = "agent{}".format(id_agent)
                nextState_predict = self.status_MultiAgent[name_agent]["nextState_predict"]

                self.status_MultiAgent[name_agent]["currentState"] = nextState_predict
                # self.store_stateAgents[id_agent,:] = nextState_predict
                path_predict = self.status_MultiAgent[name_agent]["path_predict"]
                path_predict.update({currentstep: nextState_predict})
                self.status_MultiAgent[name_agent]["path_predict"] = path_predict

                goal_CurrentAgent = self.status_MultiAgent[name_agent]["goal"]
                goalIndexX = int(goal_CurrentAgent[0][0])
                goalIndexY = int(goal_CurrentAgent[0][1])

                if nextState_predict[0][0] == goalIndexX and nextState_predict[0][1] == goalIndexY and not \
                self.count_reachgoal[id_agent]:
                    self.count_reachgoal[id_agent] = True
                    self.status_MultiAgent[name_agent]["endStep_action_predict"] = currentstep
                if currentstep >= (self.maxstep) and not self.count_reachgoal[id_agent]:
                    # self.count_reachgoal[id_agent] = False
                    self.status_MultiAgent[name_agent]["endStep_action_predict"] = currentstep
                    # print("\t \t {} - status(Reach Goal) - {}".format(name_agent, self.count_reachgoal[id_agent]))
                    if self.status_MultiAgent[name_agent]["startStep_action_predict"] is None:
                        self.status_MultiAgent[name_agent]["startStep_action_predict"] =  0 # currentstep #

        if allReachGoal or (currentstep >= self.maxstep):
        # else:
            List_endStep = []
            List_startStep = []
            self.flowtimePredict = 0

            # if (currentstep >= self.maxstep):
            #     print("################## End of loop ################## ")
            #     print("####### Case{} \t Step{}/{}\t- AllReachGoal-{}".format(self.ID_dataset, currentstep, self.maxstep,
            #                                                               allReachGoal))
            for id_agent in range(self.config.num_agents):
                name_agent = "agent{}".format(id_agent)
                List_endStep.append(self.status_MultiAgent[name_agent]["endStep_action_predict"])
                List_startStep.append(self.status_MultiAgent[name_agent]["startStep_action_predict"])
                # print("{}- \t Start Step: {} \t End Step: {} \n".format(name_agent, self.status_MultiAgent[name_agent]["startStep_action_predict"], self.status_MultiAgent[name_agent]["endStep_action_predict"]))
                self.flowtimePredict += self.status_MultiAgent[name_agent]["endStep_action_predict"] - \
                                        self.status_MultiAgent[name_agent]["startStep_action_predict"]

                len_action_predict = self.status_MultiAgent[name_agent]["endStep_action_predict"] - \
                                     self.status_MultiAgent[name_agent]["startStep_action_predict"]
                self.status_MultiAgent[name_agent]["len_action_predict"] = len_action_predict

            self.makespanPredict = max(List_endStep) - min(List_startStep)

            # if (currentstep >= self.maxstep):
            #     print("\t\t Makespan(Predict/Target) {}/{} \t flowtime(Predict/Target){}/{} \n\n ".format(self.makespanPredict, self.makespanTarget,
            #                                                                                            self.flowtimePredict, self.flowtimeTarget))



        return allReachGoal, self.check_moveCollision, self.check_predictCollsion

    def count_numAgents_ReachGoal(self):
        return self.count_reachgoal.count(True)

    def count_GSO_communcationRadius(self, step):
        _ = self.getGSO(step)
        return self.store_GSO, self.store_communication_radius



    def save_success_cases(self, mode):

        inputfile_name = os.path.join(self.config.result_AnimeDemo_dir_input, '{}Cases_ID{:05d}.yaml'.format(mode, self.ID_dataset))
        if mode == 'success':
            outputfile_name = os.path.join(self.config.result_AnimeDemo_dir_predict_success, '{}Cases_ID{:05d}.yaml'.format(mode,self.ID_dataset))
        else:
            outputfile_name = os.path.join(self.config.result_AnimeDemo_dir_predict_failure,
                                           '{}Cases_ID{:05d}.yaml'.format(mode, self.ID_dataset))

        targetfile_name = os.path.join(self.config.result_AnimeDemo_dir_target,
                                       '{}Cases_ID{:05d}.yaml'.format(mode, self.ID_dataset))

        gsofile_name = os.path.join(self.config.result_AnimeDemo_dir_GSO,
                                       '{}Cases_ID{:05d}.mat'.format(mode, self.ID_dataset))

        save_statistics_GSO = {'gso':self.store_GSO, 'commRadius': self.store_communication_radius}
        sio.savemat(gsofile_name, save_statistics_GSO)

        # print('############## successCases in training set ID{} ###############'.format(self.ID_dataset))
        f = open(inputfile_name, 'w')
        f.write("map:\n")
        f.write("    dimensions: {}\n".format([self.size_map[0], self.size_map[1]]))
        f.write("    obstacles:\n")
        for ID_obs in range(self.numObstacle):
            obstacleIndexX = int(self.posObstacle[ID_obs][0].cpu().detach().numpy())
            obstacleIndexY = int(self.posObstacle[ID_obs][1].cpu().detach().numpy())
            list_obs = [obstacleIndexX, obstacleIndexY]
            f.write("    - {}\n".format(list_obs))
        f.write("agents:\n")
        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)
            log_goal_currentAgent = self.status_MultiAgent[name_agent]["goal"].cpu().detach().numpy()
            log_currentState_currentAgent = self.status_MultiAgent[name_agent]["start"].cpu().detach().numpy()
            goalX = int(log_goal_currentAgent[0][0])
            goalY = int(log_goal_currentAgent[0][1])
            startX = int(log_currentState_currentAgent[0][0])
            startY = int(log_currentState_currentAgent[0][1])
            goal_currentAgent = [goalX, goalY]
            currentState_currentAgent = [startX, startY]
            f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(id_agent, currentState_currentAgent,
                                                                              goal_currentAgent))
        f.close()

        f_sol = open(outputfile_name, 'w')
        f_sol.write("statistics:\n")
        f_sol.write("    cost: {}\n".format(self.flowtimePredict))
        f_sol.write("    makespan: {}\n".format(self.makespanPredict))
        f_sol.write("schedule:\n")

        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)
            # print(self.status_MultiAgent[name_agent]["path_predict"])
            path = self.status_MultiAgent[name_agent]["path_predict"]

            len_path = len(path)

            f_sol.write("    agent{}:\n".format(id_agent))
            for step in range(len_path):


                pathIndexX = int(path[step][0][0].cpu().detach().numpy())
                pathIndexY = int(path[step][0][1].cpu().detach().numpy())

                f_sol.write("       - x: {}\n         y: {}\n         t: {}\n".format(pathIndexX,pathIndexY, step))
        f_sol.close()

        f_target = open(targetfile_name, 'w')
        f_target.write("statistics:\n")
        f_target.write("    cost: {}\n".format(self.flowtimeTarget))
        f_target.write("    makespan: {}\n".format(self.makespanTarget))
        f_target.write("schedule:\n")

        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)
            # print(self.status_MultiAgent[name_agent]["path_predict"])
            path = self.status_MultiAgent[name_agent]["path_target"]

            len_path = len(path)

            f_target.write("    agent{}:\n".format(id_agent))
            for step in range(len_path):
                pathIndexX = int(path[step][0][0].cpu().detach().numpy())
                pathIndexY = int(path[step][0][1].cpu().detach().numpy())

                f_target.write("       - x: {}\n         y: {}\n         t: {}\n".format(pathIndexX, pathIndexY, step))
        f_target.close()





    def checkOptimality(self, collisionFreeSol):

        if self.makespanPredict <= self.makespanTarget and self.flowtimePredict <= self.flowtimeTarget and collisionFreeSol:
            findOptimalSolution = True
        else:
            findOptimalSolution = False

        return findOptimalSolution, [self.makespanPredict, self.makespanTarget], [self.flowtimePredict, self.flowtimeTarget]

    def draw(self, ID_dataset):
        status_MultiAgent = {}
        status_MultiAgent_Target = {}
        status_MultiAgent_Predict = {}
        for id_agent in range(self.config.num_agents):
            name_agent = "agent{}".format(id_agent)
            status_CurrentAgents_Target = {"goal": self.status_MultiAgent[name_agent]["goal"],
                                           "start": self.status_MultiAgent[name_agent]["start"],
                                           "path": self.status_MultiAgent[name_agent]["path_target"],
                                           "action": self.status_MultiAgent[name_agent]["action_target"],
                                           "len_action": self.status_MultiAgent[name_agent]["len_action_target"]
                                           }
            status_CurrentAgents_Predict = {"goal": self.status_MultiAgent[name_agent]["goal"],
                                            "start": self.status_MultiAgent[name_agent]["start"],
                                            "path": self.status_MultiAgent[name_agent]["path_predict"],
                                            "action": self.status_MultiAgent[name_agent]["action_predict"],
                                            "len_action": self.status_MultiAgent[name_agent]["len_action_predict"]
                                            }

            status_MultiAgent_Target.update({name_agent: status_CurrentAgents_Target})
            status_MultiAgent_Predict.update({name_agent: status_CurrentAgents_Predict})

        status_MultiAgent_Target.update({"makespan": self.makespanTarget, "flowtime": self.flowtimeTarget})
        status_MultiAgent_Predict.update({"makespan": self.makespanPredict, "flowtime": self.flowtimePredict})

        status_MultiAgent.update({"target":status_MultiAgent_Target, "predict":status_MultiAgent_Predict})
        draw = DrawpathCombine(self.config, self.channel_map, self.posObstacle, status_MultiAgent)

        draw.draw(ID_dataset)
        draw.save()



