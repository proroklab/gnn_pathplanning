import numpy as np
import torch


class AgentState:
    def __init__(self, num_agents):
        # self.config = config
        # self.num_agents = self.config.num_agents
        self.num_agents = num_agents
        # self.FOV = 5
        self.FOV = 9
        self.FOV_width = int(self.FOV/2)
        self.border = 1
        self.W = self.FOV + 2
        self.H = self.FOV + 2
        self.dist = int(np.floor(self.W/2))
        self.border_down = 0
        self.border_left = 0

        self.centerX = self.dist #+ 1
        self.centerY = self.dist #+ 1
        self.map_pad = None

    def pad_with(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def setmap(self, map_channel):
        self.map_global = map_channel
        self.map_pad = np.pad(map_channel, self.FOV_width, self.pad_with, padder=1)

    def setPosAgents(self, state_allagents):
        # the second channel represent position of local agent and agents within FOV
        # channel_allstate = np.zeros([self.W, self.H], dtype=np.int64)
        channel_allstate = np.zeros_like(self.map_global, dtype=np.int64)

        for id_agent in range(self.num_agents):
            currentX = int(state_allagents[id_agent][0])
            currentY = int(state_allagents[id_agent][1])
            channel_allstate[currentX][currentY] = 1

        channel_allstate_pad = np.pad(channel_allstate, self.FOV_width, self.pad_with, padder=0)

        return channel_allstate_pad

    def projectedgoal(self, goal_FOV, state_agent, goal_agent):
        channel_goal = np.pad(goal_FOV, self.border, self.pad_with, padder=0)

        dy = float(goal_agent[1]-state_agent[1])
        dx = float(goal_agent[0]-state_agent[0])
        y_sign = np.sign(dy)
        x_sign = np.sign(dx)

        # angle between position of agent an goal
        angle = np.arctan2(dy,dx)

        if (angle >= np.pi / 4 and angle <= np.pi * 3 / 4) or (angle >= -np.pi * (3 / 4) and angle <= -np.pi / 4):
            goalY_FOV = int(self.dist * (y_sign + 1))
            goalX_FOV = int(self.centerX + np.round(self.dist * dx / np.abs(dy)))
        else:
            goalX_FOV = int(self.dist * (x_sign + 1))
            goalY_FOV = int(self.centerX + np.round(self.dist * dy / np.abs(dx)))

        channel_goal[goalX_FOV][goalY_FOV] = 1
        return channel_goal

    def stackinfo_(self, goal_allagents, state_allagents):
        input_step = np.stack((goal_allagents, state_allagents),axis=1)

        input_tensor = torch.FloatTensor(input_step)
        return input_tensor

    def stackinfo(self, goal_allagents, state_allagents):

        input_tensor = np.stack((goal_allagents, state_allagents))

        input_tensor = torch.FloatTensor(input_tensor)

        return input_tensor

    def toInputTensor(self, goal_allagents, state_allagents):

        channel_allstate_pad = self.setPosAgents(state_allagents)

        input_step = []
        for id_agent in range(self.num_agents): #range(3,6):#
            input_step_currentAgent = []

            currentX_global = int(state_allagents[id_agent][0])
            currentY_global = int(state_allagents[id_agent][1])
            goalX_global = int(goal_allagents[id_agent][0])
            goalY_global = int(goal_allagents[id_agent][1])


            # check position
            FOV_X = [currentX_global, currentX_global + 2*self.FOV_width + 1]
            FOV_Y = [currentY_global, currentY_global + 2*self.FOV_width + 1]

            channel_state_FOV = channel_allstate_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
            channel_state = np.pad(channel_state_FOV, self.border, self.pad_with, padder=0)

            channel_map_FOV = self.map_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
            channel_map = np.pad(channel_map_FOV, self.border, self.pad_with, padder=0)


            # channel_goal = np.zeros([self.W, self.H], dtype=np.int64)
            channel_goal_global = np.zeros_like(self.map_global, dtype=np.int64)
            channel_goal_global[goalX_global][goalY_global] = 1
            channel_goal_pad = np.pad(channel_goal_global, self.FOV_width, self.pad_with, padder=0)
            channel_goal_FOV = channel_goal_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
            if (channel_goal_FOV>0).any():
                channel_goal = np.pad(channel_goal_FOV, self.border, self.pad_with, padder=0)
            else:
                channel_goal = self.projectedgoal(channel_goal_FOV, [currentX_global, currentY_global], [goalX_global,goalY_global])

            # print("Agent-{}".format(id_agent))
            # # print("----------- Map -----------\n", channel_map)
            # channel_goal_global[currentX_global-self.FOV_width:currentX_global+1+self.FOV_width, currentY_global-self.FOV_width:currentY_global+1+self.FOV_width] = -1
            # channel_goal_global[currentX_global][currentY_global] = 2
            # print("----------- Goal -----------\n", channel_goal_global)
            # print("\n",channel_goal)
            # print("----------- State -----------\n", channel_state)
            input_step_currentAgent.append(channel_map)
            input_step_currentAgent.append(channel_goal)
            input_step_currentAgent.append(channel_state)
            input_step.append(input_step_currentAgent)

        input_tensor = torch.FloatTensor(input_step)
        return input_tensor


    def toSeqInputTensor(self, goal_allagents, state_AgentsSeq, makespan):

        list_input = []

        for step in range(makespan):

            state_allagents = state_AgentsSeq[step][:]
            channel_allstate_pad = self.setPosAgents(state_allagents)

            input_step = []
            for id_agent in range(self.num_agents): #range(3,6):#
                input_step_currentAgent = []

                currentX_global = int(state_allagents[id_agent][0])
                currentY_global = int(state_allagents[id_agent][1])
                goalX_global = int(goal_allagents[id_agent][0])
                goalY_global = int(goal_allagents[id_agent][1])


                # check position
                FOV_X = [currentX_global, currentX_global + 2*self.FOV_width + 1]
                FOV_Y = [currentY_global, currentY_global + 2*self.FOV_width + 1]

                channel_state_FOV = channel_allstate_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
                channel_state = np.pad(channel_state_FOV, self.border, self.pad_with, padder=0)

                channel_map_FOV = self.map_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
                channel_map = np.pad(channel_map_FOV, self.border, self.pad_with, padder=0)


                # channel_goal = np.zeros([self.W, self.H], dtype=np.int64)
                channel_goal_global = np.zeros_like(self.map_global, dtype=np.int64)
                channel_goal_global[goalX_global][goalY_global] = 1
                channel_goal_pad = np.pad(channel_goal_global, self.FOV_width, self.pad_with, padder=0)
                channel_goal_FOV = channel_goal_pad[FOV_X[0]:FOV_X[1],FOV_Y[0]:FOV_Y[1]]
                if (channel_goal_FOV>0).any():
                    channel_goal = np.pad(channel_goal_FOV, self.border, self.pad_with, padder=0)
                else:
                    channel_goal = self.projectedgoal(channel_goal_FOV, [currentX_global, currentY_global], [goalX_global,goalY_global])

                # print("Agent-{}".format(id_agent))
                # # print("----------- Map -----------\n", channel_map)
                # channel_goal_global[currentX_global-self.FOV_width:currentX_global+1+self.FOV_width, currentY_global-self.FOV_width:currentY_global+1+self.FOV_width] = -1
                # channel_goal_global[currentX_global][currentY_global] = 2
                # print("----------- Goal -----------\n", channel_goal_global)
                # print("\n",channel_goal)
                # print("----------- State -----------\n", channel_state)
                input_step_currentAgent.append(channel_map)
                input_step_currentAgent.append(channel_goal)
                input_step_currentAgent.append(channel_state)
                input_step.append(input_step_currentAgent)
            list_input.append(input_step)

        input_tensor = torch.FloatTensor(list_input)
        return input_tensor