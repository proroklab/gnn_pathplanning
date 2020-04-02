"""
This file will contain the metrics of the framework
"""

import os
import torch
import numpy as np
import scipy.io as sio


class MonitoringMultiAgentPerformance:

    def __init__(self, config):
        self.config = config
        self.count_validset = None
        self.count_reachGoal = None
        self.count_collisionFreeSol = None
        self.count_CollisionPredictedinLoop = None
        self.count_findOptimalSolution = None
        self.reachGoal_cases = None
        self.reachGoal_cases_order = None
        self.findShortestPath_cases = None
        self.findShortestPath_cases_order = None
        self.increasePathRate_cases = None
        self.increasePathRate_cases_order = None
        self.List_MeanincreasePathRate = None

        self.rateReachGoal = None
        self.ratefindOptimalSolution = None
        self.rateCollsionFreeSol = None
        self.rateCollisionPredictedinLoop = None

        self.RateIncreasePathLen = None

        self.makespanPredict = None
        self.makespanTarget = None
        self.flowtimePredict = None
        self.flowtimeTarget = None

        self.list_reachGoal = None
        self.list_compareMP = None
        self.list_compareFT = None

        self.list_rate_deltaMP = None
        self.list_rate_deltaFT = None

        self.avg_rate_deltaMP = None
        self.avg_rate_deltaFT = None
        self.std_rate_deltaMP = None
        self.std_rate_deltaFT = None

        self.save_statistics = None

        self.list_computationTime = None

    def reset(self):
        self.count_validset = 0
        self.count_reachGoal = 0
        self.count_noReachGoalSH = 0
        self.count_collisionFreeSol = 0
        self.count_CollisionPredictedinLoop = 0
        self.count_findOptimalSolution = 0

        self.rateReachGoal = 0
        self.rateFailedReachGoalSH = 0
        self.ratefindOptimalSolution = 0
        self.rateCollsionFreeSol = 0
        self.rateCollisionPredictedinLoop = 0

        self.reachGoal_cases = {}
        self.reachGoal_cases_order = {}
        self.findShortestPath_cases = {}
        self.findShortestPath_cases_order = {}
        self.increasePathRate_cases = {}
        self.increasePathRate_cases_order = {}
        self.List_MeanincreasePathRate = []

        self.list_reachGoal = []
        self.list_noReachGoalSH = []

        self.list_MP_predict = []
        self.list_MP_target = []
        self.list_FT_predict = []
        self.list_FT_target = []

        self.list_compareMP = []
        self.list_compareFT = []

        self.listCase_GSO = []
        self.listCase_commRadius = []

        self.list_rate_deltaMP = []
        self.list_numAgentReachGoal = []
        self.list_rate_deltaFT = []
        self.avg_rate_deltaMP = 0
        self.avg_rate_deltaFT = 0
        self.std_rate_deltaMP = 0
        self.std_rate_deltaFT = 0

        self.list_computationTime = []
        self.list_ForwardPassTime = []
        self.save_statistics = {}

    def update(self, maxstep, log_result):

        [allReachGoal, noReachGoalbyCollsionShielding, findOptimalSolution, check_collisionFreeSol, check_CollisionPredictedinLoop, compare_makespan, compare_flowtime, num_agents_reachgoal, storeCase_GSO, storeCase_communication_radius, time_record , Time_cases_ForwardPass] = log_result
        [self.makespanPredict, self.makespanTarget] = compare_makespan
        [self.flowtimePredict, self.flowtimeTarget] = compare_flowtime

        rate_deltaMP = abs(self.makespanPredict - self.makespanTarget) / self.makespanTarget
        rate_deltaFT = abs(self.flowtimePredict - self.flowtimeTarget) / self.flowtimeTarget

        self.list_MP_predict.append(self.makespanPredict)
        self.list_MP_target.append(self.makespanTarget)
        self.list_FT_predict.append(self.flowtimePredict)
        self.list_FT_target.append(self.flowtimeTarget)

        self.listCase_GSO.append(storeCase_GSO)
        self.listCase_commRadius.append(storeCase_communication_radius)

        self.list_compareMP.append(compare_makespan)
        self.list_compareFT.append(compare_flowtime)

        self.list_rate_deltaMP.append(rate_deltaMP)
        self.list_rate_deltaFT.append(rate_deltaFT)

        self.list_computationTime.append(time_record)
        self.list_ForwardPassTime.append(Time_cases_ForwardPass)

        if allReachGoal:
            self.count_reachGoal += 1
            self.list_reachGoal.append(1)
        else:
            self.list_reachGoal.append(0)

        self.list_numAgentReachGoal.append(num_agents_reachgoal)
        if noReachGoalbyCollsionShielding:
            self.count_noReachGoalSH += 1
            self.list_noReachGoalSH.append(1)
        else:
            self.list_noReachGoalSH.append(0)

        if findOptimalSolution:
            self.count_findOptimalSolution += 1

        if check_collisionFreeSol:
            self.count_collisionFreeSol += 1

        if check_CollisionPredictedinLoop:
            self.count_CollisionPredictedinLoop += 1

        self.count_validset += 1

    def summary(self, label, summary_writer, current_epoch):
        self.rateReachGoal = self.count_reachGoal / self.count_validset

        self.rateFailedReachGoalSH = self.count_noReachGoalSH / self.count_validset

        self.ratefindOptimalSolution = self.count_findOptimalSolution / self.count_validset

        self.rateCollsionFreeSol = self.count_collisionFreeSol / self.count_validset
        self.rateCollisionPredictedinLoop = self.count_CollisionPredictedinLoop / self.count_validset


        # self.avg_rate_deltaMP = sum(self.list_rate_deltaMP) / self.count_validset
        # self.avg_rate_deltaFT = sum(self.list_rate_deltaFT) / self.count_validset


        self.array_rate_deltaMP = np.array(self.list_rate_deltaMP)
        self.array_rate_deltaFT = np.array(self.list_rate_deltaFT)

        self.avg_rate_deltaMP = np.mean(self.array_rate_deltaMP)
        # sample std
        self.std_rate_deltaMP = np.std(self.array_rate_deltaMP, ddof=1)

        self.avg_rate_deltaFT = np.mean(self.array_rate_deltaFT)
        self.std_rate_deltaFT = np.std(self.array_rate_deltaFT, ddof=1)
        # sample std



        if label == 'test':
            self.test_summary(summary_writer)

        else:
            summary_writer.add_scalar("epoch/{}_set_Accuracy_reachGoalNoCollision".format(label),
                                      self.rateReachGoal, current_epoch)

            summary_writer.add_scalar("epoch/{}_set_DeteriorationRate_MakeSpan".format(label),
                                      self.avg_rate_deltaMP, current_epoch)

            summary_writer.add_scalar("epoch/{}_set_DeteriorationRate_FlowTime".format(label),
                                      self.avg_rate_deltaFT, current_epoch)

            summary_writer.add_scalar("epoch/{}_set_Rate_CollisionPredictedinLoop".format(label),
                                      self.rateCollisionPredictedinLoop, current_epoch)

            summary_writer.add_scalar("epoch/{}_set_Rate_FailedReachGoalSH".format(label),
                                      self.rateFailedReachGoalSH, current_epoch)


        return summary_writer

    def test_summary(self, summary_writer):
        label = 'test'
        # label = 'train'

        self.count_numAgentReachGoal = [] #= np.zeros([1, self.config.num_agents])
        for i in range(self.config.num_agents+1):
            self.count_numAgentReachGoal.append(self.list_numAgentReachGoal.count(i))


        summary_writer.add_scalar("{}_set/Accuracy_reachGoalNoCollision".format(label),
                                  self.rateReachGoal, self.config.num_agents)

        summary_writer.add_scalar("{}_set/Rate_FailedReachGoalbyCollsionShielding".format(label),
                                  self.rateFailedReachGoalSH, self.config.num_agents)

        summary_writer.add_scalar("{}_set/DeteriorationRate_MakeSpan".format(label),
                                  self.avg_rate_deltaMP, self.config.num_agents)

        summary_writer.add_scalar("{}_set/DeteriorationRate_FlowTime".format(label),
                                  self.avg_rate_deltaFT, self.config.num_agents)


        exp_setup = '{}{}x{}_rho{}_{}Agent'.format(self.config.map_type, self.config.map_w, self.config.map_w, self.config.map_density, self.config.num_agents)
        dir_name = os.path.join(self.config.result_statistics_dir, self.config.exp_net, exp_setup)
        try:
            # Create target Directory
            os.makedirs(dir_name)
            print("Directory ", dir_name, " Created ")
        except FileExistsError:
            pass

        self.save_statistics.update({'exp_net': self.config.exp_net,
                                     'exp_stamps': self.config.exp_time,
                                     'commRadius': self.config.commR,
                                     'map_size_trained':[self.config.trained_map_w,self.config.trained_map_w],
                                     'map_density_trained':self.config.trained_map_density,
                                     'num_agents_trained': self.config.trained_num_agents,
                                     'map_size_testing': [self.config.map_w, self.config.map_h],
                                     'map_density_testing': self.config.map_density,
                                     'num_agents_testing': self.config.num_agents,

                                     'K': self.config.nGraphFilterTaps,
                                     'hidden_state':self.config.hiddenFeatures,

                                     'rate_ReachGoal': self.rateReachGoal,
                                     'num_ReachGoal': self.count_reachGoal,
                                     'rate_notReachGoalSH': self.rateFailedReachGoalSH,
                                     'num_notReachGoalSH': self.count_noReachGoalSH,
                                     'list_reachGoal': self.list_reachGoal,
                                     'list_noReachGoalSH': self.list_noReachGoalSH,
                                     'list_numAgentReachGoal': self.list_numAgentReachGoal,
                                     'hist_numAgentReachGoal': self.count_numAgentReachGoal,

                                     'list_MP_predict': self.list_MP_predict,
                                     'list_MP_target': self.list_MP_target,
                                     'list_FT_predict': self.list_FT_predict,
                                     'list_FT_target': self.list_FT_target,

                                     'listCase_GSO': self.listCase_GSO,
                                     'listCase_commRadius': self.listCase_commRadius,
                                     'list_computationTime': self.list_computationTime,
                                     'list_ForwardPassTime':self.list_ForwardPassTime,

                                     'list_compareMP': self.list_compareMP,
                                     'list_compareFT': self.list_compareFT,
                                     'list_deltaMP': self.array_rate_deltaMP,
                                     'mean_deltaMP': self.avg_rate_deltaMP,
                                     'std_deltaMP': self.std_rate_deltaMP,
                                     'list_deltaFT': self.array_rate_deltaFT,
                                     'mean_deltaFT': self.avg_rate_deltaFT,
                                     'std_deltaFT': self.std_rate_deltaFT,
                                     'num_CollisionPredicted': self.count_CollisionPredictedinLoop,
                                     'num_validset': self.count_validset,
                                     })

        # save the result of inference stage
        exp_HyperPara = "{}_K{}_HS{}_".format(self.config.exp_net, self.config.nGraphFilterTaps, self.config.hiddenFeatures)
        exp_Setup_training = "TR_M{}p{}_{}Agent_".format(self.config.trained_map_w, self.config.trained_map_density,
                                                                       self.config.trained_num_agents)
        exp_Setup_testing = "TE_M{}p{}_{}Agent_".format(self.config.map_w, self.config.map_density,
                                                                       self.config.num_agents)

        dsecription = exp_HyperPara + exp_Setup_training + exp_Setup_testing + "{}".format(self.config.exp_time)
        file_name = os.path.join(dir_name,'statistics_{}_comR_{}.mat'.format(dsecription,self.config.commR))
        sio.savemat(file_name, self.save_statistics)