
import shutil

from fnmatch import fnmatch
import os
import time
import torch
from torch.backends import cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
# from torchsummary import summary as modelshow
import torch.nn.functional as F
from torch.autograd import Variable


# import your classes here
from agents.base import BaseAgent
from utils.multirobotsim_dcenlocal_onlineExpert import multiRobotSim
# from utils.multirobotsim_dcenlocal_onlineExpert_anime import multiRobotSim

# from dataloader.decentralplanner_local import DecentralPlannerDataLoader
# from dataloader.decentralplanner_nonTFlocal import DecentralPlannerDataLoader

from onlineExpert.DataTransformer_local_onlineExpert import DataTransformer
from onlineExpert.ECBS_onlineExpert import ComputeECBSSolution
from dataloader.Dataloader_dcplocal_notTF_onlineExpert import DecentralPlannerDataLoader



# whether to use skip connection for feature before and after GNN

from graphs.models.decentralplanner import *


from utils.metrics import MonitoringMultiAgentPerformance

from graphs.losses.cross_entropy import CrossEntropyLoss
from graphs.losses.regularizer import L1Regularizer, L2Regularizer
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class DecentralPlannerAgentLocalWithOnlineExpert(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.onlineExpert = ComputeECBSSolution(self.config)
        self.dataTransformer = DataTransformer(self.config)
        self.recorder = MonitoringMultiAgentPerformance(self.config)

        self.model = DecentralPlannerNet(self.config)
        self.logger.info("Model: \n".format(print(self.model)))

        # define data_loader
        self.data_loader = DecentralPlannerDataLoader(config=config)

        # define loss
        self.loss = CrossEntropyLoss()
        self.l1_reg = L1Regularizer(self.model)
        self.l2_reg = L2Regularizer(self.model)

        # define optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        print(self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.max_epoch, eta_min=1e-6)

        # for param in self.model.parameters():
        #     print(param)

        # for name, param in self.model.state_dict().items():
        #     print(name, param)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.current_iteration_validStep = 0
        self.rateReachGoal = 0.0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            self.config.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.config.device)
            self.loss = self.loss.to(self.config.device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.config.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        if self.config.train_TL or self.config.test_general:
            self.load_pretrained_checkpoint(self.config.test_epoch, lastest=self.config.lastest_epoch, best=self.config.best_epoch)
        else:
            self.load_checkpoint(self.config.test_epoch, lastest=self.config.lastest_epoch, best=self.config.best_epoch)
        # Summary Writer

        self.robot = multiRobotSim(self.config)
        self.switch_toOnlineExpert = False
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='NerualMAPP')
        self.plot_graph = True
        self.save_dump_input = False
        self.dummy_input = None
        self.dummy_gso = None
        self.time_record = None
        # dummy_input = (torch.zeros(self.config.map_w,self.config.map_w, 3),)
        # self.summary_writer.add_graph(self.model, dummy_input)

    def save_checkpoint(self, epoch, is_best=0, lastest=True):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        if lastest:
            file_name = "checkpoint.pth.tar"
        else:
            file_name = "checkpoint_{:03d}.pth.tar".format(epoch)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        # Save the state
        torch.save(state, os.path.join(self.config.checkpoint_dir, file_name))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(os.path.join(self.config.checkpoint_dir, file_name),
                            os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))

    def load_pretrained_checkpoint(self, epoch, lastest=True, best=False):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        if lastest:
            file_name = "checkpoint.pth.tar"
        elif best:
            file_name = "model_best.pth.tar"
        else:
            file_name = "checkpoint_{:03d}.pth.tar".format(epoch)

        filename = os.path.join(self.config.checkpoint_dir_load, file_name)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(filename, map_location='cuda:{}'.format(self.config.gpu_device))

            self.current_epoch = checkpoint['epoch']

            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir_load, checkpoint['epoch'], checkpoint['iteration']))

            if self.config.train_TL:
                param_name_GFL = '*GFL*'
                param_name_action = '*actions*'
                assert param_name_GFL != '', 'you must specified the name of the parameters to be re-trained'
                for model_param_name, model_param_value in self.model.named_parameters():
                    # print("---All layers -- \n", model_param_name)
                    if fnmatch(model_param_name, param_name_GFL) or fnmatch(model_param_name, param_name_action):  # and model_param_name.endswith('weight'):
                        # print("---retrain layers -- \n", model_param_name)
                        model_param_value.requires_grad = True
                    else:
                        # print("---freezed layers -- \n", model_param_name)
                        model_param_value.requires_grad = False


        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")


    def load_checkpoint(self, epoch, lastest=True, best=False):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        if lastest:
            file_name = "checkpoint.pth.tar"
        elif best:
            file_name = "model_best.pth.tar"
        else:
            file_name = "checkpoint_{:03d}.pth.tar".format(epoch)
        filename = os.path.join(self.config.checkpoint_dir, file_name)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(filename, map_location='cuda:{}'.format(self.config.gpu_device))

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        The main operator
        :return:
        """
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'test':
                print("-------test------------")
                start = time.process_time()
                self.test('test')
                self.time_record = time.process_time()-start
                # self.test('test_trainingSet')
                # self.pipeline_onlineExpert(self.current_epoch)
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """

        for epoch in range(self.current_epoch, self.config.max_epoch + 1):
        # for epoch in range(1, self.config.max_epoch + 1):
            self.current_epoch = epoch
            # TODO: Optional 1: del dataloader before train
            self.train_one_epoch()
            self.logger.info('Train {} on Epoch {}: Learning Rate: {}]'.format(self.config.exp_name, self.current_epoch, self.scheduler.get_lr()))
            print('Train {} on Epoch {} Learning Rate: {}'.format(self.config.exp_name, self.current_epoch, self.scheduler.get_lr()))

            rateReachGoal = 0.0
            if self.config.num_agents >= 10:
                if epoch % self.config.validate_every == 0:
                    rateReachGoal = self.test(self.config.mode)
                    self.switch_toOnlineExpert = True
                    self.test('test_trainingSet')
                    # self.test_step()
                    self.save_checkpoint(epoch, lastest=False)
            else:
                if epoch <= 4:
                    rateReachGoal = self.test(self.config.mode)
                    self.switch_toOnlineExpert = True
                    self.test('test_trainingSet')
                    # self.test_step()
                    self.save_checkpoint(epoch, lastest=False)
                elif epoch % self.config.validate_every == 0:
                    rateReachGoal = self.test(self.config.mode)
                    self.switch_toOnlineExpert = True
                    self.test('test_trainingSet')
                    # self.test_step()
                    self.save_checkpoint(epoch, lastest=False)
                    # pass

            is_best = rateReachGoal > self.rateReachGoal
            if is_best:
                self.rateReachGoal = rateReachGoal
            self.save_checkpoint(epoch, is_best=is_best, lastest=True)
            self.scheduler.step()
            # TODO: Optional 2: del dataloader after train
            self.excuation_onlineExport(epoch)

    def excuation_onlineExport(self, epoch):
        if epoch >= self.config.Start_onlineExpert:
            if self.config.num_agents >= 10:
                if epoch % self.config.validate_every == 0:

                    self.pipeline_onlineExpert(epoch)
            else:
                if epoch <= 4:
                    self.pipeline_onlineExpert(epoch)
                elif epoch % self.config.validate_every == 0:
                    self.pipeline_onlineExpert(epoch)

    def pipeline_onlineExpert(self, epoch):
        # TODO: del dataloader
        # create dataloader
        self.onlineExpert.set_up()
        self.onlineExpert.computeSolution()
        self.dataTransformer.set_up(epoch)
        self.dataTransformer.solutionTransformer()
        del self.data_loader
        self.data_loader = DecentralPlannerDataLoader(config=self.config)


    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        # Set the model to be in training mode
        self.model.train()
        # for param in self.model.parameters():
        #     print(param.requires_grad)
        # for batch_idx, (input, target, GSO) in enumerate(self.data_loader.train_loader):
        for batch_idx, (batch_input, batch_target, _, batch_GSO, _) in enumerate(self.data_loader.train_loader):

            inputGPU = batch_input.to(self.config.device)
            gsoGPU = batch_GSO.to(self.config.device)
            # gsoGPU = gsoGPU.unsqueeze(0)
            targetGPU = batch_target.to(self.config.device)
            batch_targetGPU = targetGPU.permute(1,0,2)
            self.optimizer.zero_grad()

            # loss
            loss = 0

            # model

            self.model.addGSO(gsoGPU)
            predict = self.model(inputGPU)


            for id_agent in range(self.config.num_agents):
            # for output, target in zip(predict, target):
                batch_predict_currentAgent = predict[id_agent][:]
                batch_target_currentAgent = batch_targetGPU[id_agent][:][:]
                loss = loss + self.loss(batch_predict_currentAgent,  torch.max(batch_target_currentAgent, 1)[1])
                # print(loss)

            loss = loss/self.config.num_agents

            loss.backward()
            # for param in self.model.parameters():
            #     print(param.grad)
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train {} on Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.config.exp_name,
                    self.current_epoch, batch_idx * len(inputGPU), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1

            # print(loss)
            log_loss = loss.item()
            self.summary_writer.add_scalar("iteration/loss", log_loss, self.current_iteration)

    def test_step(self):
        """
        One epoch of testing the accuracy of decision-making of each step
        :return:
        """

        # Set the model to be in training mode
        self.model.eval()

        log_loss_validStep = []
        for batch_idx, (batch_input, batch_target, _, batch_GSO, _) in enumerate(self.data_loader.validStep_loader):

            inputGPU = batch_input.to(self.config.device)
            gsoGPU = batch_GSO.to(self.config.device)
            # gsoGPU = gsoGPU.unsqueeze(0)
            targetGPU = batch_target.to(self.config.device)
            batch_targetGPU = targetGPU.permute(1, 0, 2)
            self.optimizer.zero_grad()

            # loss
            loss_validStep = 0

            # model
            self.model.addGSO(gsoGPU)
            predict = self.model(inputGPU)

            for id_agent in range(self.config.num_agents):
                # for output, target in zip(predict, target):
                batch_predict_currentAgent = predict[id_agent][:]
                batch_target_currentAgent = batch_targetGPU[id_agent][:][:]
                loss_validStep = loss_validStep + self.loss(batch_predict_currentAgent, torch.max(batch_target_currentAgent, 1)[1])
                # print(loss)

            loss_validStep = loss_validStep/self.config.num_agents

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('ValidStep {} on Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.config.exp_name,
                                                                                                self.current_epoch,
                                                                                                batch_idx * len(inputGPU),
                                                                                                len(self.data_loader.validStep_loader.dataset),
                                                                                                100. * batch_idx / len(self.data_loader.validStep_loader),
                                                                                                loss_validStep.item()))

            log_loss_validStep.append(loss_validStep.item())

            # self.current_iteration_validStep += 1
            # self.summary_writer.add_scalar("iteration/loss_validStep", loss_validStep.item(), self.current_iteration_validStep)
            # print(loss)


        avg_loss = sum(log_loss_validStep)/len(log_loss_validStep)
        self.summary_writer.add_scalar("epoch/loss_validStep", avg_loss, self.current_epoch)

    def test(self, mode):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if mode == 'test':
            dataloader = self.data_loader.test_loader
            label = 'test'
        elif mode == 'test_trainingSet':
            dataloader = self.data_loader.test_trainingSet_loader
            label = 'test_training'
            if self.switch_toOnlineExpert:
                self.robot.createfolder_failure_cases()
        else:
            dataloader = self.data_loader.valid_loader
            label = 'valid'

        size_dataset = dataloader.dataset.data_size
        self.logger.info('\n{} set on {} in {} testing set \n'.format(label, self.config.exp_name, size_dataset))

        self.recorder.reset()
        # maxstep = self.robot.getMaxstep()
        with torch.no_grad():
            for input, target, makespan, _, tensor_map in dataloader:

                inputGPU = input.to(self.config.device)
                targetGPU = target.to(self.config.device)


                log_result = self.mutliAgent_ActionPolicy(inputGPU, targetGPU, makespan, tensor_map, self.recorder.count_validset,mode)
                self.recorder.update(self.robot.getMaxstep(), log_result)

        self.summary_writer = self.recorder.summary(label, self.summary_writer, self.current_epoch)


        self.logger.info('Accurracy(reachGoalnoCollision): {} \n  '                        
                         'DeteriorationRate(MakeSpan): {} \n  '
                         'DeteriorationRate(FlowTime): {} \n  '
                         'Rate(collisionPredictedinLoop): {} \n  '
                         'Rate(FailedReachGoalbyCollisionShielding): {} \n '.format(
                                                                  round(self.recorder.rateReachGoal, 4),
                                                                  round(self.recorder.avg_rate_deltaMP, 4),
                                                                  round(self.recorder.avg_rate_deltaFT, 4),
                                                                  round(self.recorder.rateCollisionPredictedinLoop, 4),
                                                                  round(self.recorder.rateFailedReachGoalSH, 4),
                                                                  ))

        # if self.config.mode == 'train' and self.plot_graph:
        #     self.summary_writer.add_graph(self.model,None)
        #     self.plot_graph = False

        return self.recorder.rateReachGoal

    def mutliAgent_ActionPolicy(self, input, load_target, makespanTarget, tensor_map, ID_dataset,mode):

        self.robot.setup(input, load_target, makespanTarget, tensor_map, ID_dataset)
        maxstep = self.robot.getMaxstep()

        allReachGoal = False
        noReachGoalbyCollsionShielding = False

        check_collisionFreeSol = False

        check_CollisionHappenedinLoop = False

        check_CollisionPredictedinLoop = False

        findOptimalSolution = False

        compare_makespan, compare_flowtime = self.robot.getOptimalityMetrics()
        currentStep = 0

        Case_start = time.process_time()
        Time_cases_ForwardPass = []
        for step in range(maxstep):
            currentStep = step + 1
            currentState = self.robot.getCurrentState()
            currentStateGPU = currentState.to(self.config.device)

            gso = self.robot.getGSO(step)
            gsoGPU = gso.to(self.config.device)
            self.model.addGSO(gsoGPU)
            # self.model.addGSO(gsoGPU.unsqueeze(0))

            step_start = time.process_time()
            actionVec_predict = self.model(currentStateGPU)

            time_ForwardPass = time.process_time() - step_start

            Time_cases_ForwardPass.append(time_ForwardPass)
            allReachGoal, check_moveCollision, check_predictCollision = self.robot.move(actionVec_predict, currentStep)


            if check_moveCollision:
                check_CollisionHappenedinLoop = True


            if check_predictCollision:
                check_CollisionPredictedinLoop = True

            if allReachGoal:
                # findOptimalSolution, compare_makespan, compare_flowtime = self.robot.checkOptimality()
                # print("### Case - {} within maxstep - RealGoal: {} ~~~~~~~~~~~~~~~~~~~~~~".format(ID_dataset, allReachGoal))
                break
            elif currentStep >= (maxstep):
                # print("### Case - {} exceed maxstep - RealGoal: {} - check_moveCollision: {} - check_predictCollision: {}".format(ID_dataset, allReachGoal, check_CollisionHappenedinLoop, check_CollisionPredictedinLoop))
                break

        num_agents_reachgoal = self.robot.count_numAgents_ReachGoal()
        store_GSO, store_communication_radius = self.robot.count_GSO_communcationRadius(currentStep)

        if allReachGoal and not check_CollisionHappenedinLoop:
            check_collisionFreeSol = True
            noReachGoalbyCollsionShielding = False
            findOptimalSolution, compare_makespan, compare_flowtime = self.robot.checkOptimality(True)
            if self.config.log_anime and self.config.mode == 'test':
                self.robot.save_success_cases('success')

        if currentStep >= (maxstep):
            findOptimalSolution, compare_makespan, compare_flowtime = self.robot.checkOptimality(False)
            if mode == 'test_trainingSet' and self.switch_toOnlineExpert:
                self.robot.save_failure_cases()

        if currentStep >= (maxstep) and not allReachGoal and check_CollisionPredictedinLoop and not check_CollisionHappenedinLoop:
            findOptimalSolution, compare_makespan, compare_flowtime = self.robot.checkOptimality(False)
            # print("### Case - {} -Step{} exceed maxstep({})- ReachGoal: {} due to CollsionShielding \n".format(ID_dataset,currentStep,maxstep, allReachGoal))
            noReachGoalbyCollsionShielding = True
            if self.config.log_anime and self.config.mode == 'test':
                self.robot.save_success_cases('failure')
        time_record = time.process_time() - Case_start

        if self.config.mode == 'test':
            exp_status = "################## {} - End of loop ################## ".format(self.config.exp_name)
            case_status = "####### Case{} \t Computation time:{} \t Step{}/{}\t- AllReachGoal-{}\n".format(ID_dataset, time_record,
                                                                                             currentStep,
                                                                                             maxstep, allReachGoal)

            self.logger.info('{} \n {}'.format(exp_status, case_status))


        # if self.config.mode == 'test':
        #     self.robot.draw(ID_dataset)


        # return [allReachGoal, noReachGoalbyCollsionShielding, findOptimalSolution, check_collisionFreeSol, check_CollisionPredictedinLoop, makespanPredict, makespanTarget, flowtimePredict,flowtimeTarget,num_agents_reachgoal]

        return allReachGoal, noReachGoalbyCollsionShielding, findOptimalSolution, check_collisionFreeSol, check_CollisionPredictedinLoop, compare_makespan, compare_flowtime, num_agents_reachgoal, store_GSO, store_communication_radius, time_record,Time_cases_ForwardPass


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        if self.config.mode == 'train':
            print(self.model)
        print("Experiment on {} finished.".format(self.config.exp_name))
        print("Please wait while finalizing the operation.. Thank you")
        # self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
        if self.config.mode == 'test':
            print("################## End of testing ################## ")
            print("Computation time:\t{} ".format(self.time_record))

