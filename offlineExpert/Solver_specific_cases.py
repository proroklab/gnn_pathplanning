
import os
import sys
import yaml
import subprocess
import signal

import shutil
import time
from os.path import dirname, realpath, pardir

from easydict import EasyDict

# os.system("taskset -p -c 0 %d" % (os.getpid()))
# os.system("taskset -p 0xFFFFFFFF %d" % (os.getpid()))
# os.system("taskset -p -c 8-15,24-31 %d" % (os.getpid()))

from multiprocessing import Queue, Pool, Lock, Manager, Process
 #TODO: Ajudst process number w.r.t your workstation performance.
# Note: normally, 2~3 for a real CPU processor with 2 threads

def handler(signum, frame):
    raise Exception("Solution computed by ECBS is timeout.")

class ComputeECBSSolution:
    def __init__(self, config):
        self.config = config
        self.timeout = 60
        self.list_failureCases_input = None
        self.PROCESS_NUMBER = 4


    def set_up(self):
        self.dir_input = os.path.join(self.config.failCases_dir, "input/")
        self.dir_sol = os.path.join(self.config.failCases_dir, "output_ECBS/")

        self.list_failureCases_input = self.search_failureCases(self.dir_input)
        self.list_failureCases_input = sorted(self.list_failureCases_input)
        self.len_failureCases_input = len(self.list_failureCases_input)
        self.task_queue = Queue()

        if os.path.exists(self.dir_sol) and os.path.isdir(self.dir_sol):
            shutil.rmtree(self.dir_sol)
        try:
            # Create target Directory
            os.makedirs(self.dir_sol)
        except FileExistsError:
            pass

    def computeSolution(self):

        #todo: run multiprocess for runExpertSolver

        for id_case in range(self.len_failureCases_input):
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
            name_inputfile = self.list_failureCases_input[id_case]
            id_input_case = name_inputfile.split('_ID')[-1]
            name_outputfile = self.dir_sol + 'failureCases_solved_ID{}'.format(id_input_case)
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
            with open(name_outputfile) as output_file:
                return yaml.safe_load(output_file)
        except Exception as e:
            print(e)

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




if __name__ == '__main__':



    config = {'num_agents': 10,
              'map_w': 20,
              'map_h': 20,
              'failCases_dir': '/local/scratch/ql295/Data/MultiAgentDataset/Solution_BMap/demo20x20_density_p1/10_Agent',
              'exp_net': 'dcp'
              }

    config_setup = EasyDict(config)

    Expert = ComputeECBSSolution(config_setup)
    Expert.set_up()
    Expert.computeSolution()



