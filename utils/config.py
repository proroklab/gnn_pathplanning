import os
import shutil
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
import datetime
import time
from easydict import EasyDict
from pprint import pprint
import time
from utils.dirs import create_dirs


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(args):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(args.config)
    print(" THE Configuration of your experiment ..")

    config.mode = args.mode
    config.num_agents = args.num_agents
    config.map_w = args.map_w
    config.map_h = args.map_w
    config.map_density = args.map_density
    config.map_type = args.map_type

    config.trained_num_agents = args.trained_num_agents
    config.trained_map_w = args.trained_map_w
    config.trained_map_h = args.trained_map_w
    config.trained_map_density = args.trained_map_density
    config.trained_map_type = args.trained_map_type

    config.nGraphFilterTaps = args.nGraphFilterTaps
    config.hiddenFeatures = args.hiddenFeatures

    config.num_testset = args.num_testset

    config.con_train = args.con_train
    config.lastest_epoch = args.lastest_epoch
    config.best_epoch = args.best_epoch
    config.test_general = args.test_general
    config.train_TL = args.train_TL
    config.test_epoch = args.test_epoch
    config.Use_infoMode = args.Use_infoMode
    config.log_anime = args.log_anime
    config.rate_maxstep = args.rate_maxstep
    config.commR = args.commR
    pprint(config)

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_net))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    if config.mode == 'train':
        if config.con_train:
            config.exp_time_load = args.log_time_trained
            config.exp_time = config.exp_time_load
        elif config.train_TL:
            # pretrain in one setup and fine-tune in another
            config.exp_time_load = args.log_time_trained
            log_time = datetime.datetime.now()
            config.exp_time = str(int(time.mktime(log_time.timetuple())))  # + "/"
            # config.exp_time = log_time.strftime("%H%M-%d%b%Y")
            # config.exp_time = str(int(time.time()))
        else:
            log_time = datetime.datetime.now()
            config.exp_time = str(int(time.mktime(log_time.timetuple()))) #+ "/"
            # config.exp_time = log_time.strftime("%H%M-%d%b%Y")
            # config.exp_time = str(int(time.time()))

    elif config.mode == "test":
        # exp_time = config.log_time
        config.exp_time_load = args.log_time_trained + "/"
        config.exp_time = args.log_time_trained

    env_Setup = "{}{}x{}_rho{}_{}Agent".format(config.map_type, config.map_w, config.map_w, config.map_density, config.trained_num_agents)

    config.exp_hyperPara = "K{}_HS{}".format(config.nGraphFilterTaps, config.hiddenFeatures)

    if config.con_train or config.train_TL:

        config.exp_name_load = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.exp_time_load)
        config.checkpoint_dir_load = os.path.join(config.save_data, "experiments", config.exp_name_load, "checkpoints/")

        config.exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.exp_time)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints")
        
    elif config.test_general:

        env_Setup_load = "{}{}x{}_rho{}_{}Agent".format(config.trained_map_type, config.trained_map_w, config.trained_map_w,
                                                        config.trained_map_density, config.trained_num_agents)
        env_Setup_test = "{}{}x{}_rho{}_{}Agent".format(config.map_type, config.map_w, config.map_w,
                                                        config.map_density, config.num_agents)

        config.exp_name_load = os.path.join("{}_{}".format(config.exp_net, env_Setup_load), config.exp_hyperPara, config.exp_time_load)
        config.checkpoint_dir_load = os.path.join(config.save_data, "experiments", config.exp_name_load, "checkpoints/")

        config.exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup_test), config.exp_hyperPara, config.exp_time)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints")

    else:
        config.exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.exp_time)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints/")


    config.summary_dir = os.path.join(config.save_tb_data,config.exp_name)
    config.out_dir = os.path.join(config.save_data, "experiments", config.exp_name, "out/")
    config.log_dir = os.path.join(config.save_data, "experiments", config.exp_name, "logs/")

    config.failCases_dir = os.path.join(config.save_data, "experiments", config.exp_name, "failure_cases/")

    if config.best_epoch:
        label_result_folder = "Results_best"
    else:
        label_result_folder = "Results"

    if config.test_general:
        label_Statistics_folder = "Statistics_generalization"
    else:
        label_Statistics_folder = "Statistics"

    exp_Setup_training = "TR_M{}p{}_{}Agent".format(config.trained_map_w, config.trained_map_density,
                                                     config.trained_num_agents)
    exp_HyperPara = "K{}_HS{}".format(config.nGraphFilterTaps,config.hiddenFeatures)

    testEnv_Setup = "{}{}x{}_rho{}_{}Agent".format(config.map_type, config.map_w, config.map_w, config.map_density,
                                                config.num_agents)


    config.result_statistics_dir = os.path.join(config.save_data, label_result_folder, label_Statistics_folder)
    config.result_demo_dir = os.path.join(config.save_data, label_result_folder, 'Demo/{}/'.format(config.exp_name))
    log_str_AnimeDemo = os.path.join('AnimeDemo',config.exp_net, testEnv_Setup, exp_HyperPara, exp_Setup_training,  config.exp_time, 'commR_{}'.format(config.commR))
    config.result_AnimeDemo_dir_input = os.path.join(config.save_data, label_result_folder,log_str_AnimeDemo,'input/')
    config.result_AnimeDemo_dir_predict_success = os.path.join(config.save_data, label_result_folder, log_str_AnimeDemo,'predict_success')
    config.result_AnimeDemo_dir_predict_failure = os.path.join(config.save_data, label_result_folder, log_str_AnimeDemo,
                                                       'predict_failure')
    config.result_AnimeDemo_dir_target = os.path.join(config.save_data, label_result_folder, log_str_AnimeDemo,
                                                      'target')
    config.result_AnimeDemo_dir_GSO = os.path.join(config.save_data, label_result_folder, log_str_AnimeDemo,
                                                      'GSO')
    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir, config.failCases_dir,
                 config.result_statistics_dir, config.result_demo_dir,
                 config.result_AnimeDemo_dir_input,config.result_AnimeDemo_dir_predict_success,
                 config.result_AnimeDemo_dir_predict_failure, config.result_AnimeDemo_dir_target, config.result_AnimeDemo_dir_GSO])

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config
