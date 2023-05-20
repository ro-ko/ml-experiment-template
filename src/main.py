#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import fire
import torch
from pathlib import Path

from utils import set_random_seed
from data import Mnist
from models.mymodel.train import MyTrainer
from models.mymodel.eval import MyEvaluator
from utils import log_param
from loguru import logger

from experiments import exp_hyper_param


def run_mymodel(device, train_data, test_data, hyper_param):
    trainer = MyTrainer(device=device,
                        in_dim=train_data.in_dim,
                        out_dim=train_data.out_dim)

    model = trainer.train_with_hyper_param(train_data=train_data,
                                           hyper_param=hyper_param)

    evaluator = MyEvaluator(device=device)
    accuracy = evaluator.evaluate(model, test_data)

    return accuracy


def main(cfg_name="default",):
    """
    Handle user arguments of ml-project-template
    :param cfg_name: config file name
    """

    # Step 0. Initialization
    logger.add(os.path.join("./experiments/log", cfg_name + ".log"))
    logger.info("The main procedure has started with the following parameters:")
    args = exp_hyper_param.parse(cfg_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=args.seed, device=device)
    
    # set gpu
    if device=='cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # config
    param = dict()
    param['model'] = args.model
    param['seed'] = args.seed
    param['device'] = device
    log_param(param)

    # Step 1. Load datasets
    data_path = Path(__file__).parent.parent.absolute().joinpath("datasets")
    train_data = Mnist(data_path=data_path, train=True)
    test_data = Mnist(data_path=data_path, train=False)
    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info("- # of training instances: {}".format(len(train_data)))
    logger.info("- # of test instances: {}".format(len(test_data)))

    # Step 2. Run (train and evaluate) the specified model

    logger.info("Training the model has begun with the following hyperparameters:")
    hyper_param = dict()
    hyper_param['batch_size'] = args.batch_size
    hyper_param['epochs'] = args.epochs
    hyper_param['learning_rate'] = args.learning_rate
    hyper_param['checkpoint'] = args.checkpoint
    hyper_param['path'] = args.path
    hyper_param['cfg_name'] = cfg_name
    log_param(hyper_param)

    if args.model == 'mymodel':
        accuracy = run_mymodel(device=device,
                               train_data=train_data,
                               test_data=test_data,
                               hyper_param=hyper_param)

        # - If you want to add other model, then add an 'elif' statement with a new runnable function
        #   such as 'run_my_model' to the below
        # - If models' hyperparamters are varied, need to implement a function loading a configuration file
    else:
        logger.error("The given \"{}\" is not supported...".format(args.model))
        return

    # Step 3. Report and save the final results
    logger.info("The model has been trained. The test accuracy is {:.4}.".format(accuracy))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
