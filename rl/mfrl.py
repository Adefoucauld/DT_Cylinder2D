import time
import csv
import os

import numpy as np
import torch as th
# import wandb

from .orl import ORL
from data.data_handler import Data
from decision_transformer.agents.dt import DecisionTransformer



class MFRL(ORL):
    """
    Model-Free Reinforcement Learning (MFRL) module
        1. Set and build basic components of the MFRL experiment
        2. Handle the agent learning loop
    """
    def __init__(self, config):
        super(MFRL, self).__init__(config)
        # print('Initialize MFRL!')
        self.config = config
        self.device = config['experiment']['device']
        # self.seed = seed
        self._build()


    def _build(self):
        super(MFRL, self)._build()
        self._set_data_handler()
        self._set_agent()


    def _set_data_handler(self):
        self.data = Data(self.state_dim, self.act_dim, self.config, self.seed)


    def _set_agent(self):
        self.agent = DecisionTransformer(self.state_dim,
                                         self.act_dim,
                                         self.config).to(self.device)


    def learn(self):
        N = self.config['learning']['nIter'] # Number of learning iterations
        NT = self.config['learning']['iter_steps'] # Number of warmup learning iterations
        Ni = self.config['learning']['niIter'] # Number of training steps/iteration
        EE = self.config['evaluation']['eval_episodes'] # Number of episodes

        logs = dict()
        # gif = True
        best_ret = 0.0

        # print('Start Learning!')
        start_time = time.time()
        for n in range(N):
           
            # learn
            train_start = time.time()
            trainLosses = self.train_agent(NT)
            logs['time/training'] = time.time() - train_start

            # evaluate
            eval_start = time.time()
            eval_logs = self.evaluate_agent(EE, n)

            for k, v in eval_logs.items():
                logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(trainLosses)
            logs['training/train_loss_std'] = np.std(trainLosses)
            
            name = "loss.csv"
            if (not os.path.exists("saved_models")):
                os.mkdir("saved_models")
            if (not os.path.exists("saved_models/" + name)):
                with open("saved_models/" + name, "w") as csv_file:
                    spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Time", "Loss_mean", "Loss_std"])
                    spam_writer.writerow([time.time() - start_time, np.mean(trainLosses), np.std(trainLosses)])
            else:
                with open("saved_models/" + name, "a") as csv_file:
                    spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([time.time() - start_time, np.mean(trainLosses), np.std(trainLosses)])

            # # log
            # if print_logs:
            #     for k, v in logs.items():
            #         print(f'{k}: {v}')

            # # WandB
            # if self.config['experiment']['WandB']:
            #     wandb.log(logs)

        return self.agent
