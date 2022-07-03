# import gym
import numpy as np
import torch as th

from simulation_base.env import resume_env, nb_actuations


class ORL:
    """
    Offline Reinforcement Learning module
        Handle higher training and evalution of the trained agent
    """
    def __init__(self, config):
        # print('Initialize ORL!')
        self.config = config
        # self.seed = seed


    def _build(self):
        self._set_env()


    def _set_env(self):
        env_name = self.config['experiment']['env_name']
        # if env_name == 'hopper':
        #     name = 'Hopper-v3'
        # elif env_name == 'halfcheetah':
        #     name = 'HalfCheetah-v3'
        # elif env_name == 'walker2d':
        #     name = 'Walker2d-v3'
        
        '''---> self.eval_env = resume_env(....)'''
        self.eval_env = resume_env(plot = False, dump_CL = 100, dump_debug = 1, dump_vtu = 50)
        
        # # self.eval_env = gym.make(name) #MakeEnv(self.environment)
        
        ''' remove seed because no such thing in Env2DCylinder'''
        # self.eval_env.seed(self.seed)
        # self.eval_env.action_space.seed(self.seed)
        # self.eval_env.observation_space.seed(self.seed)

        ''' adapt following to Env2DCylinder '''
        # self.state_dim = self.eval_env.observation_space.shape[0]
        self.state_dim = self.states['obs']['shape']
        # self.act_dim = self.eval_env.action_space.shape[0]
        self.act_dim = self.actions['shape']
        self.rew_dim = 1
        # self.act_upper_lim = self.eval_env.action_space.high
        self.act_upper_lim = self.actions['max_value']
        # self.act_lower_lim = self.eval_env.action_space.low
        self.act_lower_lim = self.actions['min_value']


    def train_agent(self, NT):
        self.agent.train()

        Losses = []
        for nt in range(NT):
            # if print_logs:
                # print(f' [ Agent Training ] Step: {nt}   ', end='\r')
            loss = self.agent.train_model(self.data)
            Losses.append(loss)
            if self.agent.scheduler: self.agent.scheduler.step()

        return Losses


    # def evaluate_agent(self, EE, gif, n, print_logs=True):
    def evaluate_agent(self, EE, n):
        env_targets = self.config['experiment']['env_targets']
        device = self.config['experiment']['device']
        # mode = self.config['experiment']['mode']
        scale = self.config['experiment']['scale']
        E = self.config['experiment']['max_env_len']

        self.agent.eval()

        eval_logs = dict()
        for target_rew in env_targets:
            returns, lengths = [], []

            for ee in range(EE):
                # if print_logs:
                #     print(f' [ Agent Evaluation ] Target: {target_rew}, Episode: {ee}   ', end='\r')
                # if ee > 0:
                #     gif = False
                with th.no_grad():
                    # ret, length = self.agent.evaluate_model(
                    #                             self.eval_env, gif, n,
                    #                             device, mode, scale, E,
                    #                             self.data.state_mean, self.data.state_std,
                    #                             target_return=target_rew/scale)
                    ret, length = self.agent.evaluate_model(
                                                self.eval_env, n, 
                                                scale, E,
                                                self.data.state_mean, self.data.state_std,
                                                target_return=target_rew/scale)
                
                returns.append(ret)
                lengths.append(length)

            eval_logs.update({
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                # f'target_{target_rew}_length_std': np.std(lengths)
                })

        return eval_logs
