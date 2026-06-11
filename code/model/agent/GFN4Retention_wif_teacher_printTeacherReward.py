import time
import copy
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
from typing import Dict, List, Tuple, Union, cast

import utils
from model.agent.BaseRLAgent import BaseRLAgent

# version 1: (discrepancies**2) * (1 + C)
# def get_teacher_reward(
#     discrepancies: list,
#     reward: torch.Tensor,
#     log: bool = True,
#     C: float = 19.0,
#     alpha: float = 2.0,
# ) -> torch.Tensor:
#     # Clone and detach to avoid gradient issues
#     _d = discrepancies[0].clone().detach()
#     _r = reward.clone().detach()
#     _d_positive = _d ** 2
#     _a_r = torch.where(
#         _d > 0,
#         _d_positive * torch.tensor(C, device=_d.device, dtype=_d.dtype),
#         _d_positive,
#     )
#     if log:
#         teacher_reward = (1.0 + 1e-3 + _a_r).log() * ((1.0 + _r) ** alpha)
#     else:
#         teacher_reward = _a_r * ((1.0 + _r) ** alpha)
#     return teacher_reward

# version 2: (10 * discrepancies)**2 * (1 + C)
# 由于discrepancies的值很小，大部分处在0到1之间，所以平方以后可能越乘越小，
# 所以这里乘以10来放大到大于1的数，这样的话平方以后都是变大的，再乘以一个(1 + C)这样的数值变化就比较合理
def get_teacher_reward(
    discrepancies: list,
    reward: torch.Tensor,
    log: bool = False,
    C: float = 19.0,
    alpha: float = 2.0,
) -> torch.Tensor:
    # Clone and detach to avoid gradient issues
    _d = discrepancies[0].clone().detach()
    _r = reward.clone().detach()
    _d_positive = (10 * _d) ** 2
    _a_r = torch.where(
        _d > 0,
        _d_positive * (1.0 + torch.tensor(C, device=_d.device, dtype=_d.dtype)),
        _d_positive,
    )
    if log:
        teacher_reward = (1.0 + 1e-3 + _a_r).log() * (_r ** alpha)
    else:
        teacher_reward = _a_r * (_r ** alpha)
    return _a_r, (_r ** alpha), teacher_reward

class GFN4Retention_wif_teacher_printTeacherReward(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
            '''
        parser = BaseRLAgent.parse_model_args(parser)

        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        
        parser.add_argument('--gfn_flow_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of flow estimator')
        
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, 
                            help='smooth offset of forward logp of TB loss') #b_f
        
        parser.add_argument('--gfn_backward_offset', type=float, default=1.0, 
                            help='smooth offset of backward logp of TB loss') #b_b
        
        parser.add_argument('--gfn_reward_smooth', type=float, default=0.5, 
                            help='reward smooth offset in the backward part of TB loss') #b_r 0.5
        
        parser.add_argument('--gfn_Z', type=float, default=0.1, 
                            help='average reward offset') #b_z 0.1
            
        parser.add_argument('--lambda_gfn', type=float, default=0.5, 
                            help='-')
            
        parser.add_argument('--tn_balance', type=float, default=0.5, 
                            help='-')

        parser.add_argument('--zzz', type=float, default=0.5, 
                            help='balance two rewards')

        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                                help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                                help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                                help='mitigation factor')
        parser.add_argument('--noise_std', type=float, default=0.1, 
                                help='noise standard deviation for action exploration')
        parser.add_argument('--noise_clip', type=float, default=1.0, 
                                help='noise clip bound for action exploration')
        parser.add_argument('--teacher_reward_alpha', type=float, default=2.0,
                                help='alpha for teacher reward')
        parser.add_argument('--teacher_reward_C', type=float, default=19.0,
                                help='C for teacher reward')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        Initialize the GFN model.
        Setup the flow model, optimizer, and other necessary components.
        '''
        args, env, actor, critics, buffer = input_args
        
        super().__init__(args, env, actor, buffer)
        
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flow_hidden_dims = args.gfn_flow_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset
        self.gfn_backward_offset = args.gfn_backward_offset
        self.gfn_reward_smooth = args.gfn_reward_smooth
        self.gfn_Z = args.gfn_Z
        self.lambda_gfn = args.lambda_gfn
        self.tn_balance = args.tn_balance
        self.zzz = args.zzz
        self.alpha = args.teacher_reward_alpha
        self.C = args.teacher_reward_C
        
        self.noise_std = args.noise_std
        self.noise_clip = args.noise_clip
        
        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        # Initialize the scheduler

        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=400, gamma=0.01)
        
        self.flow_estimator = critics[0]
        self.critic1 = critics[0]
        self.critic1_optimizer = torch.optim.Adam(self.flow_estimator.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.critic1_optimizer, step_size=400, gamma=0.01)

        self.backward_estimator = critics[1]
        self.critic2 = critics[1]
        self.critic2_optimizer = torch.optim.Adam(self.backward_estimator.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.scheduler3 = torch.optim.lr_scheduler.StepLR(self.critic2_optimizer, step_size=400, gamma=0.01)
        # teacher model
        self.teacher_actor = copy.deepcopy(self.actor)
        self.teacher_flow_estimator = copy.deepcopy(self.flow_estimator)
        self.teacher_backward_estimator = copy.deepcopy(self.backward_estimator)
        self.actor_optimizer_teacher = torch.optim.Adam(self.teacher_actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay)
        self.critic1_optimizer_teacher = torch.optim.Adam(self.teacher_flow_estimator.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)
        self.critic2_optimizer_teacher = torch.optim.Adam(self.teacher_backward_estimator.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)
        # register models that will be saved
        self.registered_models.append((self.critic1, self.critic1_optimizer, "_critic1"))
        self.registered_models.append((self.critic2, self.critic2_optimizer, "_critic2"))
        self.registered_models.append((self.teacher_flow_estimator, self.critic1_optimizer_teacher, "_critic1_teacher"))
        self.registered_models.append((self.teacher_backward_estimator, self.critic2_optimizer_teacher, "_critic2_teacher"))
        
        
    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({'Q1_loss': [], 'Q2_loss': [],
                                      'Q1': [], 'Q2': [], 'next_Q1': [], 'next_Q2': [], 'target_Q': [],
                                      'DB_forward': [], 'DB_backward': [],
                                      'DB_forward_teacher': [], 'DB_backward_teacher': [],
                                      'teacher_loss': [], 'teacher_discrepancies_reward': [], 'teacher_student_weight_reward': [],
                                      # Student vs Teacher trajectory evidence (for reviewer rebuttal)
                                      'student_retention': [], 'teacher_retention': [],
                                      'student_total_reward': [], 'teacher_total_reward': [],
                                      'student_DB_diff': [], 'teacher_DB_diff': [],
                                      'student_immediate_reward': [], 'teacher_immediate_reward': [],
                                      'student_is_teacher': [], 'teacher_is_teacher': [],
                                      })
        self.eval_history.update({'avg_retention': [], 'max_retention': [], 'min_retention': [],
                                   'avg_retention_teacher': [], 'max_retention_teacher': [], 'min_retention_teacher': []})
        

    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
        is_teacher = (episode_iter % 2 != 0)  # Track which model is being used
        if episode_iter % 2 == 0:
            actor = self.actor
            flow_estimator = self.flow_estimator
            backward_estimator = self.backward_estimator
            actor_optimizer = self.actor_optimizer
            critic1_optimizer = self.critic1_optimizer
            critic2_optimizer = self.critic2_optimizer
        else:
            actor = self.teacher_actor
            flow_estimator = self.teacher_flow_estimator
            backward_estimator = self.teacher_backward_estimator
            actor_optimizer = self.actor_optimizer_teacher
            critic1_optimizer = self.critic1_optimizer_teacher
            critic2_optimizer = self.critic2_optimizer_teacher
        with torch.no_grad():
            # sample action
            policy_output = self.apply_policy(observation, actor, 
                                              epsilon, do_explore, is_train)
            # apply action on environment and update replay buffer
            action_dict = {'action': policy_output['action']}
            new_observation, user_feedback, updated_info = self.env.step(action_dict)
            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']
                                     
            # monitor update
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_retention'].append(user_feedback['retention'].mean().item()) 
                self.eval_history['max_retention'].append(user_feedback['retention'].max().item()) 
                self.eval_history['min_retention'].append(user_feedback['retention'].min().item())
                self.eval_history['avg_total_reward'].append(self.current_sum_reward.mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward.max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward.min().item())
                if is_teacher:
                    self.eval_history['avg_retention_teacher'].append(user_feedback['retention'].mean().item())
                    self.eval_history['max_retention_teacher'].append(user_feedback['retention'].max().item())
                    self.eval_history['min_retention_teacher'].append(user_feedback['retention'].min().item())
                self.current_sum_reward *= 0
            
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())

            # ── Student vs Teacher trajectory evidence (for reviewer rebuttal) ──
            # Only record when the respective model is active (avoids polluting with zeros)
            immediate_reward = self._compute_immediate_reward(user_feedback)
            if not is_teacher:
                self.training_history['student_retention'].append(user_feedback['retention'].mean().item())
                self.training_history['student_total_reward'].append(R.mean().item())
                self.training_history['student_immediate_reward'].append(immediate_reward.mean().item())
            else:
                self.training_history['teacher_retention'].append(user_feedback['retention'].mean().item())
                self.training_history['teacher_total_reward'].append(R.mean().item())
                self.training_history['teacher_immediate_reward'].append(immediate_reward.mean().item())

            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, updated_info['updated_observation'])
            observation = new_observation
        return new_observation
    
    
    def _compute_immediate_reward(self, user_feedback):
        """Compute per-step immediate reward from user feedback (shared by run_episode_step and step_train)."""
        user_feedback['immediate_response_weight'] = self.env.response_weights
        point_reward = user_feedback['immediate_response'].reshape(-1, 6, 7) * user_feedback['immediate_response_weight'].view(1, 1, -1)
        combined_reward = torch.sum(point_reward, dim=2)
        return torch.mean(combined_reward, dim=1)

    def get_flow(self, state_dict, is_teacher=False):
        '''
        The flow estimator F(s_t|u)
        :param params:
        :param state_dict: {'emb': (B, state_dim), 'wtm': (B,)}
        :return: {'flow': (B, 1), 'log_f': (B, 1)}
        '''
        if not is_teacher:
            flow_estimator = self.flow_estimator
        else:
            flow_estimator = self.teacher_flow_estimator
        log_flow = flow_estimator(state_dict)['v']
        flow = torch.exp(log_flow)
        
        return {'flow': flow, 'log_f': log_flow}
    
    
    
    def get_forward_prob(self, observation, is_teacher=False):
        
        # policy output
        
        
        '''
        Forward function P(s_{t+1} | s_t, u) in GFN
        :param params: 输入参数
        :param current_state_dict: {'emb': (B, state_dim), 'wtm': (B,)}
        :param current_action: a_{t} 采样中的当前动作
        :return: {'prob_f': (B, 1), forward probability P(s_{t+1}|s_{t}) = P(a_{t}|s_{t}),
                  'mu', 'sigma': (B, action_dim), 输出动作的 distribution 参数}
        '''
        # current action distribution
        is_train = True
        if not is_teacher:
            actor = self.actor
        else:
            actor = self.teacher_actor
        current_policy_output = self.apply_policy(observation, actor, self.epsilon, True, is_train)
        mu = current_policy_output['mu']
        sigma = current_policy_output['sigma']
        current_action = current_policy_output['action']
        action_dim = mu.shape[1]
        mu_bias = 0.01
        #mu, sigma, mu_bias = self.take_action(current_state_dict['emb'], current_state_dict['wtm'], params)
        mu_avg = (mu + mu_bias) / 2

        dist = Normal(mu_avg, sigma)
        P = dist.log_prob(current_action).exp() + 0.001  # Add a small constant for numerical stability
        P = P.view(-1, action_dim)
        gfn_p_forward = P.sum(dim=1, keepdim=True)  # Sum over action dimensions

        return {'prob_f': gfn_p_forward, 'mu': mu_avg, 'sigma': sigma}


    def get_backward_prob(self, current_dict, next_dict, is_teacher=False):

        if not is_teacher:
            backward_estimator = self.backward_estimator
        else:
            backward_estimator = self.teacher_backward_estimator
        output_prob = backward_estimator(current_dict, next_dict)['prob_b']

        return {'prob_b': output_prob}
    
    
    def step_train(self):
        '''
        @process:
        '''
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        #reward = user_feedback['reward']
        #reward = reward.to(torch.float)
        done_mask = done_mask.to(torch.float)
        
        #critic_loss_list, actor_loss = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        # loss = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        loss, discrepancies, student_reward, student_db_diff = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation, is_teacher=False)
        teacher_discrepancies_reward, teacher_student_weight_reward, teacher_reward = get_teacher_reward(discrepancies, student_reward, C=self.C, alpha=self.alpha)
        teacher_loss, _, _, teacher_db_diff = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation, is_teacher=True, new_reward=teacher_reward)

        # Immediate reward for student
        student_immediate = self._compute_immediate_reward(user_feedback)
        # Teacher immediate reward (reuse same user_feedback, same immediate reward; DB_diff is what differs)
        teacher_immediate = self._compute_immediate_reward(user_feedback)

        self.training_history['actor_loss'].append(loss.item())
        self.training_history['teacher_loss'].append(teacher_loss.item())
        self.training_history['teacher_discrepancies_reward'].append(teacher_discrepancies_reward.mean().item())
        self.training_history['teacher_student_weight_reward'].append(teacher_student_weight_reward.mean().item())
        # ── Student vs Teacher DB diff & immediate reward (for reviewer rebuttal) ──
        self.training_history['student_DB_diff'].append(torch.mean(student_db_diff).item())
        self.training_history['teacher_DB_diff'].append(torch.mean(teacher_db_diff).item())
        self.training_history['student_immediate_reward'].append(student_immediate.mean().item())
        self.training_history['teacher_immediate_reward'].append(teacher_immediate.mean().item())
        self.training_history['Q1_loss'].append(0)
        self.training_history['Q2_loss'].append(0)
        self.training_history['Q1'].append(0)
        self.training_history['Q2'].append(0)
        self.training_history['next_Q1'].append(0)
        self.training_history['next_Q2'].append(0)
        self.training_history['target_Q'].append(0)
        
        #target_Q, next_Q1, next_Q2, Q1_loss, Q1, Q2_loss, Q2 = critic_loss_list
        #self.training_history['actor_loss'].append(actor_loss.item())
        #self.training_history['Q1_loss'].append(Q1_loss)
        #self.training_history['Q2_loss'].append(Q2_loss)
        #self.training_history['Q1'].append(Q1)
        #self.training_history['Q2'].append(Q2)
        #self.training_history['next_Q1'].append(next_Q1)
        #self.training_history['next_Q2'].append(next_Q2)
        #self.training_history['target_Q'].append(target_Q)

    
    
    def get_gfn_loss(self, observation, policy_output, user_feedback, done_mask, next_observation, is_teacher=False, new_reward=None):
        is_train = True
        if not is_teacher:
            actor = self.actor
        else:
            assert new_reward is not None
            actor = self.teacher_actor

        discrepancies = []

        # Apply policy for the current and next observations
        current_policy_output = self.apply_policy(observation, actor, self.epsilon, True, is_train)
        next_policy_output = self.apply_policy(next_observation, actor, self.epsilon, True, is_train)

        # Extract necessary components for loss calculation
        b_f, b_b, b_r, b_z = self.gfn_forward_offset, self.gfn_backward_offset, self.gfn_reward_smooth, self.gfn_Z
        zzz = self.zzz
        lambda_gfn, tn_balance = self.lambda_gfn, self.tn_balance

        # Calculate flow, forward prob, backward prob for current and next states
        current_flow_output = self.get_flow(current_policy_output, is_teacher)
        log_F_t = current_flow_output['log_f']

        forward_output = self.get_forward_prob(observation, is_teacher)
        P_F = forward_output['prob_f']
        log_P_F = torch.log(P_F + b_f)

        next_flow_output = self.get_flow(next_policy_output, is_teacher)
        log_F_next = next_flow_output['log_f']

        backward_output = self.get_backward_prob(current_policy_output, next_policy_output, is_teacher)
        P_B = backward_output['prob_b']
        log_P_B = torch.log(P_B + b_b)

        # Calculate reward components and loss components
        if not is_teacher:
            trajectory_reward = user_feedback['reward'].view(-1, 1) + 0.001
            log_reward = torch.log(trajectory_reward + b_r)
        else:
            log_reward = torch.log(new_reward)

        DB_forward_terminal = log_F_t
        DB_forward_nonterminal = log_F_t + log_P_F
        DB_forward = DB_forward_terminal * done_mask * tn_balance + \
                     DB_forward_nonterminal * (1 - done_mask) * (1 - tn_balance)

        DB_backward_terminal = log_reward
        
        # With immediate feedback (use shared helper)
        immediate_reward = self._compute_immediate_reward(user_feedback)

        # DB_backward_nonterminal = log_F_next + log_P_B + torch.pow(immediate_reward, zzz)
        DB_backward_nonterminal = log_F_next + log_P_B + torch.mul(immediate_reward, zzz)
        
        DB_backward = DB_backward_terminal * done_mask * tn_balance + \
                      DB_backward_nonterminal * (1 - done_mask) * (1 - tn_balance)

        # Calculate the final GFN DB loss
        gfn_db_diff = DB_forward - DB_backward + b_z
        gfn_db_loss = torch.mean(torch.square(gfn_db_diff))
        if not is_teacher:
            discrepancies.append(gfn_db_diff.detach())

        # Calculate the total loss
        total_loss = lambda_gfn * gfn_db_loss
        
        if not is_teacher:
            self.actor_optimizer.zero_grad()
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()

            total_loss.backward()

            self.actor_optimizer.step()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
        else:
            self.actor_optimizer_teacher.zero_grad()
            self.critic1_optimizer_teacher.zero_grad()
            self.critic2_optimizer_teacher.zero_grad()
            total_loss.backward()
            self.actor_optimizer_teacher.step()
            self.critic1_optimizer_teacher.step()
            self.critic2_optimizer_teacher.step()
        #全部一起 train
        if not is_teacher:
            self.training_history['DB_forward'].append(torch.mean(DB_forward).item())
            self.training_history['DB_backward'].append(torch.mean(DB_backward).item())
        else:
            self.training_history['DB_forward_teacher'].append(torch.mean(DB_forward).item())
            self.training_history['DB_backward_teacher'].append(torch.mean(DB_backward).item())
        #self.scheduler1.step()
        #self.scheduler2.step()
        #self.scheduler3.step()
        loss = total_loss
        student_log_reward = torch.log(user_feedback['reward'].view(-1, 1) + 0.001 + b_r)
        return loss, discrepancies, student_log_reward, gfn_db_diff


    
    def apply_policy(self, observation, actor, *policy_args):
        '''
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        '''
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        out_dict = actor(observation)
        
        if do_explore:
            mu = out_dict['action']
            out_dict['mu'] = mu
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                action = torch.clamp(torch.rand_like(mu)*self.noise_std, -self.noise_clip, self.noise_clip)
            else:
                action = mu + torch.clamp(torch.rand_like(mu)*self.noise_std, 
                                                      -self.noise_clip, self.noise_clip)
            out_dict['action'] = action
            out_dict['sigma'] = self.noise_std
        return out_dict

