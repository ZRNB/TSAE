import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
from typing import Dict, List, Tuple, Union, cast

import utils
from model.agent.BaseRLAgent import BaseRLAgent
from model.components import DNN

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


class ItemWeightModel(nn.Module):
    """
    Item Weight Model for learning per-item importance weights in GFN.
    
    This model learns which items are more important for predicting long-term user value.
    It takes current state, action, next state, and reward as input and outputs normalized
    per-item weights using softmax.
    """
    
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - weight_hidden_dims: hidden dimensions for item weight MLP
        - weight_dropout_rate: dropout rate in weight model
        """
        parser.add_argument('--weight_hidden_dims', type=int, nargs='+', default=[128, 64],
                            help='hidden dimensions for item weight model')
        parser.add_argument('--weight_dropout_rate', type=float, default=0.2,
                            help='dropout rate for item weight model')
        return parser
    
    def __init__(self, state_dim, action_dim, hidden_dims, dropout_rate=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Input: state + action + next_state + reward (4 * state_dim for reward)
        input_dim = state_dim * 3 + 1
        self.net = DNN(input_dim, hidden_dims, action_dim,
                       dropout_rate=dropout_rate, do_batch_norm=True)
    
    def forward(self, state, action, next_state, reward):
        """
        @input:
        - state: (B, state_dim) - current user state
        - action: (B, action_dim) - action embedding
        - next_state: (B, state_dim) - next user state
        - reward: (B, 1) - trajectory reward
        @output:
        - weights: (B, action_dim) - normalized per-item weights (softmax over items)
        """
        # Concatenate all features
        combined = torch.cat([state, action, next_state, reward], dim=1)
        
        # Get raw weights from MLP
        raw_weights = self.net(combined)
        
        # Apply softmax to get normalized weights (per-item importance)
        weights = F.softmax(raw_weights, dim=1)
        
        return weights

class GFN4Retention_wif_teacher(BaseRLAgent):
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
        parser.add_argument('--weight_lr', type=float, default=1e-4,
                                help='learning rate for item weight model')
        parser.add_argument('--lambda_weight', type=float, default=0.1,
                                help='weight for item weight loss')
        parser = ItemWeightModel.parse_model_args(parser)
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
        
        self.weight_lr = args.weight_lr
        self.lambda_weight = args.lambda_weight
        
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
        
        # Item Weight Model initialization
        self.item_weight_model = ItemWeightModel(
            state_dim=self.actor.state_dim,
            action_dim=self.actor.action_dim,
            hidden_dims=args.weight_hidden_dims,
            dropout_rate=args.weight_dropout_rate
        ).to(self.device)
        self.item_weight_optimizer = torch.optim.Adam(
            self.item_weight_model.parameters(),
            lr=args.weight_lr,
            weight_decay=args.actor_decay
        )
        
        # register models that will be saved
        self.registered_models.append((self.critic1, self.critic1_optimizer, "_critic1"))
        self.registered_models.append((self.critic2, self.critic2_optimizer, "_critic2"))
        self.registered_models.append((self.teacher_flow_estimator, self.critic1_optimizer_teacher, "_critic1_teacher"))
        self.registered_models.append((self.teacher_backward_estimator, self.critic2_optimizer_teacher, "_critic2_teacher"))
        self.registered_models.append((self.item_weight_model, self.item_weight_optimizer, "_item_weight"))
        
        
    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({'Q1_loss': [], 'Q2_loss': [],
                                      'Q1': [], 'Q2': [], 'next_Q1': [], 'next_Q2': [], 'target_Q': [], 'DB_forward': [], 'DB_backward': [], 'DB_forward_teacher': [], 'DB_backward_teacher': [],
                                       'teacher_loss': [], 'teacher_discrepancies_reward': [], 'teacher_student_weight_reward': [],
                                       'item_weight_loss': [], 'mean_item_weight': [], 'max_item_weight': [], 'min_item_weight': []})
        self.eval_history.update({'avg_retention': [], 'max_retention': [], 'min_retention': []})
        

    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
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
                self.current_sum_reward *= 0
            
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())
            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, updated_info['updated_observation'])
            observation = new_observation
        return new_observation
    
    
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


    def update_item_weight_model(self, observation, policy_output, user_feedback, next_observation, discrepancies):
        """
        Update the item weight model using policy gradient.
        
        The weight model learns which items are more important by maximizing
        the advantage-weighted log probability of the learned weights.
        
        @input:
        - observation: current user observation
        - policy_output: policy output from current observation
        - user_feedback: user feedback including rewards
        - next_observation: next user observation
        - discrepancies: flow discrepancies for advantage calculation
        @output:
        - item_weight_loss: the loss value for monitoring
        """
        is_train = True
        
        # Get current policy outputs
        current_policy_output = self.apply_policy(observation, self.actor, self.epsilon, True, is_train)
        next_policy_output = self.apply_policy(next_observation, self.actor, self.epsilon, True, is_train)
        
        # Extract features for weight model
        state_emb = current_policy_output['state']
        action_emb = current_policy_output['action']
        next_state_emb = next_policy_output['state']
        traj_reward = user_feedback['reward'].view(-1, 1)
        
        # Compute item weights: (B, action_dim)
        item_weights = self.item_weight_model(state_emb, action_emb, next_state_emb, traj_reward)
        
        # Compute advantage for weight learning
        # Use the GFN discrepancies as advantage - positive discrepancy means forward flow > backward flow
        advantage = discrepancies[0].detach()  # (B, 1) or (B,)
        
        # Ensure advantage has correct shape
        if advantage.dim() == 2:
            advantage = advantage.squeeze(-1)  # (B,)
        
        # Compute weight loss using policy gradient
        # Maximize: sum_i (weight_i * advantage), which encourages important items to have higher weights
        # Equivalent to minimizing: -sum_i (weight_i * advantage)
        
        # Option 1: Direct advantage-weighted loss
        # weight_loss = -torch.mean(torch.sum(item_weights * advantage.unsqueeze(1), dim=1))
        
        # Option 2: Use entropy-regularized policy gradient to encourage exploration
        entropy_bonus = -torch.sum(item_weights * torch.log(item_weights + 1e-8), dim=1).mean()
        policy_gradient_loss = -torch.mean(torch.sum(item_weights * advantage.unsqueeze(1), dim=1))
        
        # Combined loss with entropy regularization
        item_weight_loss = policy_gradient_loss - 0.01 * entropy_bonus
        
        # Update weight model
        self.item_weight_optimizer.zero_grad()
        item_weight_loss.backward(retain_graph=True)
        self.item_weight_optimizer.step()
        
        return item_weight_loss
    
    
    def step_train(self):
        '''
        @process:
        '''
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        #reward = user_feedback['reward']
        #reward = reward.to(torch.float)
        done_mask = done_mask.to(torch.float)
        
        # Student GFN loss with item weights
        loss, discrepancies, student_reward, item_weights = self.get_gfn_loss(
            observation, policy_output, user_feedback, done_mask, next_observation, 
            is_teacher=False, return_weights=True
        )
        teacher_discrepancies_reward, teacher_student_weight_reward, teacher_reward = get_teacher_reward(discrepancies, student_reward, C=self.C, alpha=self.alpha)
        teacher_loss, _, _ = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation, is_teacher=True, new_reward=teacher_reward)

        # Item Weight Model learning using policy gradient
        # The weight model learns which items are important by maximizing advantage-weighted log probability
        item_weight_loss = self.update_item_weight_model(
            observation, policy_output, user_feedback, next_observation, discrepancies
        )

        self.training_history['actor_loss'].append(loss.item())
        self.training_history['teacher_loss'].append(teacher_loss.item())
        self.training_history['teacher_discrepancies_reward'].append(teacher_discrepancies_reward.mean().item())
        self.training_history['teacher_student_weight_reward'].append(teacher_student_weight_reward.mean().item())
        self.training_history['Q1_loss'].append(0)
        self.training_history['Q2_loss'].append(0)
        self.training_history['Q1'].append(0)
        self.training_history['Q2'].append(0)
        self.training_history['next_Q1'].append(0)
        self.training_history['next_Q2'].append(0)
        self.training_history['target_Q'].append(0)
        
        # Record item weight statistics
        if item_weights is not None:
            self.training_history['item_weight_loss'].append(item_weight_loss.item())
            self.training_history['mean_item_weight'].append(torch.mean(item_weights).item())
            self.training_history['max_item_weight'].append(torch.max(item_weights).item())
            self.training_history['min_item_weight'].append(torch.min(item_weights).item())
        
        #target_Q, next_Q1, next_Q2, Q1_loss, Q1, Q2_loss, Q2 = critic_loss_list
        #self.training_history['actor_loss'].append(actor_loss.item())
        #self.training_history['Q1_loss'].append(Q1_loss)
        #self.training_history['Q2_loss'].append(Q2_loss)
        #self.training_history['Q1'].append(Q1)
        #self.training_history['Q2'].append(Q2)
        #self.training_history['next_Q1'].append(next_Q1)
        #self.training_history['next_Q2'].append(next_Q2)
        #self.training_history['target_Q'].append(target_Q)

    
    
    def get_gfn_loss(self, observation, policy_output, user_feedback, done_mask, next_observation, is_teacher=False, new_reward=None, return_weights=False):
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
        
        # Calculate immediate reward
        user_feedback['immediate_response_weight'] = self.env.response_weights
        point_reward = user_feedback['immediate_response'].reshape(-1, 6, 7) * user_feedback['immediate_response_weight'].view(1,1,-1)
        combined_reward = torch.sum(point_reward, dim = 2)
        immediate_reward = torch.mean(combined_reward, dim = 1)
        
        # Compute item weights using the weight model (only for student, not teacher)
        item_weights = None
        if not is_teacher:
            state_emb = current_policy_output['state']
            action_emb = current_policy_output['action']
            next_state_emb = next_policy_output['state']
            traj_reward = user_feedback['reward'].view(-1, 1)
            
            # Compute item weights: (B, action_dim) - per-item importance
            item_weights = self.item_weight_model(state_emb, action_emb, next_state_emb, traj_reward)
            
            # Apply weights to immediate_reward for DB_backward_nonterminal
            # immediate_reward: (B,) - per-batch-item mean reward
            # item_weights: (B, action_dim) - per-item weights
            # weighted_immediate: (B,) - weighted sum of item rewards
            weighted_immediate_reward = torch.sum(immediate_reward.unsqueeze(1) * item_weights, dim=1)
            
            DB_backward_nonterminal = log_F_next.squeeze() + log_P_B.squeeze() + torch.mul(weighted_immediate_reward, zzz)
        else:
            # For teacher, use unweighted immediate reward
            DB_backward_nonterminal = log_F_next.squeeze() + log_P_B.squeeze() + torch.mul(immediate_reward, zzz)
        
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
        
        loss = total_loss
        student_log_reward = torch.log(user_feedback['reward'].view(-1, 1) + 0.001 + b_r)
        
        if return_weights:
            return loss, discrepancies, student_log_reward, item_weights
        return loss, discrepancies, student_log_reward


    
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

