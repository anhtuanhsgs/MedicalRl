"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch
from torch.autograd import Variable
import sys
import os
import gym.spaces
import itertools
import numpy as np
import copy
import random
from collections import namedtuple
from utils.replay_buffer import *
from utils.schedules import *
from utils.gym_setup import *
from logger import Logger
import time

from refineEnv import *
import skimage.io as io

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
LOG_EVERY_N_STEPS = 100
SAVE_MODEL_EVERY_N_STEPS = 1000
LOG_SAMPLE_EVERY_N_STEPS = 5000
MAX_LV = 4

# Set the logger
logger = Logger('./logs')
def to_np(x):
    return x.data.cpu().numpy() 

def learn (replay_buffer, Q, gamma, Q_target, optimizer, t, logger, batch_size, dqn_device, 
            double_dqn, num_param_updates, target_update_freq):
    # sample transition batch from replay memory
    # done_mask = 1 if next state is end of episode
    obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)

    #######DEBUG##############
    # tmp = obs_t [0]
    # img_a = tmp[0, 0]
    # tmp = obs_tp1 [0]
    # img_b = tmp[0, 0]
    # print ('action', act_t [0])
    # print ('done', done_mask [0])
    # plt.imshow (np.concatenate ([img_a, img_b], 0), cmap='gray')
    # plt.show ()
    #########################

    obs_t = torch.tensor(obs_t, device=dqn_device, dtype=torch.float) / 255.0
    act_t = torch.tensor(act_t, device=dqn_device, dtype=torch.long) 
    rew_t = torch.tensor(rew_t, device=dqn_device, dtype=torch.float) 
    obs_tp1 = torch.tensor(obs_tp1, device=dqn_device, dtype=torch.float)  / 255.0
    done_mask = torch.tensor(done_mask, device=dqn_device, dtype=torch.float) 
    # input batches to networks
    # get the Q values for current current_obs (Q(s,a, theta_i))
    q_values = Q(obs_t)
    q_s_a = q_values.gather(1, act_t.unsqueeze(1))
    q_s_a = q_s_a.squeeze()

    if (double_dqn):
        # ---------------
        #   double DQN
        # ---------------

        # get the Q values for best actions in obs_tp1 
        # based off the current Q network
        # max(Q(s', a', theta_i)) wrt a'
        q_tp1_values = Q(obs_tp1).detach()
        _, a_prime = q_tp1_values.max(1)

        # get Q values from frozen network for next state and chosen action
        # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        q_target_tp1_values = Q_target(obs_tp1).detach()
        q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        # if current state is end of episode, then there is no next Q value
        q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime 

        error = rew_t + gamma * q_target_s_a_prime - q_s_a
    else:
        # ---------------
        #   regular DQN
        # ---------------

        # get the Q values for best actions in obs_tp1 
        # based off frozen Q network
        # max(Q(s', a', theta_i_frozen)) wrt a'
        q_tp1_values = Q_target(obs_tp1).detach()
        q_s_a_prime, a_prime = q_tp1_values.max(1)

        # if current state is end of episode, then there is no next Q value
        q_s_a_prime = (1 - done_mask) * q_s_a_prime 

        # Compute Bellman error
        # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
        error = rew_t + gamma * q_s_a_prime - q_s_a

    # clip the error and flip 
    clipped_error = -1.0 * error.clamp(-1, 1)

    # backwards pass
    optimizer.zero_grad()
    q_s_a.backward(clipped_error.detach ())

    # update
    optimizer.step()
    num_param_updates += 1

    # update target Q network weights with current Q network weights
    if num_param_updates % target_update_freq == 0:
        Q_target.load_state_dict(Q.state_dict())

    # (2) Log values and gradients of the parameters (histogram)
    if t % LOG_EVERY_N_STEPS == 0:
        for tag, value in Q.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), t+1)
            logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
    #####

def save_obs_tif (path, obs):
    raw = obs [..., 0].astype (np.uint8)
    mask = obs [..., 1].astype (np.uint8)
    refined = obs [..., 2].astype (np.uint8)
    zoomed = obs [..., 3].astype (np.uint8)
    raw = np.concatenate ([raw, raw.transpose (1, 0, 2), raw.transpose (2, 0, 1)], 2)
    mask = np.concatenate ([mask, mask.transpose (1, 0, 2), mask.transpose (2, 0, 1)], 2)
    refined =  np.concatenate ([refined, refined.transpose (1, 0, 2), refined.transpose (2, 0, 1)], 2)
    zoomed = np.concatenate ([zoomed, zoomed.transpose (1, 0, 2), zoomed.transpose (2, 0, 1)], 2)
    log_vol = np.concatenate ([raw, mask, refined, zoomed], 1)
    io.imsave (path, log_vol)
    print ('Done saving log_vol')


def dqn_learning(env,
          env_id,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(10000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=100000,
          batch_size=32,
          gamma=0.99,
          learning_starts=10000,
          learning_freq=4,
          frame_history_len=1,
          target_update_freq=10000,
          double_dqn=False,
          dueling_dqn=False):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    env_id: string
        gym environment id for model saving.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    # assert type(env.observation_space) == gym.spaces.Box
    # assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional current_obs (e.g. RAM)
        input_shape = env.observation_space.shape
        in_channels = input_shape[0]
    else:
        img_d ,img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_d, img_h, img_w, frame_history_len * img_c)
        in_channels = input_shape[3]
    num_actions = env.action_space.n

    # print ('in_channels', in_channels)
    
    # define Q target and Q 
    dqn_device = torch.device("cuda:0")
    Q = q_func(in_channels, num_actions, input_shape).type(dtype).to(dqn_device)
    Q_target = q_func(in_channels, num_actions, input_shape).type(dtype).to(dqn_device)

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ######

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()

    episode_rewards = [0]
    stack = []
    obs_act_stacks = []
    current_lv = 0

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # store last frame, returned idx used later
        # last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get current_obs to input to Q network (need to append prev frames)
        # current_obs = replay_buffer.encode_recent_observation()
        current_obs = copy.deepcopy (last_obs)
        if len (current_obs.shape) < 5:
            current_obs = current_obs [None]

        if (len (obs_act_stacks) <= current_lv):
            obs_act_stacks += [[]]


        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs_tr = current_obs.transpose (0, 4, 1, 2, 3)
                obs = torch.tensor(obs_tr, device=dqn_device, dtype=torch.float) / 255.0
                with torch.no_grad ():
                    q_value_all_actions = Q(torch.tensor(obs, device=dqn_device, dtype=torch.float)).cpu()
                action = ((q_value_all_actions).detach ().max(1)[1])[0]
                action = action.item ()
            else:
                action = np.random.randint(num_actions)

        current_level_cell_cnt = env.cell_count ()
        obs, reward, done, info = env.step(action)

        if action == 9 and not info['up_level']:
            obs_act_stacks [current_lv] += [
                {'old_obs': current_obs [0], 
                 'old_act': action,
                 'old_total_reward': reward,
                 'old_cell_count': 1}
            ]

        elif (info['down_level']):
            obs_act_stacks [current_lv] += [{'old_obs': current_obs [0], 'old_act': action}]
            current_lv += 1

        elif info['up_level'] or (current_lv == 0):
            # Flush all obs current level
            stk = obs_act_stacks [current_lv]
            reward_sum = 0
            assert (done == True)
            assert (reward == 0)

            for i in range (len (stk)):
                old_info = stk [i]
                old_tot_reward = old_info ['old_total_reward']
                old_act = old_info ['old_act']
                old_obs = old_info ['old_obs']
                old_cell_count = old_info ['old_cell_count']
                last_stored_frame_idx = replay_buffer.store_frame (old_obs)
                replay_buffer.store_effect (last_stored_frame_idx, old_act, old_tot_reward, False)
                reward_sum += old_tot_reward
                last_stored_frame_idx = replay_buffer.store_frame (current_obs [0])
                replay_buffer.store_effect (last_stored_frame_idx, action, reward, done)
            while (len (stk) > 0):
                stk.pop ()
            # Update reward sum in upper layer (stored in stack)
            if (current_lv > 0):
                current_lv -= 1
                old_info = obs_act_stacks [current_lv][-1]
                old_info ['old_total_reward'] = reward_sum
                old_info ['old_cell_count'] = current_level_cell_cnt
                done=False

        elif not info['up_level'] and not info['down_level'] and not (current_lv == 0):
            # CURRENT LEVEL IS LAST LEVEL
            assert (current_lv == MAX_LV - 1)
            last_stored_frame_idx = replay_buffer.store_frame(last_obs)
            replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

            # reset env if reached episode boundary
        if done:
            assert (current_lv == 0)
            env.reset()
            obs_act_stacks = []
            episode_rewards += [reward_sum]
            if (len(episode_rewards) > 100):
                episode_rewards.pop (0)
            # current_lv = 0

            # update last_obs
        last_obs = env.observation ()

        ### 3. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            learn (replay_buffer, Q, gamma, Q_target, optimizer, t, logger, batch_size, dqn_device, 
                double_dqn, num_param_updates, target_update_freq)
            

        ### 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = ''
            if (double_dqn):
                add_str = 'double' 
            if (dueling_dqn):
                add_str = 'dueling'
            model_save_path = "models/%s_%s_%d_%s.model" %(str(env_id), add_str, t, str(time.ctime()).replace(' ', '_'))
            torch.save(Q.state_dict(), model_save_path)

        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': episode_rewards[-1],
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    'best_mean_episode_reward': best_mean_episode_reward
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

        if t % LOG_SAMPLE_EVERY_N_STEPS == 0 and t >= LOG_SAMPLE_EVERY_N_STEPS and t >= learning_starts:
            print ('Sample episode step:', t)
            env_sample = get_medical_env ()
            done = False
            obs = env_sample.reset ()
            sample_reward = 0
            nstep = 0
            
            action_list = []
            Q.eval ()
            while not done:
                obs_t = torch.tensor(obs.transpose (3, 0, 1, 2)[None], device=dqn_device, dtype=torch.float) / 255.0
                with torch.no_grad ():
                    q_value_all_actions = Q(torch.tensor(obs_t, device=dqn_device, dtype=torch.float)).cpu()
                print (q_value_all_actions)
                action = ((q_value_all_actions).detach ().max(1)[1])[0]
                action = action.item ()

                obs, reward, done, info = env_sample.step(action)
                if (info['up_level']):
                    done = False
                sample_reward += reward
                nstep += 1
                action_list += [action]
            print ('sample reward: ', sample_reward)
            info = {
                'Sample reward': sample_reward
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            print (action_list)
            obs = env_sample.observation ()
            
            save_obs_tif ('sample_log/log_vol' + str(t + 1) + '.tif', obs)
            Q.train ()
