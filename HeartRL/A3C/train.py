from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import *
from utils import ensure_shared_grads
from model import *
from player_util import Agent
from torch.autograd import Variable
from Utils.Logger import Logger

import numpy as np

def train (rank, args, shared_model, optimizer, env_conf, datasets):
    ptitle('Training Agent: {}'.format(rank))
    print ('Start training agent: ', rank)
    
    if rank == 0:
        logger = Logger (args.log_dir)
        train_step = 0

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    raw, gt_lbl = datasets
    env = EM_env (raw, gt_lbl, env_conf)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop (shared_model.parameters (), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam (shared_model.parameters (), lr=args.lr, amsgrad=args.amsgrad)
    gamma = torch.tensor (1.0)
    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            gamma = gamma.cuda ()
        # env.seed (args.seed + rank)
    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    # player.model = A3Clstm (env.observation_space.shape, env_conf["num_action"], args.hidden_feat)
    player.model = SimpleCNN (env.observation_space.shape, env_conf["num_action"])

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.state = player.state.cuda ()
            player.model = player.model.cuda ()
    player.model.train ()

    if rank == 0:
        eps_reward = 0
        pinned_eps_reward = 0
        mean_log_prob = 0

    while True:
        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.model.load_state_dict (shared_model.state_dict ())
        else:
            player.model.load_state_dict (shared_model.state_dict ())
        
        if player.done:
            player.eps_len = 0
            if rank == 0:
                pinned_eps_reward = eps_reward
                eps_reward = 0
                mean_log_prob = 0

        for step in range(args.num_steps):
            player.action_train()
            # print ('step: ', step, 'reward_len: ', len (player.rewards))
            if rank == 0:
                eps_reward += player.reward
                # mean_log_prob += player.log_probs [-1] 
            if player.done:
                break

        if player.done:
            state = player.env.reset ()
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _ = player.model(Variable(player.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        # print ("updating -------------------")
        # print ("values:", player.values)
        # print ("gamma:", args.gamma)
        # print ("rewards:", player.rewards)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # print ("advatage: ", advantage)
            # print ("value_loss: ", value_loss)
            # print ("delta_t: ", player.values[i + 1].data + player.rewards[i])
            # Generalized Advantage Estimataion
            delta_t = player.values[i + 1].data * args.gamma + player.rewards[i] - \
                        player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad ()
        sum_loss = (policy_loss + value_loss)
        sum_loss.backward ()
        ensure_shared_grads (player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step ()
        player.clear_actions ()

        if rank == 0:
            train_step += 1
            if train_step % (args.log_period) == 0:
                log_info = {
                    'train: sum_loss': sum_loss, 
                    'train: value_loss': value_loss, 
                    'train: policy_loss': policy_loss, 
                    'train: advanage': advantage,
                    # 'entropy': entropy,
                    'train: eps reward': pinned_eps_reward,
                    # 'mean log prob': mean_log_prob
                }

                for tag, value in log_info.items ():
                    logger.scalar_summary (tag, value, train_step)








