from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import *
from utils import ensure_shared_grads
from model import *
from player_util import Agent, Agent_continuous
from torch.autograd import Variable
from Utils.Logger import Logger

import numpy as np

def train (rank, args, shared_model, optimizer, env_conf, datasets=None):
    ptitle('Training Agent: {}'.format(rank))
    print ('Start training agent: ', rank)
    
    if rank == 0:
        logger = Logger (args.log_dir)
        train_step = 0

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    env_conf ["env_gpu"] = gpu_id
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    if "EM_env" in args.env:
        raw, lbl, prob, gt_lbl = datasets
        env = EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl)
    else:
        env = Voronoi_env (env_conf)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop (shared_model.parameters (), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam (shared_model.parameters (), lr=args.lr, amsgrad=args.amsgrad)

        # env.seed (args.seed + rank)
    if not args.continuous:
        player = Agent (None, env, args, None)
    else:
        player = Agent_continuous (None, env, args, None)
    player.gpu_id = gpu_id
    if not args.continuous:
        player.model = A3Clstm (env.observation_space.shape, env_conf["num_action"], args.hidden_feat)
    else:
        player.model = A3Clstm_continuous (env.observation_space.shape, env_conf["num_action"], args.hidden_feat)

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()
    old_score = player.env.old_score
    final_score = 0

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.state = player.state.cuda ()
            player.model = player.model.cuda ()
    player.model.train ()

    if rank == 0:
        eps_reward = 0
        pinned_eps_reward = 0
        mean_log_prob = 0

    # print ("rank: ", rank)

    while True:
        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.model.load_state_dict (shared_model.state_dict ())
        else:
            player.model.load_state_dict (shared_model.state_dict ())
        
        if player.done:
            player.eps_len = 0
            if rank == 0:
                if 0 <= (train_step % args.train_log_period) < args.max_episode_length:
                    print ("train: step", train_step, "\teps_reward", eps_reward, 
                        "\timprovement", final_score - old_score)
                old_score = player.env.old_score
                pinned_eps_reward = eps_reward
                eps_reward = 0
                mean_log_prob = 0
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, args.hidden_feat).cuda())
                    player.hx = Variable(torch.zeros(1, args.hidden_feat).cuda())
            else:
                player.cx = Variable(torch.zeros(1, args.hidden_feat))
                player.hx = Variable(torch.zeros(1, args.hidden_feat))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        for step in range(args.num_steps):
            player.action_train ()
            if rank == 0:
                # if 0 <= (train_step % args.train_log_period) < args.max_episode_length:
                #     print ("train: step", train_step, "\taction = ", player.action)
                eps_reward += player.reward
                # print (eps_reward)
                mean_log_prob += player.log_probs [-1] / env_conf ["T"]
            if player.done:
                break

        if player.done:
            # if rank == 0:
            #     print ("----------------------------------------------")
            final_score = player.env.old_score
            state = player.env.reset ()
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()

        R = torch.zeros (1, 1)
        if not player.done:
            if not args.continuous:
                value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            else:
                value, _, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
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

        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = player.values[i + 1].data * args.gamma + player.rewards[i] - \
                        player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t
            # print (player.rewards [i])
            if not args.continuous:
                policy_loss = policy_loss - \
                    player.log_probs[i] * \
                    Variable(gae) - 0.01 * player.entropies[i]
            else:
                policy_loss = policy_loss - \
                    player.log_probs[i].sum () * Variable(gae) - \
                    0.01 * player.entropies[i].sum ()

        player.model.zero_grad ()
        sum_loss = (policy_loss + value_loss)

        sum_loss.backward ()
        ensure_shared_grads (player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step ()
        player.clear_actions ()

        if rank == 0:
            train_step += 1
            if train_step % args.log_period == 0:
                log_info = {
                    # 'train: sum_loss': sum_loss, 
                    'train: value_loss': value_loss, 
                    'train: policy_loss': policy_loss, 
                    'train: advanage': advantage,
                    # 'train: entropy': entropy,
                    'train: eps reward': pinned_eps_reward,
                    # 'train: mean log prob': mean_log_prob
                }

                for tag, value in log_info.items ():
                    logger.scalar_summary (tag, value, train_step)








