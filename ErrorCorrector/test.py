from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import *
from utils import setup_logger
from model import *
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from Utils.Logger import Logger
from Utils.utils import *
import numpy as np

def test (args, shared_model, env_conf, datasets, rank=-1):
    ptitle('Training Agent: {}'.format(rank))
    print ('Start training agent: ', rank)
    
    if rank == -1:
        logger = Logger (args.log_dir)
        train_step = 0

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    raw, lbl, prob, gt_lbl = datasets
    env = EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl)

        # env.seed (args.seed + rank)
    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm (env.observation_space.shape, env_conf["num_action"], args.hidden_feat)
    player.model.eval ()

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.state = player.state.cuda ()
            player.model = player.model.cuda ()
    player.model.train ()

    if rank == -1:
        eps_reward = 0
        pinned_eps_reward = 0
        mean_log_prob = 0

    log_train_period = 30
    logging = True

    while True:
        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.model.load_state_dict (shared_model.state_dict ())
        else:
            player.model.load_state_dict (shared_model.state_dict ())
        
        if player.done:
            player.eps_len = 0
            if rank == -1:
                # if log_train_period <= train_step < log_train_period + 6:
                print ("test: eps_reward", eps_reward) 
                    # if train_step > log_train_period + 6:
                    #     log_train_period += 30
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
            player.action_train()
            if rank == -1:
                # if log_train_period <= train_step < log_train_period + 6:
                # print ("action = ", player.action)
                eps_reward += player.reward
                mean_log_prob += player.log_probs [-1] / env_conf ["T"]
            if player.done:
                break

        if player.done:
            state = player.env.reset ()
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()

        R = torch.zeros (1, 1)
        if not player.done:
            value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
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

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        # player.model.zero_grad ()
        # sum_loss = (policy_loss + value_loss)
        # sum_loss.backward ()
        # ensure_shared_grads (player.model, shared_model, gpu=gpu_id >= 0)
        # optimizer.step ()
        player.clear_actions ()

        if rank == -1:
            train_step += 1
            if train_step % args.log_period == 0:
                log_info = {
                    # 'train: sum_loss': sum_loss, 
                    'test: value_loss': value_loss, 
                    'test: policy_loss': policy_loss, 
                    'test: advanage': advantage,
                    # 'train: entropy': entropy,
                    'test: eps reward': pinned_eps_reward,
                    # 'train: mean log prob': mean_log_prob
                }

                for tag, value in log_info.items ():
                    logger.scalar_summary (tag, value, train_step)

# def test (args, shared_model, env_conf, datasets):
#     ptitle ('Test agent')
#     gpu_id = args.gpu_ids [-1]
#     log = {}

#     logger = Logger (args.log_dir)

#     setup_logger ('{}_log'.format (args.env), r'{0}{1}_log'.format (args.log_dir, args.env))
#     log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
#         args.env))
#     d_args = vars (args)
#     for k in d_args.keys ():
#         log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, d_args[k]))

#     torch.manual_seed (args.seed)

#     if gpu_id >= 0:
#         torch.cuda.manual_seed (args.seed)
        
#     raw, lbl, prob, gt_lbl = datasets
#     env = EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl)
#     reward_sum = 0
#     start_time = time.time ()
#     num_tests = 0
#     reward_total_sum = 0

#     player = Agent (None, env, args, None)
#     player.gpu_id = gpu_id
#     player.model = A3Clstm (env.observation_space.shape, env_conf["num_action"], args.hidden_feat)
#     player.state = player.env.reset ()
#     player.state = torch.from_numpy (player.state).float ()
#     if gpu_id >= 0:
#         with torch.cuda.device (gpu_id):
#             player.model = player.model.cuda ()
#             player.state = player.state.cuda ()

#     flag = True

#     create_dir (args.save_model_dir)

#     recent_episode_scores = []
#     renderlist = []
#     renderlist.append (player.env.render ())
#     max_score = 0
#     while True:
#         if flag:
#             if gpu_id >= 0:
#                 with torch.cuda.device (gpu_id):
#                     print ("-------------- test load -----------------")
#                     player.model.load_state_dict (shared_model.state_dict ())
#             else:
#                 player.model.load_state_dict (shared_model.state_dict ())
#             # player.model.train ()
#             flag = False
#         for i in range (4):
#             player.action_test ()
#             if player.done:
#                 break
#         # reward_sum += player.reward
#         # renderlist.append (player.env.render ()) 

#         if player.done:
#             flag = True
#             # if gpu_id >= 0:
#             #     with torch.cuda.device (gpu_id):
#             #         player.state = player.state.cuda ()

#             # num_tests += 1
#             # reward_total_sum += reward_sum
#             # reward_mean = reward_total_sum / num_tests
#             # log ['{}_log'.format (args.env)].info (
#             #     "Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".
#             #     format (
#             #         time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
#             #         reward_sum, player.eps_len, reward_mean, num_tests))

#             # recent_episode_scores += [reward_sum]
#             # if len (recent_episode_scores) > 200:
#             #     recent_episode_scores.pop (0)

#             # if args.save_max and np.mean (recent_episode_scores) >= max_score:
#             #     max_score = np.mean (recent_episode_scores)
#             #     if gpu_id >= 0:
#             #         with torch.cuda.device (gpu_id):
#             #             state_to_save = player.model.state_dict ()
#             #             torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, 'best_model_' + args.env))

#             # if num_tests % args.save_period == 0:
#             #     if gpu_id >= 0:
#             #         with torch.cuda.device (gpu_id):
#             #             state_to_save = player.model.state_dict ()
#             #             torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, args.env + '_' + str (num_tests)))

#             # if num_tests % args.log_period == 0:
#             #     print ("------------------------------------------------")
#             #     print ("Log test #:", num_tests)
#             #     print ("Actions :", player.actions)
#             #     print ("Actions transformed: ")
#             #     print (player.actions_explained)
#             #     print ("rewards: ", player.rewards)
#             #     print ("sum rewards: ", reward_sum)
#             #     print ("------------------------------------------------")
#             #     log_img = np.concatenate (renderlist, 0)
#             #     log_info = {"traning_sample": log_img}
#             #     for tag, img in log_info.items ():
#             #         img = img [None]
#             #         logger.image_summary (tag, img, num_tests)

#             #     log_info = {'mean_reward': reward_mean}
#             #     for tag, value in log_info.items ():
#             #         logger.scalar_summary (tag, value, num_tests)

#             renderlist = []
#             reward_sum = 0
#             player.eps_len = 0
                       
#             time.sleep (15)
#             player.clear_actions ()
#             state = player.env.reset ()
#             player.state = torch.from_numpy (state).float ()
#             if gpu_id >= 0:
#                 with torch.cuda.device (gpu_id):
#                     player.state = player.state.cuda ()








        

