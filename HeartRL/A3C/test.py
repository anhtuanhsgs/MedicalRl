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

def test (args, shared_model, env_conf, datasets):
    ptitle ('Test agent')
    gpu_id = args.gpu_ids [-1]
    log = {}

    logger = Logger (args.log_dir)

    setup_logger ('{}_log'.format (args.env), r'{0}{1}_log'.format (args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars (args)
    for k in d_args.keys ():
        log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, d_args[k]))

    torch.manual_seed (args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed (args.seed)
        
    raw, gt_lbl = datasets
    env = EM_env (raw, gt_lbl, env_conf)
    reward_sum = 0
    start_time = time.time ()
    num_tests = 0
    reward_total_sum = 0

    player = Agent (None, env, args, None)
    player.gpu_id = gpu_id
    # player.model = A3Clstm (env.observation_space.shape, env_conf["num_action"], args.hidden_feat)
    player.model = SimpleCNN (env.observation_space.shape, env_conf["num_action"])
    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()
    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.model = player.model.cuda ()
            player.state = player.state.cuda ()

    flag = True
    create_dir (args.save_model_dir)

    recent_episode_scores = []
    renderlist = []
    renderlist.append (player.env.render ())
    max_score = 0
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):                    
                    player.model.load_state_dict (shared_model.state_dict ())
            else:
                player.model.load_state_dict (shared_model.state_dict ())
            player.model.eval ()
            flag = False

        player.action_test ()
        reward_sum += player.reward
        renderlist.append (player.env.render ()) 

        if player.done:
            flag = True
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()

            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log ['{}_log'.format (args.env)].info (
                "Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".
                format (
                    time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
                    reward_sum, player.eps_len, reward_mean, num_tests))

            recent_episode_scores += [reward_sum]
            if len (recent_episode_scores) > 200:
                recent_episode_scores.pop (0)

            if args.save_max and np.mean (recent_episode_scores) >= max_score:
                max_score = np.mean (recent_episode_scores)
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, 'best_model_' + args.env))

            if num_tests % args.save_period == 0:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, args.env + '_' + str (num_tests)))

            if num_tests % args.log_period == 0:
                print ("------------------------------------------------")
                print ("Log test #:", num_tests)
                print ("Prob: ")
                for i in range (player.env.agent_out_shape [1]):
                    for j in range (player.env.agent_out_shape [2]):
                        print ("{:.3f}\t".format (player.prob_cpu [0, i, j]), end='')
                    print ()
                print ("Actions :", player.actions)
                print ("Actions transformed: ")
                print (player.actions_explained)
                print ("rewards: ", player.rewards)
                print ("sum rewards: ", reward_sum)
                print ("------------------------------------------------")
                log_img = np.concatenate (renderlist, 0)
                log_info = {"test: traning_sample": log_img}
                for tag, img in log_info.items ():
                    img = img [None]
                    logger.image_summary (tag, img, num_tests)

                log_info = {'test: mean_reward': reward_mean}
                for tag, value in log_info.items ():
                    logger.scalar_summary (tag, value, num_tests)

            renderlist = []
            reward_sum = 0
            player.eps_len = 0
            time.sleep (30)
            player.clear_actions ()
            state = player.env.reset ()
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()








        

