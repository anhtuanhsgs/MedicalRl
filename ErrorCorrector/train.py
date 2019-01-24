from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import EM_env
from utils import ensure_shared_grads
from model import CNN
from player_util import Agent
from torch.autograd import Variable
import numpy as np

def train (rank, args, shared_model, optimizer, env_conf, datasets):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    raw, lbl, prob, gt_lbl = datasets
    env = EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop (shared_model.parameters (), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam (shared_model.parameters (), lr=args.lr, amsgrad=args.amsgrad)

        # env.seed (args.seed + rank)
        player = Agent (None, env, args, None)
        player.gpu_id = gpu_id
        player.model = CNN (env.observation_space.shape, env_conf["num_action"])

        player.state = player.env.reset ()
        player.state = torch.from_numpy (player.state).float ()

        if gpu_id >= 0:
            with torch.cuda.device (gpu_id):
                player.state = player.state.cuda ()
                player.model = player.model.cuda ()
        player.model.train ()

        while True:
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.model.load_state_dict (shared_model.state_dict ())
            else:
                player.model.load_state_dict (shared_model.state_dict ())
            
            for step in range(args.num_steps):
                player.action_train()
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
                value, _ = player.model (Variable (player.state.unsqueeze (0)))
                R = value.data

            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    R = R.cuda ()

            player.values.append (Variable (R))
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros (1, 1)

            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    R = R.cuda ()

            player.values.append (Variable (R))
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros (1, 1)
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    gae = gae.cuda ()
            R = Variable (R)
            for i in reversed (range (len (player.rewards))):
                R = args.gamma * R + player.rewards [i]
                advantage = R - player.values [i]
                value_loss = value_loss + 0.5 * advantage.abs ()

                delta_t = player.rewards [i] + args.gamma * player.values [i + 1].data - player.values [i].data
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - player.log_probs [i] * Variable (gae) - 0.01 * player.entropies [i]

            player.model.zero_grad ()
            (policy_loss + 0.5 * value_loss).backward ()
            ensure_shared_grads (player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step ()
            player.clear_actions ()



