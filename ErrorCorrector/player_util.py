from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import normal  # , pi

class Agent (object):
    def __init__ (self, model, env, args, state):
        self.args = args

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        
        self.eps_len = 0
        self.done = True
        self.info = None
        self.reward = 0

        self.gpu_id = -1
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []

    def action_train (self, use_max=False):
        value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        # print ("train: prob", prob)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        
        if not use_max:
            action = prob.multinomial(1).data
            self.action = action.cpu().numpy() [0][0]
            log_prob = log_prob.gather(1, Variable(action))
            state, self.reward, self.done, self.info = self.env.step(
                action.cpu().numpy())
        else:
            with torch.no_grad():
                action = prob.max (1)[1].data
                self.action = action.cpu().numpy() [0]
                log_prob = log_prob.gather(1, Variable(action.unsqueeze (0)))
                state, self.reward, self.done, self.info = self.env.step(
                    self.action)
        if not use_max:
            self.state = torch.from_numpy(state).float()
        else:
            with torch.no_grad ():
                self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.eps_len += 1
        return self

    def action_test (self):
        with torch.no_grad():
            if self.done:
                # print ("re load")
                self.origin_score = self.env.old_score
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, self.args.hidden_feat).cuda())
                        self.hx = Variable(
                            torch.zeros(1, self.args.hidden_feat).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, self.args.hidden_feat))
                    self.hx = Variable(torch.zeros(1, self.args.hidden_feat))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, (self.hx, self.cx) = self.model((Variable (
                self.state.unsqueeze(0)), (self.hx, self.cx)))
            
        prob = F.softmax (logit, dim=1)
        action = prob.max (1)[1].data.cpu ().numpy ()
        state, self.reward, self.done, self.info = self.env.step (action [0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.rewards.append (self.reward)
        # print ("action test", self.rewards)
        self.actions.append (action [0])
        self.actions_explained.append (self.env.int2index (action [0], self.env.agent_out_shape))
        
        self.eps_len += 1
        return self

    def clear_actions (self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        return self

class Agent_continuous (object):
    def __init__ (self, model, env, args, state):
        self.args = args

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        
        self.eps_len = 0
        self.done = True
        self.info = None
        self.reward = 0

        self.gpu_id = -1
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []

    def action_train (self, print_log=False):
        self.state = self.state.unsqueeze(0)
        value, mu, sigma, (self.hx, self.cx) = self.model (
            (Variable(self.state), (self.hx, self.cx)))
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = sigma + 1e-3
        eps = torch.randn (mu.size())
        pi = np.array ([math.pi])
        pi = torch.from_numpy (pi).float ()
        
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                eps = Variable (eps).cuda()
                pi = Variable (pi).cuda()
        else:
            eps = Variable (eps)
            pi = Variable (pi)

        action = (mu + sigma.sqrt () * eps).data
        if (print_log):
            print (mu.cpu ().detach ().numpy ())
        # print (sigma.cpu (). detach ().numpy ())
        act = Variable (action)
        prob = normal (act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as (sigma)).log() + 1)
        self.entropies.append (entropy)
        log_prob = (prob + 1e-6).log ()
        self.log_probs.append(log_prob)
        state, self.reward, self.done, self.info = self.env.step (action.cpu().numpy()[0])

        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()

        self.reward = max(min(self.reward, 1), -1)
        # print ("Train: ", self.reward, "Done", self.done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.eps_len += 1
        return self

    def action_test (self):
        with torch.no_grad():
            if self.done:
                self.origin_score = self.env.old_score
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(torch.zeros(
                            1, self.args.hidden_feat).cuda())
                        self.hx = Variable(torch.zeros(
                            1, self.args.hidden_feat).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, self.args.hidden_feat))
                    self.hx = Variable(torch.zeros(1, self.args.hidden_feat))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            self.state = self.state.unsqueeze(0)
            value, mu, sigma, (self.hx, self.cx) = self.model(
                (Variable(self.state), (self.hx, self.cx)))
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()[0]
        state, self.reward, self.done, self.info = self.env.step(action)

        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.rewards.append (self.reward)
        self.actions.append ((action [0], action [1]))
        y_apx, x_apx = action [0], action [1]
        error_index = self.env.approx2index (y_apx, x_apx, self.env.raw.shape)
        self.actions_explained.append (error_index)
        self.eps_len += 1
        return self

    def clear_actions (self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        return self


