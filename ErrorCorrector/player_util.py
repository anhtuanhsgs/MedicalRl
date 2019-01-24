from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Agent (object):
    def __init__ (self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.args = args
        
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

    def action_train (self):
        value, logit = self.model (Variable (self.state.unsqueeze (0)))
        prob = F.softmax (logit, dim=1)
        log_prob = F.log_softmax (logit, dim=1)
        entropy = -(log_prob * prob).sum (1)
        self.entropies.append (entropy)
        action = prob.multinomial (1).data
        log_prob = log_prob.gather (1, Variable (action))
        state, self.reward, self.done, self.info = self.env.step (
            action.cpu ().numpy ())
        self.state = torch.from_numpy (state).float ()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.reward = max (min (self.reward, 1), -1)
        self.values.append (value)
        self.log_probs.append (log_prob)
        self.rewards.append (self.reward)
        self.actions.append (action)

        return self

    def action_test (self):
        with torch.no_grad ():
            value, logit = self.model (Variable (
                self.state.unsqueeze (0)))
            prob = F.softmax (logit, dim=1)
            action = prob.max (1)[1].data.cpu ().numpy ()
            state, self.reward, self.done, self.info = self.env.step (action [0])
            self.actions_explained.append (self.env.int2index (action [0], self.env.agent_out_shape))
            if self.gpu_id >= 0:
                with torch.cuda.device (self.gpu_id):
                    self.state = self.state.cuda ()
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