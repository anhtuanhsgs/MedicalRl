from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.prob_cpu = []

    def action_train (self):
        # value, logit, (self.hx, self.cx) = self.model((Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
        value, logit= self.model(Variable(self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        print ("action = ", action)
        self.action = action.cpu().numpy() [0][0]
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy() [0][0])
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
            # if self.done:
            #     if self.gpu_id >= 0:
            #         with torch.cuda.device(self.gpu_id):
            #             self.cx = Variable(
            #                 torch.zeros(1, self.args.hidden_feat).cuda())
            #             self.hx = Variable(
            #                 torch.zeros(1, self.args.hidden_feat).cuda())
            #     else:
            #         self.cx = Variable(torch.zeros(1, self.args.hidden_feat))
            #         self.hx = Variable(torch.zeros(1, self.args.hidden_feat))
            # else:
            #     self.cx = Variable(self.cx.data)
            #     self.hx = Variable(self.hx.data)
            value, logit = self.model(Variable(self.state.unsqueeze(0)))
        prob = F.softmax (logit, dim=1)
        prob_cpu = prob.cpu ().numpy ()
        prob_cpu = prob_cpu.reshape (self.env.agent_out_shape)
        self.prob_cpu = prob_cpu
        action = prob.max (1)[1].data.cpu ().numpy ()
        state, self.reward, self.done, self.info = self.env.step (action [0])
        self.rewards.append (self.reward)
        self.actions.append (action [0])
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