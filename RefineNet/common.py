# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu
import random
import time
import multiprocessing
from tqdm import tqdm
from six.moves import queue

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs

from refineEnv import *

def play_one_episode(env, func, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.01 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        if random.random() < 0.01:
            spc = env.action_space
            act = spc.sample()

        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r


def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    for k in range(nr):
        score = play_one_episode(player, predfunc, render=render)
        print("{}/{}, score={}".format(k, nr, score))


def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        r = q.get()
        stat.feed(r)
        if verbose:
            logger.info("Score: {}".format(r))

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    # waiting is necessary, otherwise the estimated mean score is biased
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()

    if stat.count > 0:
        return (stat.average, stat.max)
    return (0, 0)


def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> [#action]
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean, max = eval_with_funcs(
            [pred] * NR_PROC, nr_eval,
            get_player_fn, verbose=True)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)

class LogVisualizeEpisode (Callback):

    def __init__ (self, input_names, output_names, get_player_fn):
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph (self):
        self.pred = self.trainer.get_predictor (self.input_names, self.output_names)

    def _trigger (self):
        player = self.get_player_fn ()
        current_obs = player.reset ()
        log_imgs = [player.render ()]

        done = False
        step = 0
        tot_reward = 0
        stack = []

        tab = '    '
        print ('Log episode: ')
        print ('img_id:', player.state.img_id)

        while not done:
            step += 1
            # Choosing action
            state_action_values = self.pred (np.expand_dims (current_obs, 0)) [0]
            state_action_values = np.squeeze (state_action_values)
            action = np.argmax (state_action_values)
            
            ##############Done choosing action############
            level = player.state.node.level
            line_header = ''
            for i in range (level + 1):
                line_header += tab

            obs_from_step, reward, done, info = player.step (action)

            print (line_header, 'Step :', step)
            print (line_header, 'State_action_value:')
            print (line_header, state_action_values)
            print (line_header, 'Choose:', action)
            
            if (info['down_level']):
                stack += [ { 'old_obs': current_obs,
                             'old_score': info ['current_score'],
                             'old_act': action
                            }]
                current_obs = obs_from_step
                log_imgs += [player.render ()]
                continue
            print (line_header, 'reward:', reward)

            current_obs = player.observation ()
            log_imgs += [player.render ()]
            level = player.state.node.level
            line_header = ''
            for i in range (level + 1):
                line_header += tab

            if (info['up_level']):
                old_info = stack.pop ()
                old_score = old_info ['old_score']
                old_obs = old_info ['old_obs']
                old_act = old_info ['old_act']
                new_score = player.cal_metric ()
                delayed_reward = new_score - old_score
                done=False
                print (line_header, 'Reward:', delayed_reward)
                            
            # if player.get_state ().done != True: 

        concated_img = np.concatenate (log_imgs, 0)
        self.trainer.monitors.put_image ('Log episode', concated_img)
        print ('FINAL SCORE:', player.cal_metric ())
