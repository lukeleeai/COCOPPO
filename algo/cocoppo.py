import tensorflow as tf
import copy
from network_models import cocoppo_cocob_optimizer as cocob_optimizer
from tensorflow.python.training.optimizer import Optimizer

'''
# cv, c1, c2, c3, alpha
0 0.2, 1, 0.01, 0.05, 0.01 --> slow convergence. better than previous attempts
1 0.2, 1, 0.01, 0.05, 0.05 --> slow, unstable
2 0.3, 1, 0.01, 0.05, 0.01 --> slow, but sometimes stable. yet unstable at first. 
3 0.3, 1, 0.01, 0.05, 0.03 --> slow, unstable
4 0.3, 1, 0.01, 0.05, 0.009 --> slow, but stable as training goes on.
5 0.3, 1, 0.01, 0.05, 0.008 --> fast as PPO. also stable!!!! ***
5_1                         --> maybe slow sometimes, but stable at last
6 0.3, 1, 0.01, 0.05, 0.005 --> unstable -> stable -> unstable --> really stable
7 0.3, 1, 0.01, 0.05, 0.01 --> hmm... not bad
8 0.3, 1, 0.01, 0.05, 0.01 --> great... bad only once. fast convergence. quite stable ***
8_1                        --> not bad! but somewhat unstable. 
8_2                          --> hmm... not satisfied
8_3 0.3, 1, 0.01, 0.05, 0.009 -->  less than expected.
10 0.3, 1, 0.01, 0.04, 0.01 -->  
11 0.3, 1, 0.01, 0.04, 0.008 --> 


best: 



depends on luck too much. need something trustworthy. 

ppo cv = 0.3 doesn't work well.  but cocoppo DOES.
 
 not bad: def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.3, c_1=1, c_2=0.01, c_3=0.1, alpha=0.01, max_reward=200):
0.3, 1, 0.01, 0.04~5 0.08

'''

class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.3, c_1=1, c_2=0.01, c_3=0.05, alpha=0.008, max_reward=500):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        :param c_3: parameter for COCOPPO loss
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma
        self.reward_ema = 0
        self.alpha = alpha
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.optimizer1 = cocob_optimizer.COCOB()
        self.c_3 = c_3
        self.max_reward = max_reward

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss'):
            # construct computation graph for loss_clip
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                            - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

            # construct computation graph for loss of entropy bonus
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            tf.summary.scalar('entropy', entropy)

            # construct computation graph for loss of value function
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('value_difference', loss_vf)

            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy

            # minimize -loss == maximize loss
            loss = -loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)

        self.train_op = self.optimizer1.minimize(loss)

        total_loss = self.add_cocoppo_loss(loss)
        tf.summary.scalar('real_total', total_loss)
        self.merged = tf.summary.merge_all()
        self.train_op = self.optimizer1.minimize(total_loss)

    def get_beta(self):
        beta_list = []
        for var in self.var_list:
            beta = self.optimizer1.get_slot(var, 'beta')
            if beta:
                beta = tf.reduce_mean(beta)
                beta_list.append(beta)
        beta = tf.reduce_mean(beta_list)
        return beta

    def add_cocoppo_loss(self, loss):
        beta = self.get_beta()
        tf.summary.scalar('beta', beta)
        reward_ema = self.alpha * self.reward_ema + (1 - self.alpha) * tf.reduce_sum(self.rewards)
        tf.summary.scalar('reward ema', reward_ema)
        loss_cocob = ((reward_ema / self.max_reward) ** 3) * beta
        loss += self.c_3 * loss_cocob
        tf.summary.scalar('cocoppo_loss', -1 * loss_cocob)
        return loss

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})
        # print(tf.global_variables())

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})
