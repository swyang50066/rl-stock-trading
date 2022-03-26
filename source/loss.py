import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.python.ops import math_ops


"""
Refer to below link of 'stable-baseline' project:

    https://stable-baselines.readthedocs.io/en/master/common/distributions.html
"""


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space
    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the appropriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert (
            len(ac_space.shape) == 1
        ), "Error: the action space must be a vector"
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space of type {}.".format(
                type(ac_space)
            )
            + " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


class CategoricalProbabilityDistribution:
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input
        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def neglogp(self, x):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=tf.stop_gradient(one_hot_actions)
        )

    def kl(self, other):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.reduce_max(
            other.logits, axis=-1, keepdims=True
        )
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(
            p_0 * (a_0 - tf.log(z_0) - a_1 + tf.log(z_1)), axis=-1
        )

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)
        uniform = tf.random_uniform(
            tf.shape(self.logits), dtype=self.logits.dtype
        )
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)


class MultiCategoricalProbabilityDistribution:
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input
        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(
            map(
                CategoricalProbabilityDistribution,
                tf.split(flat, nvec, axis=-1),
            )
        )

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.stack([p.mode() for p in self.categoricals], axis=-1)

    def neglogp(self, x):
        return tf.add_n(
            [
                p.neglogp(px)
                for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))
            ]
        )

    def kl(self, other):
        return tf.add_n(
            [p.kl(q) for p, q in zip(self.categoricals, other.categoricals)]
        )

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.stack([p.sample() for p in self.categoricals], axis=-1)


class DiagGaussianProbabilityDistribution(object):
    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input
        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(
            axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat
        )
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        super(DiagGaussianProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        return (
            0.5
            * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1)
            + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32)
            + tf.reduce_sum(self.logstd, axis=-1)
        )

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(
            other.logstd
            - self.logstd
            + (tf.square(self.std) + tf.square(self.mean - other.mean))
            / (2.0 * tf.square(other.std))
            - 0.5,
            axis=-1,
        )

    def entropy(self):
        return tf.reduce_sum(
            self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1
        )

    def sample(self):
        # Bounds are taken into acount outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mean + self.std * tf.random_normal(
            tf.shape(self.mean), dtype=self.mean.dtype
        )


class BernoulliProbabilityDistribution(object):
    def __init__(self, logits):
        """
        Probability distributions from Bernoulli input
        :param logits: ([float]) the Bernoulli input data
        """
        self.logits = logits
        self.probabilities = tf.sigmoid(logits)
        super(BernoulliProbabilityDistribution, self).__init__()

    def mode(self):
        return tf.round(self.probabilities)

    def neglogp(self, x):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=tf.cast(x, tf.float32)
            ),
            axis=-1,
        )

    def kl(self, other):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=other.logits, labels=self.probabilities
            ),
            axis=-1,
        ) - tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.probabilities
            ),
            axis=-1,
        )

    def entropy(self):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=self.probabilities
            ),
            axis=-1,
        )

    def sample(self):
        samples_from_uniform = tf.random_uniform(tf.shape(self.probabilities))
        return tf.cast(
            math_ops.less(samples_from_uniform, self.probabilities),
            tf.float32,
        )


def policy_loss_func(y_true, y_pred, alpha=0.01, eps=1.0e-16):
    """Policy loss function"""

    def _neglogp(x, mu, sigma):
        return 0.5 * K.sum(
            K.square((x - mu) / sigma), axis=-1, keepdims=True
        ) + 0.5 * K.sum(
            K.log(2.0 * np.pi * K.square(sigma) + eps),
            axis=-1,
            keepdims=True,
        )

    def _entropy(sigma):
        return K.sum(
            K.log(sigma + eps) + 0.5 * np.log(2.0 * np.pi * np.e),
            axis=-1,
            keepdims=True,
        )

    # Parse inputs
    mu, log_sigma = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    action, advantage = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get negative logarithmic of Gaussian distribution
    neglogp = K.mean(_neglogp(action, mu, K.exp(log_sigma)))

    # Eligibility
    eligibility = K.mean(K.stop_gradient(advantage) * neglogp)

    # Entropy
    entropy = K.mean(_entropy(K.exp(log_sigma)))

    return eligibility - alpha * entropy


def value_loss_func(y_true, y_pred, beta=0.5):
    """Value loss function"""
    return beta * K.mean(K.square(y_true - y_pred))
