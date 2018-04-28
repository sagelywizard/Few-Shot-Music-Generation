import numpy as np


class AbstractModel(object):

    def __init__(self, config):
        self._config = config

    @property
    def name(self):
        return self._config['name']

    def train(self, episode):
        """Train model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        raise NotImplementedError()

    def eval(self, episode):
        """Evaluate model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        raise NotImplementedError()

    def sample(self, support_set, num):
        """Sample a sequence of size num conditioned on support_set.

        Args:
            support_set (numpy array): support set to condition the sample.
            num: size of sequence to sample.
        """
        raise NotImplementedError()

    def save(self, checkpt_path):
        """Save model's current parameters at checkpt_path.

        Args:
            checkpt_path (string): path where to save parameters.
        """
        raise NotImplementedError()

    def recover_or_init(self, init_path):
        """Recover or initialize model based on init_path.

        If init_path has appropriate model parameters, load them; otherwise,
        initialize parameters randomly.
        Args:
            init_path (string): path from where to load parameters.
        """
        raise NotImplementedError()


def flatten(set_):
    """Convert shape from [B,S,N] => [BxS,N]."""
    shape = set_.shape
    return np.reshape(set_, (shape[0] * shape[1], shape[2]))


def convert_set_to_input_and_target(set_, start_word=None):
    """Convert _set to input and target to use for model for sequence generation.

    If start_word is given, add to start of _set.
    Input is _set without last item; Target is _set without first item
    """
    X = flatten(set_)

    if start_word is None:
        Y = np.copy(X[:, 1:])
        X_new = X[:, :-1]
    else:
        Y = np.copy(X)
        start_word_column = np.full(
            shape=[np.shape(X)[0], 1], fill_value=start_word)
        X_new = np.concatenate([start_word_column, X[:, :-1]], axis=1)

    return X_new, Y
