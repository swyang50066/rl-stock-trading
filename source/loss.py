import  numpy   as  np

import  tensorflow          as  tf
import  keras
import  keras.backend       as  K


def policy_loss_func(advantage, eps=1.e-16):
    ''' Policy loss function
    '''
    def _neglogp(action, mu, sigma):
        return (
            .5*K.sum(K.square((x - mu) / sigma), axis=-1)
            + .5*K.log(2.*np.pi*K.square(sigma) + eps, axis=-1)
        )

    def _entropy(self, sigma):
        return K.sum(K.log(sigma + eps) + .5*np.log(2.*np.pi*np.e), axis=-1)
    
    def _loss(y_true, y_pred, alpha=0.01):
        # Parse inputs
        mu, log_sigma = tf.split(y_pred, num_or_size_split=2, axis=-1)

        # Get negative logarithmic of Gaussian distribution
        neglogp = K.mean(_neglogp(y_true, mu, K.exp(sigma)))

        # Eligibility
        eligibility = K.mean(advantage * neglogp)

        # Entropy
        entropy = K.mean(_entropy(K.exp(sigma)))

        return eligibility - alpha*entropy

    return _loss


def value_loss_func(y_true, y_pred, beta=.5):
    ''' Value loss function
    '''
    return beta*K.mean(K.square(y_true - y_pred))
