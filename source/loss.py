import  keras
import  keras.backend       as  K


def policy_loss_function(advantage, eps=1.e-16):
    ''' Policy loss function
    '''
    def _loss(y_true, y_pred, alpha=0.01):
        # Eligibility 
        eligibility = -K.log(K.sqrt(2*np.pi)*sigma) - (action - mu)**2/ (2*sigma*sigma)
        eligibility *= adventage

        # Entropy
        entropy = -.5*K.log(2.*np.pi*np.e*sigma)

        torch::Tensor loss_tot = loss + entropy_loss ;

        return eligibility + alpha*entropy

    return _loss


def value_loss_function(y_true, y_pred, beta=.5):
    ''' Value loss function
    '''
    return beta*K.mean(K.square(y_true - y_pred))
