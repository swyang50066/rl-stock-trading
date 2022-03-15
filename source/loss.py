import  keras
import  keras.backend       as  K


def policy_loss_function(value_true, value_pred, eps=1.e-16):
    ''' Policy loss function
    '''
    advantage = value_true - value_pred
    def _loss(y_true, y_pred):
        # Eligibility
        eligibility = K.sum(y_pred*y_true, axis=1, keepdims=True)
        eligibility = K.sum(-K.log(eligibility) + eps)
        eligibility *= K.stop_gradient(advantage))

        # Entropy
        entropy = K.sum(y_pred*K.log(y_pred + eps), axis=1)

        return alpha*entropy - eligibility

    return _loss


def value_loss_function(y_true, y_pred):
    ''' Value loss function
    '''
    return K.mean(.5*K.square(y_true - y_pred))
