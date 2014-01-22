import numpy
import cPickle

import theano
import theano.tensor as T
from theano.printing import Print

def sgd(grads, lr, momentum=0.0):

    updates = {}

    for param, gparam in grads.iteritems():
        x_t = param
        x_tm1 = theano.shared(numpy.zeros_like(x_t.value), name=param.name+'_old_value')
        # update old gradient, with new gradient
        x_tp1 = x_t + momentum * (x_t - x_tm1) - lr * gparam
        updates[x_tm1] = x_t
        updates[param] = x_tp1

    return updates

def adaptative_lr(lr, nupdates, type):

    updates = {}

    if type == 'constant': 
        pass
    elif type == 'linear_to_zero':
        t = theano.shared(0.0, name='lr_t')
        updates[lr] = lr * (nupdates - t) / nupdates
        updates[t] = t + 1

    return updates


class PolyakAveraging(object):

    params = property(lambda s: s.avg_params.values())
    param_vals  = property(lambda s: [p.value for p in s.params])
    shared_vals = property(lambda s: [(p.name,p.value) for p in s.avg_params.values()])

    def __init__(self, start=0):
       
        self.t = theano.shared(0.0, name='polyak_t')
        self.start = start
        self.avg_params = {}

    def config_updates(self, model_params, lr):

        updates = {}

        self.alpha_sum = theano.shared(numpy.asarray(0.0, dtype=theano.config.floatX))

        for param in model_params:
            avg_param = theano.shared(param.value, name=param.name)
      
            if self.start == -1:
                updates[avg_param] = param
            else:
                # once polyak averaging is engaged, perform moving average of parameter values
                # prior to self.start, simply copy parameters
                updates[avg_param] = T.switch(self.t >= self.start,
                        self.alpha_sum*avg_param + lr*param, param)
                updates[self.alpha_sum] = T.switch(self.t >= self.start,
                        self.alpha_sum + lr, self.alpha_sum)

            self.avg_params[param.name] = avg_param

        return updates

    def save(self, fname='polyak_model.pkl'):
        fp = open(fname, 'w')
        cPickle.dump(self.shared_vals, fp, protocol=cPickle.HIGHEST_PROTOCOL)
        fp.close()
