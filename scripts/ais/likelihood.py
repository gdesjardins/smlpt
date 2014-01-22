import numpy
import pickle

import theano
import theano.tensor as T

from smlpt import rbm_tools
from smlpt import ais
from smlpt import tempered_rbm
from jobman import make

def fX(a): 
    return numpy.array(a,dtype=theano.config.floatX)

def experiment(state, channel):

    seed = state.get('seed', 23098)
    assert hasattr(state, 'model')
    assert hasattr(state, 'n_runs')
    dataset = make(state.dataset)
    large = state.get('large', False)

    model = dict(pickle.load(open(state.model,'r')))
    param_vals = [model['W'],model['vbias'],model['hbias']]

    if large:
        betas =  numpy.hstack((fX(numpy.linspace(0,0.5,1e4)),
                               fX(numpy.linspace(0.5,0.9,2e4)),
                               fX(numpy.linspace(0.9,1.0,4e4))))
    else:
        betas =  numpy.hstack((fX(numpy.linspace(0,0.5,1e3)),
                               fX(numpy.linspace(0.5,0.9,1e4)),
                               fX(numpy.linspace(0.9,1.0,1e4))))
    print 'len(betas) = ', len(betas)

    (logz, var_logz), aisobj = \
            ais.rbm_ais(param_vals, 
                        n_runs=state.n_runs, 
                        betas=betas, 
                        data=fX(dataset.train.x),
                        seed=seed)
  
    fe_input = T.dmatrix('fe_input')
    fe_output = tempered_rbm.free_energy(fe_input, params=param_vals)
    free_energy = theano.function([fe_input], fe_output)

    state.valid_nll = rbm_tools.compute_nll(free_energy, fX(dataset.valid.x), logz, 
                                      preproc=getattr(dataset, 'preproc', None),
                                      dtype=theano.config.floatX)
    state.test_nll = rbm_tools.compute_nll(free_energy, fX(dataset.test.x), logz, 
                                      preproc=getattr(dataset, 'preproc', None),
                                      dtype=theano.config.floatX)

    stddev = 3*numpy.sqrt(var_logz)

    print 'Z =  %f' % logz
    print 'Valid likelihood is: %.2f (+/- %.2f)' % (state.valid_nll, stddev)
    print 'Test likelihood is:  %.2f (+/- %.2f)' % (state.test_nll, stddev)

    return channel.COMPLETE
