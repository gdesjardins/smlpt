import numpy
import time
import copy
from scipy import io

import theano
import theano.tensor as T

from smlpt import tempered_rbm, optimization, rbm_tools, ais
from pylearn.datasets import MNIST


def load_rbm(fname):
    mnistvh = io.loadmat(fname)
    (W, vbias, hbias) = mnistvh['vishid'],mnistvh['visbiases'], mnistvh['hidbiases']
    
    rbm = tempered_rbm.RBM(n_visible=W.shape[0], n_hidden=W.shape[1])
    rbm.W.value = numpy.array(W, dtype=theano.config.floatX)
    rbm.vbias.value = numpy.array(vbias[0], dtype=theano.config.floatX)
    rbm.hbias.value = numpy.array(hbias[0], dtype=theano.config.floatX)

    return rbm

def ais_nodata(fname, do_exact=True):

    rbm = load_rbm(fname)
   
    # ais estimate using tempered models as intermediate distributions
    t1 = time.time()
    (logz, log_var_dz), aisobj = ais.rbm_ais(rbm.param_vals, n_runs=100, seed=123)
    print 'AIS logZ         : %f' % logz
    print '    log_variance : %f' % log_var_dz
    print 'Elapsed time: ', time.time() - t1

    if do_exact: 
        # exact log-likelihood
        exact_logz = rbm_tools.compute_log_z(rbm)
        print 'Exact logZ = %f' % exact_logz
        numpy.testing.assert_almost_equal(exact_logz, logz, decimal=0)

def ais_data(fname, do_exact=True):

    rbm = load_rbm(fname)
  

    # load data to set visible biases to ML solution
    from pylearn.datasets import MNIST
    dataset = MNIST.train_valid_test()
    data = numpy.array(dataset.train.x, dtype=theano.config.floatX)

    # run ais using B=0 model with ML visible biases
    t1 = time.time()
    (logz, log_var_dz), aisobj = ais.rbm_ais(rbm.param_vals, n_runs=100, seed=123, data=data)
    print 'AIS logZ         : %f' % logz
    print '    log_variance : %f' % log_var_dz
    print 'Elapsed time: ', time.time() - t1

    if do_exact: 
        # exact log-likelihood
        exact_logz = rbm_tools.compute_log_z(rbm)
        print 'Exact logZ = %f' % exact_logz

        numpy.testing.assert_almost_equal(exact_logz, logz, decimal=0)

def test_ais():
    ais_data('mnistvh.mat')
    #ais_nodata('mnistvh.mat')
    #ais_nodata('mnistvh_500.mat', do_exact=False)

if __name__ == '__main__':
    test_ais()
