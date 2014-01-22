import theano
import theano.tensor as T
import numpy, time

from theano.tensor.shared_randomstreams import RandomStreams
from theano.tests import unittest_tools
from smlpt import sampler

def test_blockgibbs_RBM():
    v = theano.shared(numpy.random.rand(5,20), name='v')
    h = theano.shared(numpy.random.rand(5,10), name='h')

    W = theano.shared(numpy.zeros((20,10)), name='W')
    visb = theano.shared(numpy.ones(20)*0.1, name='visb')
    hidb = theano.shared(numpy.ones(10)*0.2, name='visb')

    ph1_v = T.nnet.sigmoid(T.dot(v,W) + hidb)
    pv1_h = T.nnet.sigmoid(T.dot(h,W.T) + visb)

    theano_rng = RandomStreams(unittest_tools.fetch_seed())

    h_sample = theano_rng.binomial(size=ph1_v.shape, n=1, p=ph1_v, dtype=theano.config.floatX)
    v_sample = theano_rng.binomial(size=pv1_h.shape, n=1, p=pv1_h, dtype=theano.config.floatX)

    s = sampler.BlockGibbsSampler({v: v_sample, h: h_sample}, n_steps=2)
    s.draw(10)
