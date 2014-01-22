import numpy
import theano

from smlpt import tempered_rbm
from smlpt import ais
from smlpt import optimization
from pylearn.datasets import caltech

def test_polyak_off():
    data = caltech.caltech_silhouette()

    rbm = tempered_rbm.RBM(n_visible=784, n_hidden=500, n_beta=1, t_batch_size=0, seed=1)
    avg_rbm = tempered_rbm.RBM(n_visible=784, n_hidden=500, n_beta=1, t_batch_size=0, seed=1)

    lr = theano.shared(0.001, name='lr')
    momentum = theano.shared(0.5, name='momentum')

    # updates without averaging
    learning_grads, sampling_updates = rbm.get_learning_gradients(k=1, l1=0, l2=0)
    learning_updates = optimization.sgd(learning_grads, lr)
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    train_rbm = theano.function([rbm.input], [], updates = updates)

    # updates with averaging
    learning_grads, sampling_updates = avg_rbm.get_learning_gradients(k=1, l1=0, l2=0)
    learning_updates = optimization.sgd(learning_grads, lr)
    polyak = optimization.PolyakAveraging(avg_rbm.params, start=-1)
    polyak_updates = polyak.get_updates(learning_updates)
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    updates.update(polyak_updates)
    train_avg_rbm = theano.function([avg_rbm.input], [], updates = updates)

    W = numpy.zeros_like(rbm.W.value)
    vbias = numpy.zeros_like(rbm.vbias.value)
    hbias = numpy.zeros_like(rbm.hbias.value)

    for i in range(10):

       x = data.train.x[i:i+1,:] * 1.0
       train_rbm(x)
       train_avg_rbm(x)

    # check that polyak.param_dict, rbm, avg_rbm all have same values
    numpy.testing.assert_array_almost_equal(rbm.W.value, avg_rbm.W.value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.vbias.value, avg_rbm.vbias.value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.hbias.value, avg_rbm.hbias.value, decimal=4)

    # check that avg_rbm and rbm have the same parameters
    numpy.testing.assert_array_almost_equal(rbm.W.value, polyak.param_dict[avg_rbm.W].value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.vbias.value, polyak.param_dict[avg_rbm.vbias].value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.hbias.value, polyak.param_dict[avg_rbm.hbias].value, decimal=4)


def test_polyak_on():
    data = caltech.caltech_silhouette()

    rbm = tempered_rbm.RBM(n_visible=784, n_hidden=500, n_beta=1, t_batch_size=0, seed=1)
    avg_rbm = tempered_rbm.RBM(n_visible=784, n_hidden=500, n_beta=1, t_batch_size=0, seed=1)

    lr = theano.shared(0.001, name='lr')
    momentum = theano.shared(0.5, name='momentum')

    # updates without averaging
    learning_grads, sampling_updates = rbm.get_learning_gradients(k=1, l1=0, l2=0)
    learning_updates = optimization.sgd(learning_grads, lr)
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    train_rbm = theano.function([rbm.input], [], updates = updates)

    # updates with averaging
    learning_grads, sampling_updates = avg_rbm.get_learning_gradients(k=1, l1=0, l2=0)
    learning_updates = optimization.sgd(learning_grads, lr)
    polyak = optimization.PolyakAveraging(avg_rbm.params, start=0)
    polyak_updates = polyak.get_updates(learning_updates)
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    updates.update(polyak_updates)
    train_avg_rbm = theano.function([avg_rbm.input], [], updates = updates)

    W = numpy.zeros_like(rbm.W.value)
    vbias = numpy.zeros_like(rbm.vbias.value)
    hbias = numpy.zeros_like(rbm.hbias.value)

    for i in range(10):

       x = data.train.x[i:i+1,:] * 1.0

       train_rbm(x)
       W += rbm.W.value
       vbias += rbm.vbias.value
       hbias += rbm.hbias.value

       train_avg_rbm(x)


    W /= 10.0
    vbias /= 10.0
    hbias /= 10.0

    # check that polyak.param_dict contains the correct "averaged" parameter values
    numpy.testing.assert_array_almost_equal(rbm.W.value, avg_rbm.W.value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.vbias.value, avg_rbm.vbias.value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.hbias.value, avg_rbm.hbias.value, decimal=4)

    # check that avg_rbm and rbm have the same parameters
    numpy.testing.assert_array_almost_equal(polyak.param_dict[avg_rbm.W].value, W, decimal=4)
    numpy.testing.assert_array_almost_equal(polyak.param_dict[avg_rbm.vbias].value, vbias, decimal=4)
    numpy.testing.assert_array_almost_equal(polyak.param_dict[avg_rbm.hbias].value, hbias, decimal=4)

def test_polyak_delayed():
    data = caltech.caltech_silhouette()

    rbm = tempered_rbm.RBM(n_visible=784, n_hidden=500, n_beta=1, t_batch_size=0, seed=1)
    avg_rbm = tempered_rbm.RBM(n_visible=784, n_hidden=500, n_beta=1, t_batch_size=0, seed=1)

    lr = theano.shared(0.001, name='lr')
    momentum = theano.shared(0.5, name='momentum')

    # updates without averaging
    learning_grads, sampling_updates = rbm.get_learning_gradients(k=1, l1=0, l2=0)
    learning_updates = optimization.sgd(learning_grads, lr)
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    train_rbm = theano.function([rbm.input], [], updates = updates)

    # updates with averaging
    learning_grads, sampling_updates = avg_rbm.get_learning_gradients(k=1, l1=0, l2=0)
    learning_updates = optimization.sgd(learning_grads, lr)
    polyak = optimization.PolyakAveraging(avg_rbm.params, start=10)
    polyak_updates = polyak.get_updates(learning_updates)
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    updates.update(polyak_updates)
    train_avg_rbm = theano.function([avg_rbm.input], [], updates = updates)

    W = numpy.zeros_like(rbm.W.value)
    vbias = numpy.zeros_like(rbm.vbias.value)
    hbias = numpy.zeros_like(rbm.hbias.value)

    n = 0
    for i in range(20):

       x = data.train.x[i:i+1,:] * 1.0

       train_rbm(x)
       if i >= 10:
           W += rbm.W.value
           vbias += rbm.vbias.value
           hbias += rbm.hbias.value
           n += 1

       train_avg_rbm(x)

    W /= n
    vbias /= n
    hbias /= n

    # check that polyak.param_dict contains the correct "averaged" parameter values
    numpy.testing.assert_array_almost_equal(rbm.W.value, avg_rbm.W.value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.vbias.value, avg_rbm.vbias.value, decimal=4)
    numpy.testing.assert_array_almost_equal(rbm.hbias.value, avg_rbm.hbias.value, decimal=4)

    # check that avg_rbm and rbm have the same parameters
    numpy.testing.assert_array_almost_equal(polyak.param_dict[avg_rbm.W].value, W, decimal=4)
    numpy.testing.assert_array_almost_equal(polyak.param_dict[avg_rbm.vbias].value, vbias, decimal=4)
    numpy.testing.assert_array_almost_equal(polyak.param_dict[avg_rbm.hbias].value, hbias, decimal=4)
