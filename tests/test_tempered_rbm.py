import numpy, time

import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from smlpt import tempered_rbm, rbm_tools

def load_data(p=0.001, size=10000, seed=238904):
    from pylearn.datasets import peaked_modes
    dataset = peaked_modes.neal94_AC(p=p, size=size, seed=seed)
    dataset.train.x = numpy.asarray(dataset.train.x, dtype=theano.config.floatX)
    dataset.train.x = theano.shared(dataset.train.x)
    return dataset

def notest_rbm(learning_rate=0.1, nupdates = 20000,
             batch_size = 1, n_hidden = 10, n_beta=10):
    """
    Demonstrate how to train and afterwards sample from it using Theano.
    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM 
    :param nupdates: number of parameter updates
    :param dataset: path the the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_samples: number of samples to plot for each chain
    """
    dataset = load_data()

    # compute number of minibatches for training, validation and testing
    n_train_batches = dataset.train.x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    # construct the RBM class
    rbm = tempered_rbm.RBM(input = x, 
              n_visible = numpy.prod(dataset.img_shape),
              n_hidden = n_hidden, 
              n_beta = n_beta,
              batch_size = batch_size,
              numpy_rng = rng,
              theano_rng = theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    updates = rbm.get_updates(lr=learning_rate, k=2)

    #################################
    #     Training the RBM          #
    #################################
    
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index], [],
           updates = updates, 
           givens = {x: dataset.train.x[index*batch_size:(index+1)*batch_size]})

    plotting_time = 0.
    start_time = time.clock()

    # go through training epochs 
    batch_index = 0
    for n in xrange(nupdates):
        train_rbm(batch_index)
        batch_index = (batch_index + 1) % n_train_batches
        if (n+1)%1000 == 0: 
            print '@ update: ', n+1

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print 'Training took %f s' % (end_time - start_time)

def test_partition_function():

    def subtest(nv, nh):
        r = tempered_rbm.RBM(n_visible=nv, n_hidden=nh)
        r.vbias.value = numpy.random.random(r.vbias.value.shape)
        r.hbias.value = numpy.random.random(r.hbias.value.shape)
        r.W.value = numpy.random.random(r.W.value.shape)

        t1 = time.time()
        logZ = rbm_tools.compute_log_Z(r, max_bits=15)
        print 'logZ = ', logZ
        print 'Elapsed time: ', time.time() - t1

        from plearn.pyext import pl as plearn
        rbm_pl = plearn.RBMModule(
                    visible_layer = plearn.RBMBinomialLayer(size = nv),
                    hidden_layer  = plearn.RBMBinomialLayer(size = nh),
                    connection    = plearn.RBMMatrixConnection(down_size=nv,up_size=nh))
        rbm_pl.connection.weights = r.W.value.T.copy()
        rbm_pl.visible_layer.bias = r.vbias.value.copy()
        rbm_pl.hidden_layer.bias = r.hbias.value.copy()

        t1 = time.time()
        rbm_pl.computePartitionFunction()
        print 'PLearn logZ = ', rbm_pl.log_partition_function
        print 'Elapsed time: ', time.time() - t1

        assert numpy.abs(logZ - rbm_pl.log_partition_function) < 1e-2

    subtest(784,15)
    subtest(15,784)
