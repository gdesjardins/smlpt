import numpy
import theano

from theano import tensor as T

def compute_log_z(rbm, max_bits=15):

    if rbm.n_visible < rbm.n_hidden:
        width = rbm.n_visible
        type = 'vis'
    else:
        width = rbm.n_hidden
        type = 'hid'

    # determine in how many steps to compute Z
    block_bits = width if (not max_bits or width < max_bits) else max_bits
    block_size = 2**block_bits

    # generate fixed sub-block
    if not hasattr(rbm, 'logz_data'):
        logz_data = numpy.zeros((block_size, width), order='C', dtype=theano.config.floatX)
        tensor_10D_idx = numpy.ndindex(*([2]*block_bits))
        for i, j in enumerate(tensor_10D_idx):
            logz_data[i, -block_bits:] = j
        rbm.logz_data = numpy.array(logz_data, order='F', dtype=theano.config.floatX)

    # define theano function to compute partial Z
    if not hasattr(rbm, 'compute_negFE'):
        mFE_max_in = T.dscalar('mFE_max_in')
        z_data = T.matrix('z_data')
        negFE = -rbm.free_energy(z_data, type=type)
        rbm.compute_negFE = theano.function([z_data], negFE)

    Z = 0

    # now loop over sub-portion of all visible/hidden configurations
    negFE = numpy.zeros(2**width, dtype=theano.config.floatX)

    for bi, upper_bits in enumerate(numpy.ndindex(*([2]*(width-block_bits)))):
        rbm.logz_data[:, :width-block_bits] = upper_bits
        negFE[bi*block_size:(bi+1)*block_size] = rbm.compute_negFE(rbm.logz_data)

    alpha = numpy.max(negFE)
    log_z = numpy.log(numpy.sum(numpy.exp(negFE - alpha))) + alpha

    return log_z


def compute_nll(free_energy_fn, data, log_z,
                bufsize=1000, preproc=None, dtype=theano.config.floatX):
    """
    param free_energy_fn: compiled theano function to compute free-energy
    param data: dataset on which to measure likelihood
    param log_z: estimate of the partition function
    param bufsize: compute likelihood in blocks of "bufsize" data examples
    param preproc: optional function to apply to data before measuring likelihood
    param dtype: default datatype to use for intermediate buffer.
    """

    i = 0.
    nll = 0
    while i < len(data):
        x = numpy.array(data[i:i+bufsize, :], dtype=dtype)
        if preproc:
            x = preproc(x)
        
        x_nll = numpy.sum(- free_energy_fn(x) - log_z)
        nll = (i*nll + x_nll) / (i + len(x))
        i += len(x)

    return nll 
