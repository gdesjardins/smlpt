"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs 
to those without visible-visible and hidden-hidden connections. 
"""
import os
import copy
import numpy
import time
import cPickle
import gzip
import PIL.Image
from scipy import stats

import theano
import theano.tensor as T
from theano import gof
from theano.compile import optdb
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor import nnet
from theano.gof.destroyhandler import fast_inplace_check

from theano.sandbox.cuda import cuda_available, cuda_enabled
if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType, float32_shared_constructor

# Labels which are attached to a given particle 
LBL_UP   = 1          # defines "up-moving" particles (from B=1 to B=0)
LBL_DOWN = 0          # defines "down-moving" particles (from B=0 to B=1)
LBL_NONE = numpy.nan  # particles which haven't been initialized yet

def rtime_deo(p_acc, n): 
    return ((1-p_acc)*2*n*(n-1)) / p_acc

sharedX = lambda X, name :\
    theano.shared(numpy.asarray(X, dtype=theano.config.floatX), name=name)

class RBM(object):
    """Restricted Boltzmann Machine (RBM) trained with SML-PT"""

    params = property(lambda s: [s.W, s.vbias, s.hbias])
    param_vals = property(lambda s: [p.value for p in s.params])
    
    shared_vars = property(lambda s: \
            [s.W, s.hbias, s.vbias, 
             s._buffer, s.mf_buffer, s.n_beta, s.n_chain, 
             s._E, s._beta, s._mixstat, s.labels, s._fup_target,
             s._nup, s._ndown, s.rtime, s.avg_rtime, s._swapstat])
    shared_vals = property(lambda s: [(p.name,p.value) for p in s.shared_vars])

    def __init__(self, input=None, n_visible=784, n_hidden=500, \
                 W=None, hbias=None, vbias=None, 
                 seed = None, theano_rng=None,
                 batch_size=0, t_batch_size=1, 
                 n_beta=10, beta_lbound=0., tau=None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
         part of a larger graph.
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param W: None for standalone RBMs or symbolic variable pointing to a
         shared weight matrix in case RBM is part of a DBN network; in a DBN,
         the weights are shared between RBMs and layers of a MLP
        :param hbias: None for standalone RBMs or symbolic variable pointing 
         to a shared hidden units bias vector in case RBM is part of a 
         different network
        :param vbias: None for standalone RBMs or a symbolic variable 
         pointing to a shared visible units bias
        :param tau: optional fixed time constant (overrides return time)
        """
        assert (n_beta > 1 and t_batch_size > 0) or (n_beta==1 and t_batch_size==0)
        if t_batch_size > 0: assert batch_size%t_batch_size==0

        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.t_batch_size = t_batch_size  # size of tempered minibatch
        self.batch_size = batch_size # size of T=1 minibatch
  
        # deal with random number generation
        if seed is None:
            rng = numpy.random.RandomState(123)
        else:
            rng = numpy.random.RandomState(seed)
        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2**30))
        self.rng = rng
        self.theano_rng = theano_rng

        if W is None : 
           # W is initialized with `initial_W` which is uniformely sampled
           # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
           # the output of uniform if converted using asarray to dtype 
           # theano.config.floatX so that the code is runable on GPU
           initial_W = 0.01 * self.rng.randn(n_visible, n_hidden)
           # theano shared variables for weights and biases
           W = sharedX(initial_W, 'W')
        self.W = W

        if hbias is None :
           # create shared variable for hidden units bias
           hbias = sharedX(numpy.zeros(n_hidden), 'hbias')
        self.hbias = hbias

        if vbias is None :
           # create shared variable for visible units bias
           vbias = sharedX(numpy.zeros(n_visible), 'vbias')
        self.vbias = vbias

        # initialize input layer for standalone RBM or layer0 of DBN
        if input is None:
            input = T.matrix('input')
        self.input = input 

        #########################################################################
        # Fields indexed by batch_size + mixstat:    buffer, E
        # Fields indexed by mixstat:    beta, labels, rtime
        # Fields indexed by temp index: mixstat, fup_target, nup, ndown, swapstat
        #########################################################################

        ### initialize tempering stuff ###
        n_chain = t_batch_size * n_beta
        self.n_chain = theano.shared(n_chain, name='n_chain') # number of active chains in buffer array
        self.n_beta  = theano.shared(n_beta, name='n_beta')   # number of temperatures in system
        self.n_chain_total = batch_size + self.n_chain

        # configure buffers for negative particles
        _buffer = self.rng.randint(0,2,size=(batch_size + 2*n_chain, n_visible))
        self._buffer   = sharedX(_buffer, name='buffer')
        self.buffer    = self._buffer[:self.n_chain_total]
        # buffer used to store mean-field activation
        self.mf_buffer = sharedX(numpy.zeros_like(_buffer), name='mf_buffer')

        # vectors containing energy of current negative particles (at T=1)
        self._E = sharedX(numpy.zeros(batch_size + 2*n_chain), name='E')
        self.E  = self._E[:self.n_chain_total]

        # Space out inverse temperature parameters linearly in [1,beta_lbound] range .
        beta = numpy.zeros(2*n_chain)
        for bi in range(t_batch_size):
            base_idx = n_beta*bi
            beta[base_idx:base_idx+n_beta] = numpy.linspace(1, beta_lbound, n_beta)
        self._beta = sharedX(beta, name='beta')
        self.beta = self._beta[:self.n_chain]

        # Used to multiply the rows of "W x + b"
        self.beta_matrix = T.vertical_stack(
                T.alloc(1.0, batch_size, 1),
                self.beta.dimshuffle([0,'x']))

        # initialize data structure to map nhid/nvis rows to a given temperature
        # mixstat stores pointers to self.nvis array
        mixstat = numpy.zeros((t_batch_size, 2*n_beta), dtype='int32')
        mixstat[:, :n_beta] = numpy.arange(n_chain).reshape(t_batch_size, n_beta)
        self._mixstat = theano.shared(mixstat, name='mixstat')
        self.mixstat = self._mixstat[:, :self.n_beta]

        ### Initialize particle properties ###

        # labels: 1 means going up in temperature, 0 going down in temperature
        labels = LBL_NONE * numpy.ones(2*n_chain, dtype='int32')
        labels[mixstat[:,0]] = LBL_UP
        self.labels = theano.shared(labels, name='labels') 

        # return time
        rtime = numpy.zeros(2*n_chain, dtype='int32')
        self.rtime = theano.shared(rtime, name='rtime') 
        self.avg_rtime = sharedX(rtime_deo(0.4,n_beta), name='avg_rtime')

        ### Initialize temperature properties ###

        # configure fup target for each chain (this shouldn't change very often)
        _fup_target = numpy.zeros(2*n_beta)
        _fup_target[:n_beta] = numpy.linspace(1,0,n_beta)
        self._fup_target = sharedX(_fup_target, name='fup_target')
        self.fup_target = self._fup_target[:self.n_beta]

        # configure histogram of up moving particles
        _nup = numpy.zeros(2*n_beta)
        _nup[:n_beta] = numpy.linspace(1,0,n_beta)
        self._nup = sharedX(_nup, name='nup')
        self.nup = self._nup[:self.n_beta]
        
        # configure histogram of down moving particles
        _ndown = numpy.zeros(2*n_beta)
        _ndown[:n_beta] = numpy.linspace(0,1,n_beta)
        self._ndown = sharedX(_ndown, name='ndown')
        self.ndown = self._ndown[:self.n_beta]

        # use return time as the time constant for all moving averages
        if not tau:
            self.tau = 1./self.avg_rtime
        else:
            self.tau = T.as_tensor(tau)
        self.get_tau = theano.function([], self.tau)

        # create PT Op
        self._swapstat = sharedX(numpy.zeros(2*n_beta), name='swapstat')
        self.swapstat = self._swapstat[:self.n_beta]

        self.pt_swaps = PT_Swaps(rng=self.rng)
        self.pt_swap_t1_sample = PT_SwapT1Sample(rng=self.rng, batch_size=self.batch_size)

    def energy(self, v_sample, h_sample):
        ''' Function to compute E(v,h) '''
        E_w = T.sum(T.dot(v_sample, self.W)*h_sample, axis=1)
        E_vbias = T.dot(v_sample, self.vbias)
        E_hbias = T.dot(h_sample, self.hbias)
        return - E_w - E_vbias - E_hbias

    def free_energy(self, sample, type='vis'):
        ''' Function to compute the free energy '''
        assert type in ('vis','hid')
  
        if type is 'vis':
            wx_b = T.dot(sample, self.W) + self.hbias
            bias_term = T.dot(sample, self.vbias)
        else:
            wx_b = T.dot(sample, self.W.T) + self.vbias
            bias_term = T.dot(sample, self.hbias)

        hidden_term = T.sum(nnet.softplus(wx_b),axis = 1)
        #hidden_term = T.sum(T.log(1 + T.exp(wx_b)),axis = 1)
        return -hidden_term - bias_term

    def propup(self, vis, non_linear=True, temper=False):
        ''' This function propagates the visible units activation upwards to
        the hidden units 
        '''
        activation = T.dot(vis, self.W) + self.hbias
        if temper:
            activation *= self.beta_matrix
        return T.nnet.sigmoid(activation) if non_linear else activation

    def sample_h_given_v(self, v0_sample, temper=False):
        ''' This function infers state of hidden units given visible units '''

        # compute the activation of the hidden units given a sample of the visibles
        h1_mean = self.propup(v0_sample, temper=temper)

        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype 
        # int64 by default. If we want to keep our computations in floatX 
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size = h1_mean.shape, n = 1, p = h1_mean,
                dtype = theano.config.floatX)
        return [h1_mean, h1_sample]

    def propdown(self, hid, non_linear=True, temper=False):
        '''This function propagates the hidden units activation downwards to
        the visible units
        '''
        activation = T.dot(hid, self.W.T) + self.vbias
        if temper:
            activation *= self.beta_matrix
        return T.nnet.sigmoid(activation) if non_linear else activation

    def sample_v_given_h(self, h0_sample, temper=False):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = self.propdown(h0_sample, temper=temper)

        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype 
        # int64 by default. If we want to keep our computations in floatX 
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size = v1_mean.shape,n = 1,p = v1_mean,
                dtype = theano.config.floatX)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, temper=False):
        ''' This function implements one step of Gibbs sampling, 
            starting from the hidden state'''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample, temper=temper)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample, temper=temper)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, temper=False):
        ''' This function implements one step of Gibbs sampling, 
            starting from the visible state'''
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample, temper=temper)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample, temper=temper)
        return [h1_mean, h1_sample, v1_mean, v1_sample]
 
    def pt_step(self, v0_sample, beta, mixstat, labels, swapstat, 
                rtime, avg_rtime, nup, ndown):

        # perform Gibbs steps for all particles
        h1_mean, h1_sample, v1_mean, v1_sample = self.gibbs_vhv(v0_sample, temper=True) 
        E = self.energy(v1_sample, h1_sample)

        if self.n_beta.value > 1:

            # propose swap between chains (k,k+1) where k is odd
            beta, mixstat, labels, swapstat, rtime, avg_rtime = \
                    self.pt_swaps(beta, mixstat, E[self.batch_size:], 
                                  labels, swapstat,
                                  rtime, avg_rtime, self.tau, offset=0)

            # update labels and histograms
            nup, ndown = pt_update_histogram(mixstat, labels, nup, ndown, self.tau)
            
            # propose swap between chains (k,k+1) where k is even
            beta, mixstat, labels, swapstat, rtime, avg_rtime = \
                    self.pt_swaps(beta, mixstat, E[self.batch_size:],
                                  labels, swapstat,
                                  rtime, avg_rtime, self.tau, offset=1)
            
            # update labels and histograms
            nup, ndown = pt_update_histogram(mixstat, labels, nup, ndown, self.tau)

        return [h1_sample, v1_sample, v1_mean, 
                beta, mixstat, E, labels, swapstat, 
                rtime, avg_rtime, nup, ndown]
 
    def get_sampling_updates(self, k=1):
        """ 
        This functions implements one step of PCD-k
        :param k: number of Gibbs steps to do in PCD-k
        """
        # perform actual negative phase
        [nh_samples, nv_samples, nv_mean, beta, mixstat, E, labels, 
         swapstat, rtime, avg_rtime, nup, ndown], updates = \
            theano.scan(self.pt_step, 
                        outputs_info = \
                            [{'initial': None,          'return_steps': 1}, # h1_sample
                             {'initial': self.buffer,   'return_steps': 1}, # v1_sample
                             {'initial': None,          'return_steps': 1}, # v1_mean 
                             {'initial': self.beta,     'return_steps': 1}, # beta
                             {'initial': self.mixstat,  'return_steps': 1}, # mixstat
                             {'initial': None,          'return_steps': 1}, # E
                             {'initial': self.labels,   'return_steps': 1}, # labels
                             {'initial': self.swapstat, 'return_steps': 1}, # swapstat
                             {'initial': self.rtime,    'return_steps': 1}, # rtime
                             {'initial': self.avg_rtime,'return_steps': 1}, # avg_rtime
                             {'initial': self.nup,      'return_steps': 1}, # nup
                             {'initial': self.ndown,    'return_steps': 1}],# ndown
                        n_steps = k)

        updates = {}

        # update particle states
        updates[self._buffer] = T.set_subtensor(self._buffer[:self.n_chain_total], nv_samples)
        updates[self.mf_buffer] = T.set_subtensor(self.mf_buffer[:self.n_chain_total], nv_mean)
    
        # update energy of each particle
        updates[self._E] = T.set_subtensor(self._E[:self.n_chain_total], E)

        # update particle<->temperature mappings
        updates[self._mixstat] = T.set_subtensor(self._mixstat[:,:self.n_beta], mixstat)
        updates[self._beta] = T.set_subtensor(self._beta[:self.n_chain], beta)

        # updates for beta adaptation
        updates[self.labels] = labels
        updates[self.rtime] = rtime
        updates[self.avg_rtime] = avg_rtime
        updates[self._nup] = T.set_subtensor(self._nup[:self.n_beta], nup)
        updates[self._ndown] = T.set_subtensor(self._ndown[:self.n_beta], ndown)

        # updates for chain spawning
        updates[self._swapstat] = T.set_subtensor(self._swapstat[:self.n_beta], swapstat)

        return [nh_samples, nv_samples, beta, mixstat, E], updates
       
    def get_learning_gradients(self, k=1, 
            l1=0.0, l2=0.0, 
            sparse_lambda=0., sparse_p=0.01, 
            waste_cutoff=1e-2,
            waste_reduction=False):
        """ 
        This functions implements one step of CD-k or PCD-k
        Returns a proxy for the cost and the updates dictionary. The dictionary contains the
        update rules for weights and biases but also an update of the shared variable used to
        store the persistent chain, if one is used.
        :param k: number of Gibbs steps to do in CD-k/PCD-k
        :param l1: weight of L1 regularization term
        :param l2: weight of L2 regularization term
        :param sparse_lambda: weight of sparsity term (see Lee07)
        :param sparse_p: target activation probability
        :param waste_cutoff: when using waste reduction, if weight is smaller than this value,
               do not bother including it in the gradient (saves on wasteful computations).
        """
        updates = {}
        self.waste_reduction_nbsize = sharedX(1.0, name='waste_reduction_nbsize')

        print '*********************************'
        print '*********************************'
        print '**  waste_reduction = ', waste_reduction
        print '*********************************'
        print '*********************************'
 
        ### compute positive phase ###
        ph_mean, ph_sample = self.sample_h_given_v(self.input, temper=False)

        ### compute negative phase ###
        [nh_samples, nv_samples, beta, mixstat, E], sampling_updates = self.get_sampling_updates(k=k)
        updates.update(sampling_updates)

        if self.n_beta.value > 1:
            # after all this... swap bottom temperature samples into T=1 minibatch
            nv_samples, E = self.pt_swap_t1_sample(nv_samples, E, mixstat)
            updates[self._buffer] = T.set_subtensor(self._buffer[:self.n_chain_total], nv_samples)
            updates[self._E] = T.set_subtensor(self._E[:self.n_chain_total], E)

        # (optionally) perform waste reduction on parallel chains
        if waste_reduction:
            chain_end, weights, wr_nbsize = \
                    pt_waste_reduction(nv_samples, beta, mixstat, E, 
                                       cut_off=waste_cutoff, batch_size=self.batch_size)
            updates[self.waste_reduction_nbsize] = wr_nbsize

            # define cost function
            cost = T.mean(self.free_energy(self.input)) - \
                   T.sum(weights*self.free_energy(chain_end)) / self.t_batch_size
            gconstant = [chain_end, weights]
        else:
            chain_end = nv_samples[:self.batch_size]
            # define cost function
            cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
            gconstant = [chain_end]

        if l1: cost += l1 * T.sum(abs(self.W))
        if l2: cost += l2 * T.sum(self.W**2)
            
        # We must not compute the gradient through the gibbs sampling 
        gparams = T.grad(cost, self.params, consider_constant=gconstant)

        grads = {}
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            grads[param] = gparam
        
        # modify hidden biases according to the sparsity regularization term of Lee07
        if sparse_lambda:
            sparse_cost = sparse_lambda * T.sum((sparse_p - T.mean(ph_mean, axis=0))**2)
            grads[self.hbias] += T.grad(sparse_cost, self.hbias)

        return grads, updates


    def config_beta_updates(self):
        if self.n_beta.value <= 1: return

        beta_lr = T.scalar('beta_lr')

        # determine fraction of up-moving particles at each temperature
        fup = self.nup / (self.nup + self.ndown)
        self.get_fup = theano.function([], fup)

        # get an ordered-list of the current temperature set
        betas = pt_unmix(self.beta, self.mixstat, batch_index=0)
        self.get_betas = theano.function([], betas)

        # find optimal temperature values
        optimal_betas = pt_get_optimal_betas(fup, self.fup_target, betas)
        self.get_optimal_betas = theano.function([], optimal_betas)

        d_beta = beta_lr * (optimal_betas[1:-1] - betas[1:-1])

        ## Gradient-based beta update
        new_grad_beta = T.inc_subtensor(betas[1:-1], d_beta)

        # remix according to mixstat matrix: update mixstat in case of overshoot
        grad_mixed_beta, mixstat, fup_target, nup, ndown = \
                pt_update_beta_vector(new_grad_beta, self.mixstat, 
                                      self.fup_target, self.nup, self.ndown)

        updates = {}
        updates[self._beta] = T.set_subtensor(self._beta[:self.n_chain], grad_mixed_beta)
        updates[self._mixstat] = T.set_subtensor(self._mixstat[:,:self.n_beta], mixstat)
        updates[self._fup_target] = T.set_subtensor(self._fup_target[:self.n_beta], fup_target)
        updates[self._nup] = T.set_subtensor(self._nup[:self.n_beta], nup)
        updates[self._ndown] = T.set_subtensor(self._ndown[:self.n_beta], ndown)
        self.grad_update_betas = theano.function([beta_lr], new_grad_beta, updates=updates)


    def increase_storage(self):
        """
        Attempts to double the allocated memory, such that chains can continue to be spawned.
        :rval: True if memory allocation succeeded, False otherwise
        """
        if self.n_beta.value > 100:
            print '************ ERROR: FAILED TO ALLOCATE NEW MEMORY *******************'
            return False
        
        n = 2*len(self._buffer.value)

        def double_shared(var):
            shp = numpy.array(var.value.shape)
            shp[0] *= 2
            new_var = numpy.zeros(shp, dtype=var.value.dtype)
            new_var[:len(var.value)] = var.value
            var.value = new_var

        map(double_shared, 
            [self._buffer, self.mf_buffer, self._E, self._beta, self._swapstat, 
             self._nup, self._ndown, self._fup_target, self.labels, self.rtime])

        n = len(self._mixstat.value[0])
        _mixstat = numpy.zeros((self.t_batch_size, 2*n), dtype='int32')
        _mixstat[:,:n] = self._mixstat.value
        self._mixstat.value = _mixstat

        return True
  
    def check_spawn(self, min_p_acc=0.4, spawn_type='spawn_rtime', spawn_rtime_multiplier=1):
        """Returns False if no spawning required, if not index of new temperature."""
        if spawn_type == 'spawn_min':
            return (self._swapstat.value[:self.n_beta.value-1] < min_p_acc).any()
        elif spawn_type == 'spawn_avg':
            return numpy.mean(self._swapstat.value[:self.n_beta.value-1]) < min_p_acc
        elif spawn_type == 'spawn_rtime':
            return spawn_rtime_multiplier * self.avg_rtime.value > rtime_deo(min_p_acc, self.n_beta.value)
        elif spawn_type == 'spawn_auc':
            beta = self._beta.value[self._mixstat.value[0,:self.n_beta.value]]
            dbeta = numpy.abs(beta[1:] - beta[:-1])
            swap = self._swapstat.value[:self.n_beta.value-1]
            return numpy.sum(swap * dbeta) < min_p_acc
        else:
            raise ValueError("Invalid criteria for spawning: %s" % spawn_type)

    def get_spawn_loc(self):
        fup = self.get_fup()
        pos = numpy.argmax(numpy.abs(fup[1:] - fup[:-1]))
        return pos

    def spawn_beta(self, pos=None):

        if pos is None:
            pos = self.n_beta.value - 2

        # check to see if we need more memory before we can spawn
        if self.batch_size + self.n_chain.value + self.t_batch_size > len(self._beta.value):
            status = self.increase_storage()
            if not status:
                return False

        # maintain references for ease of use 
        # WARNING: code above cannot use these references as they might be irrelevant by the
        # time new storage is allocated
        n_beta = self.n_beta.value
        n_chain = self.n_chain.value
        beta = self._beta.value
        mixstat = self._mixstat.value
        nvis = self._buffer.value[self.batch_size:]
        nup = self._nup.value
        ndown = self._ndown.value
        fup_target = self._fup_target.value

        # the choice of new temperature doesn't really matter as it will be overridden
        ordered_beta = beta[mixstat[0,:]]
        new_beta = (ordered_beta[pos] + ordered_beta[pos+1])/2.
        print '******** spawning new chain @ new_beta = %.12f **********' % new_beta

        # shift things to the right to make room room for new chain
        temp = numpy.arange(n_beta, pos+1, -1)
        mixstat[:, temp] = mixstat[:, temp-1]
        nup[temp] = nup[temp-1]
        ndown[temp] = ndown[temp-1]

        # retrieve pointer to where new chains should be stored ...
        tp = numpy.arange(n_chain, n_chain + self.t_batch_size)
        
        # spawned chain has same properties as chain below
        # now initialize mixstat, beta and buffer for new chain
        beta[tp] = new_beta
        mixstat[:,pos+1] = tp
      
        # we have one more temperature, but t_batch_size*1 more chains
        n_beta += 1
        n_chain += self.t_batch_size

        # statistics relating to new beta are estimated from their neighbors
        nup[pos+1] = (nup[pos] + nup[pos+2])/2.
        ndown[pos+1] = (ndown[pos] + ndown[pos+2])/2.
        self._swapstat.value[pos+1] = (self._swapstat.value[pos] + \
                                       self._swapstat.value[pos+2])/2.
        
        # particles and their respective statistics are initialized to a "clean slate" and will
        # have to be burned in.
        nvis[tp] = copy.copy(nvis[mixstat[:, pos]])
        self.labels.value[tp]= self.labels.value[mixstat[:, pos]]
        self.rtime.value[tp] = self.rtime.value[mixstat[:, pos]]
 
        # update fup_target
        fup_target[:n_beta] = numpy.linspace(1,0,n_beta)

        return True

    def save(self, fname='model.pkl'):
        fp = open(fname, 'w')
        cPickle.dump(self.shared_vals, fp, protocol=cPickle.HIGHEST_PROTOCOL)
        fp.close()

    def load(self, fname='model.pkl'):
        model = self.load_model_file(fname)

        self._buffer.value     = numpy.array(model['buffer'], dtype=theano.config.floatX)
        self.mf_buffer.value   = numpy.array(model['mf_buffer'], dtype=theano.config.floatX)
        self._mixstat.value    = model['mixstat']
        self._E.value          = numpy.array(model['E'], dtype=theano.config.floatX)
        self.hbias.value       = numpy.array(model['hbias'], dtype=theano.config.floatX)
        self.vbias.value       = numpy.array(model['vbias'], dtype=theano.config.floatX)
        self.W.value           = numpy.array(model['W'], dtype=theano.config.floatX)
        self.n_chain.value     = model['n_chain']
        self.labels.value      = model['labels[t-1][t-1]']
        self.avg_rtime.value   = numpy.array(model['avg_rtime[t-1][t-1]'], dtype=theano.config.floatX)
        self._beta.value       = numpy.array(model['beta'], dtype=theano.config.floatX)
        self._nup.value        = numpy.array(model['nup'], dtype=theano.config.floatX)
        self._ndown.value      = numpy.array(model['ndown'], dtype=theano.config.floatX)
        self._swapstat.value   = numpy.array(model['swapstat'], dtype=theano.config.floatX)
        self._fup_target.value = numpy.array(model['fup_target'], dtype=theano.config.floatX)
        self.rtime.value       = model['rtime[t-1][t-1]']

    @classmethod
    def load_model_file(cls, fname):
        fp = open(fname, 'r')
        model = dict(cPickle.load(fp))
        fp.close()
        return model

    @classmethod
    def build_from_file(cls, fname='model.pkl', input=None, tau=None):

        model = cls.load_model_file(fname)

        t_batch_size = model['mixstat'].shape[0]
        beta_lbound = numpy.min(model['beta'][:model['n_chain']]) if model['n_chain'] else 0
        batch_size = numpy.sum(numpy.sum(model['buffer'], axis=1) != 0) - \
                     t_batch_size * model['n_beta']
                    
        rbm = cls(input=None, 
                n_visible=model['W'].shape[0], 
                n_hidden=model['W'].shape[1],
                batch_size = batch_size, 
                t_batch_size = t_batch_size, 
                n_beta=model['n_beta'],
                beta_lbound=beta_lbound, tau=tau)
        
        rbm.load(fname)
        return rbm


class PT_GetOptimalBetas(theano.Op):
   
    def make_node(self, fup, fup_target, betas):
        fup, fup_target, betas = map(T.as_tensor_variable, [fup, fup_target, betas])
        return gof.Apply(self, (fup, fup_target, betas), [betas.type()])
    
    def perform(self, node, (fup, fup_target, betas), outputs):
        """ 
        Given inverse temperatures between B_0 = betas[0] and B_M = betas[-1], 
        adapt betas {B_1,...,B_{M-1}}
        """
        betas_out = numpy.zeros_like(betas)
        betas_out[0] = 1

        M = numpy.float(len(betas))
        fup_sorted_arg = numpy.argsort(fup) 
        fup_sorted = fup[fup_sorted_arg]
        idx = numpy.arange(1,M-1)
        
        for i, fup_targ in zip(idx, fup_target[1:-1]):
            ubound_i = numpy.searchsorted(fup_sorted, fup_targ)
            # for given fup target, find tighest lower and upper bounds in current array
            fup_lower, fup_upper = fup_sorted[ubound_i-1], fup_sorted[ubound_i]
            # find matching beta values
            b_lower_i, b_upper_i  = fup_sorted_arg[ubound_i-1], fup_sorted_arg[ubound_i]
            beta_lower, beta_upper = betas[b_lower_i], betas[b_upper_i]

            db_df = (beta_upper - beta_lower) /  (fup_upper - fup_lower)
            betas_out[i] = beta_lower + db_df * (fup_targ - fup_lower)
        
        temp = numpy.sort(betas_out)
        outputs[0][0] = temp[::-1]

pt_get_optimal_betas = PT_GetOptimalBetas()


class PT_UpdateHistogram(theano.Op):

    def __init__(self, inplace=False):
        self.destroy_map = {0: [2], 1: [3]} if inplace else {}
        self.inplace = inplace

    def make_node(self, mixstat, labels, nup, ndown, alpha):
        mixstat, labels, nup, ndown, alpha = map(T.as_tensor_variable, 
                [mixstat, labels, nup, ndown, alpha])
        return gof.Apply(self, (mixstat, labels, nup, ndown, alpha), 
                               [nup.type(), ndown.type()])

    def perform(self, node, (mixstat, labels, nup, ndown, alpha), outputs):

        nup_out   = nup if self.inplace else copy.copy(nup)
        ndown_out = ndown if self.inplace else copy.copy(ndown)
      
        labels = labels[mixstat]

        # do not perform moving average when there are only un-initialized particles at the
        # given temperature
        nan_mask = numpy.sum(numpy.isnan(labels), axis=0) != len(mixstat)
        nup_out[nan_mask]   *= (1-alpha)
        ndown_out[nan_mask] *= (1-alpha)

        # get average label over a minibatch, ignoring uninitialized particles
        lbl_mean = stats.nanmean(labels, axis=0)

        # histogram bins are subject to an exponential moving average
        nup_out[nan_mask] += alpha * lbl_mean[nan_mask]
        ndown_out[nan_mask] += alpha * (1-lbl_mean[nan_mask])

        outputs[0][0] = nup_out
        outputs[1][0] = ndown_out

pt_update_histogram = PT_UpdateHistogram()


class PT_WasteReduction(theano.Op):

    def __init__(self, cut_off=1e-2, batch_size=0):
        self.view_map = {0: [0]}
        self.cut_off = cut_off
        self.batch_size = batch_size

    def make_node(self, vis, beta, mixstat, E):
        vis, beta, mixstat, E = map(T.as_tensor_variable, [vis, beta, mixstat, E])
        particle_weights = T.vector('particle_weights', dtype=beta.dtype)
        nbsize = T.scalar('nbsize', dtype = beta.dtype)
        return theano.gof.Apply(self, (vis, beta, mixstat, E), 
                                [vis.type(), particle_weights, nbsize])

    def perform(self, node, (vis, beta, mixstat, E), outputs):

        t_batch_size = numpy.float(mixstat.shape[0])
        xtra_per_batch = self.batch_size / t_batch_size
        n_beta = beta.shape[0] / t_batch_size

        particle_weights = numpy.zeros(len(vis), dtype=beta.dtype)

        # we only care about the energy of the tempered chains ...
        tE = E[self.batch_size:]

        t1_i = 0
        t1_weight = 0
        for i, bi_mixstat in enumerate(mixstat):
            
            # calculate exchange probability between two particles for a given minibatch
            r_beta = beta[bi_mixstat] - 1.
            r_E = tE[bi_mixstat] - tE[bi_mixstat[0]]
            swap_prob = numpy.minimum(1, numpy.exp((r_beta * r_E)))

            # need to normalize while taking into account batch of T=1 samples
            normalizer = numpy.sum(swap_prob) + xtra_per_batch
            particle_weights[t1_i : t1_i+xtra_per_batch] = 1. / normalizer
            particle_weights[self.batch_size + bi_mixstat] = swap_prob / normalizer

            t1_i += xtra_per_batch
            t1_weight += (xtra_per_batch + 1) * 1./normalizer

        # fractional weight of T=1 samples in gradient
        nbsize = numpy.asarray(t_batch_size / t1_weight, dtype=beta.dtype)

        idx = particle_weights > 1./normalizer * self.cut_off
        outputs[0][0] = vis[numpy.where(idx == True)]
        outputs[1][0] = particle_weights[idx]
        outputs[2][0] = nbsize

def pt_waste_reduction(vis, beta, mixstat, E, cut_off=1e-2, batch_size=0): 
    return PT_WasteReduction(cut_off, batch_size)(vis, beta, mixstat, E)


class PT_Swaps(theano.Op):

    def __init__(self, rng, inplace=False, movavg_alpha=0.1):
        self.rng = rng
        self.destroy_map = {0: [0], 1: [1], 2: [3], 3: [4], 4: [5], 5: [6]} if inplace else {}
        self.inplace = inplace
        self.movavg_alpha = movavg_alpha

    def make_node(self, beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset):
        inputs = [beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset]
        beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset = map(T.as_tensor_variable, inputs)
        return gof.Apply(self, 
                (beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset), 
                [beta.type(), mixstat.type(), labels.type(), swapstat.type(),
                 rtime.type(), avg_rtime.type()])

    def perform(self, node, (beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset), outputs):
        t_batch_size = numpy.float(mixstat.shape[0])
        n_beta = beta.shape[0] / t_batch_size
        n_chain = n_beta * t_batch_size

        beta_out     = beta     if self.inplace else copy.copy(beta)
        mixstat_out  = mixstat  if self.inplace else copy.copy(mixstat)
        labels_out   = labels   if self.inplace else copy.copy(labels)
        swapstat_out = swapstat if self.inplace else copy.copy(swapstat)
        rtime_out    = rtime    if self.inplace else copy.copy(rtime)
        avg_rtime_out = avg_rtime if self.inplace else copy.copy(avg_rtime)

        def write_outputs():
            outputs[0][0] = beta_out
            outputs[1][0] = mixstat_out
            outputs[2][0] = labels_out
            outputs[3][0] = swapstat_out
            outputs[4][0] = rtime_out
            outputs[5][0] = numpy.asarray(avg_rtime_out, dtype=theano.config.floatX)

        if n_beta==1 or (offset and n_beta==2):
            write_outputs()
            return

        # we count return time in number of Gibbs steps
        if not offset: 
            rtime_out[:n_chain] += 1

        # pick pairs of chains to swap 
        ti1 = numpy.arange(offset,   n_beta, 2, dtype='int32')
        ti2 = numpy.arange(offset+1, n_beta, 2, dtype='int32')
        ti1 = ti1[:len(ti2)]

        swapstat_ti1 = 0

        for bi in numpy.arange(t_batch_size):

            # retrieve pointer to particles at given temperature
            tp1 = mixstat_out[bi, ti1]
            tp2 = mixstat_out[bi, ti2]

            # calculate exchange probability between two particles
            r_beta = beta_out[tp2] - beta_out[tp1]
            r_E = E[tp2] - E[tp1]
            log_r = r_beta * r_E
            r = numpy.exp(log_r)
            swap_prob = numpy.minimum(1, r)
            swap = self.rng.rand() < swap_prob

            # extract index list of particles to swap
            idx = numpy.where(swap == True)
            sti1, sti2 = ti1[idx], ti2[idx]
            stp1, stp2 = tp1[idx], tp2[idx]

            # move pointers around to reflect swap
            mixstat_out[bi,sti1] = stp2  # move high to low
            mixstat_out[bi,sti2] = stp1  # move low to high
            # update temperatures as well
            beta_out_stp1 = copy.copy(beta_out[stp1])
            beta_out[stp1] = beta_out[stp2]
            beta_out[stp2] = beta_out_stp1
        
            # we swapped the first two chains
            if ti1[0]==0 and swap[0]:
                # retrieve pointer to new particle at T=1
                tp = stp2[0]
                # update return time average when down particle arrives at T=1
                if labels_out[tp] == LBL_DOWN:
                   rtime_out[tp] = 0
                # it then becomes an up-particle
                labels_out[tp] = LBL_UP
           
            # list of "swapped indices" contains last chain (B=0)
            if ti2[-1]==(n_beta-1) and swap[-1]:
                # up-particles become down particles when reaching highest temp.
                if labels_out[stp1[-1]] == LBL_UP:
                    labels_out[stp1[-1]] = LBL_DOWN

            swapstat_ti1 += swap_prob
      
        swapstat_out[ti1] = (1-tau)*swapstat_out[ti1] + tau*swapstat_ti1/t_batch_size

        # modify average return time
        labeled_idx = labels_out != LBL_NONE
        avg_rtime_out = avg_rtime_out*(1-self.movavg_alpha) + \
                        2*numpy.mean(rtime_out[labeled_idx])*self.movavg_alpha

        write_outputs()


class PT_SwapT1Sample(theano.Op):

    def __init__(self, rng, batch_size=0, inplace=False):
        self.destroy_map = {0: [0], 1: [1]} if inplace else {}
        self.inplace = inplace
        self.rng = rng
        self.batch_size = batch_size

    def make_node(self, buffer, E, mixstat):
        buffer, E, mixstat = map(T.as_tensor_variable, [buffer, E, mixstat])
        return gof.Apply(self, (buffer, E, mixstat),
                               [buffer.type(), E.type()])

    def perform(self, node, (buffer, E, mixstat), outputs):
        buffer_out = buffer if self.inplace else copy.copy(buffer)
        E_out      = E if self.inplace else copy.copy(E)
        
        t_batch_size = mixstat.shape[0]
        idx = self.rng.permutation(self.batch_size)[:t_batch_size]
        
        # swap particles
        temp = buffer[idx]
        buffer_out[idx] = buffer[self.batch_size + mixstat[:,0]]
        buffer_out[self.batch_size + mixstat[:,0]] = temp
        # swap energy
        temp = E[idx]
        E_out[idx] = E[self.batch_size + mixstat[:,0]]
        E_out[self.batch_size + mixstat[:,0]] = temp

        outputs[0][0] = buffer_out
        outputs[1][0] = E_out


class PT_Unmix(theano.Op):

    def __init__(self, batch_index=None):
        self.view_map = {0: [0]}
        self.batch_index = batch_index

    def make_node(self, input, mixstat):
        input, mixstat = map(T.as_tensor_variable, [input, mixstat])
        if self.batch_index:
            output = T.matrix(dtype=input.dtype)
        else:
            output = T.vector(dtype=input.dtype)
        return theano.gof.Apply(self, (input, mixstat), [output])

    def perform(self, node, (input, mixstat), outputs):
        if self.batch_index is None:
            unmixed = numpy.zeros(mixstat.shape, dtype = input.dtype)
            for i in xrange(unmixed.shape[0]):
                unmixed[i] = input[mixstat[i]]
        else:
            unmixed = input[mixstat[self.batch_index]]

        outputs[0][0] = unmixed

def pt_unmix(input, mixstat, batch_index=None): return PT_Unmix(batch_index)(input, mixstat)


class PT_UpdateBetaVector(theano.Op):
    
    def make_node(self, new_beta, mixstat, fup_target, nup, ndown):
        new_beta, mixstat, fup_target, nup, ndown = \
                map(T.as_tensor_variable, 
                    [new_beta, mixstat, fup_target, nup, ndown])
        return gof.Apply(self, 
                (new_beta, mixstat, fup_target, nup, ndown), 
                [new_beta.type(), mixstat.type(), 
                 fup_target.type(), nup.type(), ndown.type()])

    def perform(self, node, (new_beta, mixstat, fup_target, nup, ndown), outputs):

        # reshuffle mixstat to account for possible reordering of betas
        idx = numpy.argsort(new_beta)[::-1]

        mixstat_out    = copy.copy(mixstat)[:, idx]
        fup_target_out = copy.copy(fup_target)[:, idx]
        nup_out        = copy.copy(nup)[:, idx]
        ndown_out      = copy.copy(ndown)[:, idx]

        # now take vector of "unique" beta values, and reshuffle according to mixstat
        betas = numpy.zeros(mixstat.shape[0] * mixstat.shape[1])
        for bi in xrange(len(mixstat)):
            betas[mixstat[bi]] = new_beta
        
        outputs[0][0] = betas
        outputs[1][0] = mixstat_out
        outputs[2][0] = fup_target_out
        outputs[3][0] = nup_out
        outputs[4][0] = ndown_out

pt_update_beta_vector = PT_UpdateBetaVector()


class PT_GetSamplesAtTempIndex(theano.Op):

    def __init__(self):
        self.view_map = {0: [0]}

    def make_node(self, input, mixstat, ti):
        input, mixstat, ti = map(T.as_tensor_variable, [input, mixstat, ti])
        return theano.gof.Apply(self, (input, mixstat, ti), [input.type()])

    def perform(self, node, (input, mixstat, ti), outputs):
        outputs[0][0] = input[mixstat[:,ti]]

pt_get_samples_at_ti = PT_GetSamplesAtTempIndex()
