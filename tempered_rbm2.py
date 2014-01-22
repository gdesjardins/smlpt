"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs 
to those without visible-visible and hidden-hidden connections. 
"""


import numpy, time, cPickle, gzip, PIL.Image, os, copy
from scipy import stats

import theano
import theano.tensor as T
from theano import gof
from theano.compile import optdb
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor import nnet
from theano.gof.destroyhandler import fast_inplace_check

# Labels which are attached to a given particle 
LBL_UP   = 1          # defines "up-moving" particles (from B=1 to B=0)
LBL_DOWN = 0          # defines "down-moving" particles (from B=0 to B=1)
LBL_NONE = numpy.nan  # particles which haven't been initialized yet

def rtime_deo(p_acc, n): 
    return ((1-p_acc)*2*n*(n-1)) / p_acc

class RBM(object):
    """Restricted Boltzmann Machine (RBM) trained with SML-PT"""

    params = property(lambda s: [s.W, s.vbias, s.hbias])
    param_vals = property(lambda s: [p.value for p in s.params])
    
    shared_vars = property(lambda s: \
            [s.W, s.hbias, s.vbias, s.n_beta, s.n_chain, s._nvis,
             s._E, s._beta, s._mixstat, s.labels, s._fup_target,
             s._nup, s._ndown, s.rtime, s.avg_rtime, s._swapstat])
    shared_vals = property(lambda s: [(p.name,p.value) for p in s.shared_vars])

    def __init__(self, input=None, n_visible=784, n_hidden=500, \
        W=None, hbias=None, vbias=None, numpy_rng = None, theano_rng=None,
        batch_size=1, n_beta=10, beta_lbound=0., n_swaps=10, 
        n_rtime=1, rtime_a=1, rtime_b=100, tau=None):

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
        :param n_rtime: time constant is inversely proportional to n_rtime x <return time>
        :param tau: optional fixed time constant (overries n_rtime)
        """
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
  
        # deal with random number generation
        self.numpy_rng = numpy_rng if numpy_rng is not None else numpy.random.RandomState(123)
        self.theano_rng = theano_rng if theano_rng is not None else RandomStreams(self.numpy_rng.randint(2**30))

        if W is None : 
           # W is initialized with `initial_W` which is uniformely sampled
           # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
           # the output of uniform if converted using asarray to dtype 
           # theano.config.floatX so that the code is runable on GPU
           initial_W = 0.01 * self.numpy_rng.randn(n_visible, n_hidden)
           initial_W = numpy.asarray(initial_W, dtype = theano.config.floatX)
           # theano shared variables for weights and biases
           W = theano.shared(value = initial_W, name = 'W')

        if hbias is None :
           # create shared variable for hidden units bias
           hbias = theano.shared(value = numpy.zeros(n_hidden, 
                               dtype = theano.config.floatX), name='hbias')

        if vbias is None :
           # create shared variable for visible units bias
           vbias = theano.shared(value =numpy.zeros(n_visible, 
                                dtype = theano.config.floatX),name='vbias')

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input 
        if not input:
            self.input = T.matrix('input')

        self.W          = W
        self.hbias      = hbias
        self.vbias      = vbias

        bufsize = 100

        #########################################################################
        # Fields indexed by mixstat:    nvis, E, beta, labels, rtime
        # Fields indexed by temp index: mixstat, fup_target, nup, ndown, swapstat
        #########################################################################

        ### initialize tempering stuff ###
        self.batch_size = batch_size   # size of negative minibatch
        self.n_beta  = theano.shared(n_beta, name='n_beta') # number of temperatures in system
        self.n_chain = theano.shared(batch_size * n_beta, name='n_chain') # number of active chains in nvis array

        self._nvis = theano.shared(self.numpy_rng.randint(0,2,size=(batch_size*bufsize, n_visible)), name='nvis')
        self.nvis = self._nvis[:self.n_chain]

        # vectors containing energy and free-energy of current negative particles (at T=1)
        self._E = theano.shared(numpy.zeros(batch_size*bufsize), name='E')
        self.E = self._E[:self.n_chain]

        ## Betas are parametrized as delta_bi = exp(\lambda_i)
        ## Resulting betas are linearly spaced between 1 and 0

        # shared parameters are the lambda_i
        lambdas = numpy.zeros(bufsize)  # leave room to grow ...        
        lambdas[:n_beta-2] = numpy.log((1.0 - beta_lbound)/(n_beta-1))
        self._lambdas = theano.shared(lambdas, name='lambdas')
        self.lambdas = self._lambdas[:n_beta-2]

        # initialize data structure to map nhid/nvis rows to a given temperature
        mixstat = numpy.zeros((batch_size, bufsize), dtype='int32')
        mixstat[:, :n_beta] = numpy.arange(batch_size*n_beta).reshape(batch_size, n_beta)
        self._mixstat = theano.shared(mixstat, name='mixstat')
        self.mixstat = self._mixstat[:, :self.n_beta]

        # convert lambdas to actual beta values
        _betas1 = 1 - T.cumsum(T.exp(self.lambdas))
        _betas2 = T.join(0, T.shape_padright(1.0), _betas1)
        _betas3 = T.join(0, _betas2, T.shape_padright(beta_lbound))
        self.betas = _betas3
        self.mixed_betas = pt_mix(self.betas, self.mixstat)
        self.mixed_betas_matrix = T.shape_padright(self.mixed_betas)

        self.get_betas = theano.function([], self.betas)

        # labels: 1 means going up in temperature, 0 going down in temperature
        labels = LBL_NONE * numpy.ones(batch_size*bufsize, dtype='int32')
        labels[mixstat[:,0]] = LBL_UP
        self.labels = theano.shared(labels, name='labels') 

        # configure histogram of up moving particles
        _nup = numpy.zeros(bufsize)
        _nup[:n_beta] = numpy.linspace(1,0,n_beta)
        self._nup = theano.shared(_nup, name='nup')
        self.nup = self._nup[:self.n_beta]
        
        # configure histogram of down moving particles
        _ndown = numpy.zeros(bufsize)
        _ndown[:n_beta] = numpy.linspace(0,1,n_beta)
        self._ndown = theano.shared(_ndown, name='ndown')
        self.ndown = self._ndown[:self.n_beta]

        # return time
        rtime = numpy.zeros(batch_size*bufsize, dtype='int32')
        self.rtime = theano.shared(rtime, name='rtime') 
        self.avg_rtime = theano.shared(
                numpy.asarray(rtime_deo(0.4,n_beta), dtype=theano.config.floatX), 
                name='avg_rtime')

        # use return time as the time constant for all moving averages
        if not tau:
            self.tau = rtime_a/(n_rtime*self.avg_rtime + rtime_b)
        else:
            self.tau = T.as_tensor(tau)
        self.get_tau = theano.function([], self.tau)

        # create PT Op
        self.n_swaps = n_swaps
        self._swapstat = theano.shared(numpy.zeros(bufsize), name='swapstat')
        self.swapstat = self._swapstat[:self.n_beta]

        self.pt_swaps = PT_Swaps(n_swaps=self.n_swaps, seed=self.numpy_rng.randint(1 << 32))

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
            activation *= self.mixed_betas_matrix
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
            activation *= self.mixed_betas_matrix
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
 
    def pt_step(self, v0_sample, mixstat, labels, swapstat, 
                rtime, avg_rtime, nup, ndown):

        h1_mean, h1_sample, v1_mean, v1_sample = self.gibbs_vhv(v0_sample, temper=True) 
        E = self.energy(v1_sample, h1_sample)

        if self.n_beta.value > 1:

            mixed_betas = pt_mix(self.betas, self.mixstat)

            mixstat, labels, swapstat, rtime, avg_rtime = \
                    self.pt_swaps(mixed_betas, mixstat, E, labels, swapstat,\
                                  rtime, avg_rtime, self.tau, offset=0)

            nup, ndown = pt_update_histogram(mixstat, labels, nup, ndown, self.tau)
           
            new_mixed_betas = pt_mix(self.betas, mixstat)

            mixstat, labels, swapstat, rtime, avg_rtime = \
                    self.pt_swaps(new_mixed_betas, mixstat, E, labels, swapstat,\
                                  rtime, avg_rtime, self.tau, offset=1)
            
            nup, ndown = pt_update_histogram(mixstat, labels, nup, ndown, self.tau)

        return [h1_sample, v1_sample, mixstat, E, labels, 
                swapstat, rtime, avg_rtime, nup, ndown]
 
    def get_sampling_updates(self, k=1):
        """ 
        This functions implements one step of PCD-k
        :param k: number of Gibbs steps to do in PCD-k
        """
        # perform actual negative phase
        [nh_samples, nv_samples, mixstat, E, labels, 
         swapstat, rtime, avg_rtime, nup, ndown], updates = \
            theano.scan(self.pt_step, 
                        # this provides a mapping from pt_step's outputs to inputs, and
                        # provides values for initialization. Here
                        # v0_sample<=output[0], beta<=output[2], mixstat<=output[3]
                        outputs_info = [None, self.nvis, self.mixstat, None,
                                        self.labels, self.swapstat, self.rtime, 
                                        self.avg_rtime, self.nup, self.ndown],
                        #non_sequences = [self.mixed_betas],
                        n_steps = k)

        updates = {}
        updates[self.labels] = labels[-1]
        updates[self.rtime] = rtime[-1]
        updates[self.avg_rtime] = avg_rtime[-1]
        # vectors whose first n_chain components need to be updated
        updates[self._nvis] = T.set_subtensor(self._nvis[:self.n_chain], nv_samples[-1])
        updates[self._E] = T.set_subtensor(self._E[:self.n_chain], E[-1])
        # vector and matrix whose first n_beta components need to be updated
        updates[self._swapstat] = T.set_subtensor(self._swapstat[:self.n_beta], swapstat[-1])
        updates[self._mixstat] = T.set_subtensor(self._mixstat[:,:self.n_beta], mixstat[-1])
        updates[self._nup] = T.set_subtensor(self._nup[:self.n_beta], nup[-1])
        updates[self._ndown] = T.set_subtensor(self._ndown[:self.n_beta], ndown[-1])

        return [nh_samples[-1], nv_samples[-1], mixstat[-1], E[-1]], updates
       
    def get_learning_gradients(self, k=1, l1=0.0, l2=0.0, waste_cutoff=1e-3):
        """ 
        This functions implements one step of CD-k or PCD-k
        Returns a proxy for the cost and the updates dictionary. The dictionary contains the
        update rules for weights and biases but also an update of the shared variable used to
        store the persistent chain, if one is used.
        :param k: number of Gibbs steps to do in CD-k/PCD-k
        """
        updates = {}

        # compute positive phase
        ph_mean, ph_sample = self.sample_h_given_v(self.input, temper=False)

        # compute negative phase
        [nh_samples, nv_samples, mixstat, E], sampling_updates = self.get_sampling_updates(k=k)
        updates.update(sampling_updates)

        new_mixed_betas = pt_mix(self.betas, mixstat)

        # determine gradients on RBM parameters
        self.waste_reduction_t1frac = theano.shared(1.0, name='waste_reduction_t1frac')
        chain_end, weights, t1frac = pt_waste_reduction(nv_samples, 
                new_mixed_betas, mixstat, E, cut_off=waste_cutoff)
        updates[self.waste_reduction_t1frac] = t1frac

        cost = T.mean(self.free_energy(self.input)) - \
               T.sum(weights*self.free_energy(chain_end)) / mixstat.shape[0]
        if l1: cost += l1 * T.sum(abs(self.W))
        if l2: cost += l2 * T.sum(self.W**2)

        # We must not compute the gradient through the gibbs sampling 
        gparams = T.grad(cost, self.params, consider_constant = [chain_end, weights])

        grads = {}
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            grads[param] = gparam

        return grads, updates


    def config_beta_updates(self, beta_lr=1e-3):

        # determine fraction of up-moving particles at each temperature
        fup = self.nup / (self.nup + self.ndown)
        self.get_fup = theano.function([], fup)

        # cost function is: C = \sum_i (fup_{i+1} - fup_i) ^ 2
        # \frac{dC}{d \lambda_i} = \sum_{j >= i}
        #    \frac{dC}{df_j}
        #    \frac{df_j}{d\beta_j}
        #    \frac{d\beta_j}{d \delta \beta_i      = -1
        #    \frac{d \delta \beta_i}{d\lambda_i}   = exp(\lambda_i)

        # vectors of length n_beta-3
        f_i = fup[1:-1]
        f_im1 = fup[:-2]
        f_ip1 = fup[2:]
        
        ## \frac{dC}{df_j} ##
        # vector of length n_beta-2
        dc_df = 2*(f_i - f_im1) - 2*(f_ip1 - f_i)

        ## \frac{df_j}{d\beta_j} : estimate it from empirical data ##
        # vector of length n_beta-1
        df_db = (fup[1:] - fup[:-1]) / (self.betas[1:] - self.betas[:-1] + 1e-3)
        # vector of length n_beta-2
        df_db_avg = (df_db[1:] + df_db[:-1])/2.

        dc_dlambda = T.cumsum(dc_df * df_db_avg * -1 * T.exp(self.lambdas))

        # gradient-based beta update
        new_lambdas = self.lambdas - beta_lr * dc_dlambda

        updates = {self._lambdas: T.set_subtensor(self._lambdas[:self.n_beta-2], new_lambdas)}
        self.grad_update_betas = theano.function([], new_lambdas, updates=updates)


    def increase_storage(self):
        """
        Attempts to double the allocated memory, such that chains can continue to be spawned.
        :rval: True if memory allocation succeeded, False otherwise
        """
        if self.n_beta.value > 100:
            print '************ ERROR: FAILED TO ALLOCATE NEW MEMORY *******************'
            return False
        
        n = 2*len(self._nvis.value)

        def double_shared(var):
            shp = numpy.array(var.value.shape)
            shp[0] *= 2
            new_var = numpy.zeros(shp)
            new_var[:len(var.value)] = var.value
            var.value = new_var

        map(double_shared, [self._nvis, self._E, self._lambdas, 
            self._swapstat, self._nup, self._ndown, self.labels, self.rtime])
        
        n = len(self._mixstat.value[0])
        _mixstat = numpy.zeros((self.batch_size, 2*n))
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
        if self.n_chain.value + self.batch_size > len(self._lambdas.value)+2:
            status = self.increase_storage()
            if not status:
                return False

        # maintain references for ease of use 
        # WARNING: code above cannot use these references as they might be irrelevant by the
        # time new storage is allocated
        n_beta = self.n_beta.value
        n_chain = self.n_chain.value
        lambdas = self._lambdas.value
        mixstat = self._mixstat.value
        nvis = self._nvis.value
        nup = self._nup.value
        ndown = self._ndown.value

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
        tp = numpy.arange(n_chain, n_chain + self.batch_size)
        
        # spawned chain has same properties as chain below
        # now initialize mixstat, beta and nvis for new chain
        beta[tp] = new_beta
        mixstat[:,pos+1] = tp
      
        # we have one more temperature, but batch_size*1 more chains
        n_beta += 1
        n_chain += self.batch_size

        # statistics relating to new beta are estimated from their neighbors
        nup[pos+1] = (nup[pos] + nup[pos+2])/2.
        ndown[pos+1] = (ndown[pos] + ndown[pos+2])/2.
        self._swapstat.value[pos+1] = self._swapstat.value[pos]
        
        # particles and their respective statistics are initialized to a "clean slate" and will
        # have to be burned in.
        nvis[tp] = copy.copy(nvis[mixstat[:, pos]])
        self.labels.value[tp]= self.labels.value[mixstat[:, pos]]
        self.rtime.value[tp] = self.rtime.value[mixstat[:, pos]]
        return True

    def save(self, fname='model'):
        fp = open(fname + '.pkl', 'w')
        cPickle.dump(self.shared_vals, fp, protocol=cPickle.HIGHEST_PROTOCOL)
        fp.close()


class PT_UpdateHistogram(theano.Op):

    def __init__(self, inplace=False):
        self.destroy_map = {0: [2], 1: [3]} if inplace else {}
        self.inplace = inplace

    def __eq__(self, other):
        return (type(self) == type(other) and self.inplace == other.inplace)

    def __hash__(self):
        return hash(PT_UpdateHistogram) ^ hash(self.inplace)
 
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

    def __init__(self, cut_off=1e-3):
        self.view_map = {0: [0]}
        self.cut_off = cut_off

    def __eq__(self, other):
        return type(self) == type(other) and self.cut_off == other.cut_off

    def __hash__(self):
        return hash(PT_WasteReduction) ^ hash(self.cut_off)

    def make_node(self, vis, beta, mixstat, E):
        vis, beta, mixstat, E = map(T.as_tensor_variable, [vis, beta, mixstat, E])
        particle_weights = T.vector('particle_weights', dtype=beta.dtype)
        t1_fraction = T.scalar('t1_fraction', dtype = beta.dtype)
        return theano.gof.Apply(self, (vis, beta, mixstat, E), 
                                [vis.type(), particle_weights, t1_fraction])

    def perform(self, node, (vis, beta, mixstat, E), outputs):

        particle_weights = numpy.zeros(beta.shape[0],dtype=beta.dtype)
        batch_size = numpy.float(mixstat.shape[0])
        n_beta = beta.shape[0] / batch_size

        for i, bi_mixstat in enumerate(mixstat):
            # calculate exchange probability between two particles
            r_beta = beta[bi_mixstat] - 1.
            r_E = E[bi_mixstat] - E[bi_mixstat[0]]
            swap_prob = numpy.minimum(1, numpy.exp((r_beta * r_E)))
            particle_weights[bi_mixstat] = swap_prob / numpy.sum(swap_prob)

        # fractional weight of T=1 samples in gradient
        t1_fraction = numpy.asarray(numpy.sum(particle_weights[mixstat[:,0]]) / batch_size,
                                    dtype = beta.dtype)

        idx = particle_weights > self.cut_off
        outputs[0][0] = vis[numpy.where(idx == True)]
        outputs[1][0] = particle_weights[idx]
        outputs[2][0] = t1_fraction

def pt_waste_reduction(vis, beta, mixstat, E, cut_off=1e-3): 
    return PT_WasteReduction(cut_off)(vis, beta, mixstat, E)


class PT_Swaps(theano.Op):

    def __init__(self, n_swaps, seed, inplace=False, movavg_alpha=0.1):
        self.n_swaps = n_swaps
        self.seed = seed
        self.rng = numpy.random.RandomState(seed)
        self.destroy_map = {0: [1], 1: [3], 2: [4], 3: [5], 4: [6]} if inplace else {}
        self.inplace = inplace
        self.movavg_alpha = movavg_alpha

    def __eq__(self, other):
        return (type(self) == type(other) and self.n_swaps == other.n_swaps and\
                self.seed == other.seed and self.inplace == other.inplace)

    def __hash__(self):
        return hash(PT_Swaps) ^ hash(self.n_swaps) ^ hash(self.seed) ^\
               hash(self.inplace)

    def make_node(self, beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset):
        inputs = [beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset]
        beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset = map(T.as_tensor_variable, inputs)
        return gof.Apply(self, 
                (beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset), 
                [mixstat.type(), labels.type(), swapstat.type(), rtime.type(), avg_rtime.type()])

    def perform(self, node, (beta, mixstat, E, labels, swapstat, rtime, avg_rtime, tau, offset), outputs):

        batch_size = numpy.float(mixstat.shape[0])
        n_beta = beta.shape[0] / batch_size
        n_chain = n_beta * batch_size

        mixstat_out  = mixstat  if self.inplace else copy.copy(mixstat)
        labels_out   = labels   if self.inplace else copy.copy(labels)
        swapstat_out = swapstat if self.inplace else copy.copy(swapstat)
        rtime_out    = rtime    if self.inplace else copy.copy(rtime)
        avg_rtime_out = avg_rtime if self.inplace else copy.copy(avg_rtime)

        def write_outputs():
            outputs[0][0] = mixstat_out
            outputs[1][0] = labels_out
            outputs[2][0] = swapstat_out
            outputs[3][0] = rtime_out
            outputs[4][0] = numpy.asarray(avg_rtime_out, dtype=theano.config.floatX)

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

        for bi in numpy.arange(batch_size):

            # retrieve pointer to particles at given temperature
            tp1 = mixstat_out[bi, ti1]
            tp2 = mixstat_out[bi, ti2]

            # calculate exchange probability between two particles
            r_beta = beta[tp2] - beta[tp1]
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
      
        swapstat_out[ti1] = (1-tau)*swapstat_out[ti1] + tau*swapstat_ti1/batch_size

        # modify average return time
        labeled_idx = labels_out != LBL_NONE
        avg_rtime_out = avg_rtime_out*(1-self.movavg_alpha) + \
                        numpy.mean(rtime_out[labeled_idx])*self.movavg_alpha

        write_outputs()



class PT_Unmix(theano.Op):

    def __init__(self, batch_index=None):
        self.view_map = {0: [0]}
        self.batch_index = batch_index

    def __eq__(self, other):
        return (type(self) == type(other) and self.batch_index == other.batch_index)

    def __hash__(self):
        return hash(PT_Unmix) ^ hash(self.batch_index)
 
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


class PT_Mix(theano.Op):

    def make_node(self, input, mixstat):
        input, mixstat = map(T.as_tensor_variable, [input, mixstat])
        return theano.gof.Apply(self, (input, mixstat), [input.type()])

    def perform(self, node, (input, mixstat), outputs):
        (bsize, n_beta) = mixstat.shape

        _input = numpy.tile(input, bsize)
        output = numpy.zeros_like(_input)

        for i in xrange(bsize):
            output[i*n_beta:(i+1)*n_beta] = _input[mixstat[i]]
       
        outputs[0][0] = output

pt_mix = PT_Mix()


class PT_UpdateBetaVector(theano.Op):
    
    def make_node(self, new_beta, mixstat):
        new_beta, mixstat = map(T.as_tensor_variable, [new_beta, mixstat])
        return gof.Apply(self, (new_beta, mixstat), [new_beta.type()])

    def perform(self, node, (new_beta, mixstat), outputs):
   
        # enforce constraints
        beta_out = copy.copy(new_beta)
        beta_out[beta_out > 1] = 1.0
        beta_out[beta_out < 0] = 0.0

        betas = numpy.zeros(mixstat.shape[0] * mixstat.shape[1])
        for bi in xrange(len(mixstat)):
            betas[mixstat[bi]] = beta_out
        
        outputs[0][0] = betas

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
