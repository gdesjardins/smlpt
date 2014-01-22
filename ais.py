import theano
import theano.tensor as T
import numpy, time, copy

from smlpt import sampler as sampler_module
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

if __name__ == '__main__':
    test_ais2()

# TODO: roll this into theano directly
floatX = theano.config.floatX
def shrd(X,name=None):
    return theano.shared(numpy.array(X).astype(floatX), name=name)

def rbm_ais(rbm_params, 
            n_runs, 
            vbias_a=None, 
            data=None,
            betas=None, 
            key_betas=None, 
            rng=None, 
            seed=23098):
    """
    Running time: with 500 hidden units, 3000 chains and n_runs=100, the following function
    took 263s to execute on atchoum.
    :param vbias_a: allows caller to override biases of base-rate model
    :param data: if set, configure base-rate model as a function of dataset
    """
    (W, vbias, hbias) = rbm_params

    if rng is None:
        rng = numpy.random.RandomState(seed)
 
    if data is None:
        if vbias_a is None:
            # configure base-rate biases to those supplied by user
            vbias_a = vbias
        else:
            vbias_a = vbias_a
    else:
        # set biases of base-rate model to ML solution
        data = numpy.asarray(data, dtype=floatX)
        data = numpy.mean(data, axis=0)
        data = numpy.minimum(data, 1-1e-5)
        data = numpy.maximum(data, 1e-5)
        vbias_a = -numpy.log(1./data - 1)
    hbias_a = numpy.zeros_like(hbias)
    W_a = numpy.zeros_like(W)

    # generate exact sample for the base model
    v0 = numpy.tile(1./(1+numpy.exp(-vbias_a)), (n_runs,1))
    v0 = numpy.array(v0 > rng.random_sample(v0.shape), dtype=floatX)

    # we now compute Z
    ais = rbm_z_ratio((W_a,vbias_a,hbias_a), 
                      rbm_params, n_runs, v0, 
                      betas=betas, key_betas=key_betas, rng=rng)
    dlogz, var_dlogz = ais.estimate_from_weights()

    # log Z = log_za + dlogz
    ais.log_za = W_a.shape[1]*numpy.log(2) + numpy.sum(numpy.log(1+numpy.exp(vbias_a)))
    ais.log_zb = ais.log_za + dlogz

    return (ais.log_zb, var_dlogz), ais

 
def rbm_z_ratio(rbmA_params, rbmB_params, n_runs, v0=None, 
                betas=None, key_betas=None, rng=None, seed=23098):
    if rng is None:
        rng = numpy.random.RandomState(seed)

    (W_a, vbias_a, hbias_a) = rbmA_params
    (W_b, vbias_b, hbias_b) = rbmB_params
    rbmA_params = [shrd(q) for q in rbmA_params]
    rbmB_params = [shrd(q) for q in rbmB_params]

    assert W_a.shape[0] == W_b.shape[0] and len(vbias_a)==len(vbias_b)

    # build tempered block Gibbs sampler to return visible units
    v0 = rng.rand(n_runs, W_b.shape[0]) if v0 is None else v0
    vis = shrd(v0, 'vis')
    beta = shrd(0.0, 'beta')
    sampler = rbm_ais_sampler(rbmA_params, rbmB_params, beta, vis)

    # build free energy function
    v = T.matrix('rbmais_fe_v')
    b = T.scalar('rbmais_fe_b')
    fe = rbm_ais_pk_free_energy(rbmA_params, rbmB_params, b, v)
    free_energy_fn = theano.function([b, v], fe, allow_input_downcast=False)

    ais = AIS(sampler, free_energy_fn, beta, n_runs)
    ais.set_betas(betas, key_betas=key_betas)
    ais.run()

    return ais


def rbm_ais_pk_free_energy(rbmA_params, rbmB_params, beta, v):

    def rbm_FE(rbm_params, v, b):
        (W, vbias, hbias) = rbm_params
        vis_term = b * T.dot(v, vbias)
        hid_act  = b * (T.dot(v, W) + hbias)
        fe = - vis_term - T.sum(T.log(1 + T.exp(hid_act)), axis=1)
        return fe

    fe_a = rbm_FE(rbmA_params, v, (1-beta))
    fe_b = rbm_FE(rbmB_params, v, beta)
    return fe_a + fe_b


def rbm_ais_sampler(rbmA_params, rbmB_params, beta, v, seed=23098):

    (W_a, vbias_a, hbias_a) = rbmA_params
    (W_b, vbias_b, hbias_b) = rbmB_params

    theano_rng = RandomStreams(seed)

    # equation 15
    ph_a = T.nnet.sigmoid((1-beta) * (T.dot(v,W_a) + hbias_a))
    ha_sample = theano_rng.binomial(size=(v.get_value().shape[0],len(hbias_a.get_value())), n=1, p=ph_a, dtype=floatX)
    
    # equation 16
    ph_b = T.nnet.sigmoid(   beta  * (T.dot(v,W_b) + hbias_b))
    hb_sample = theano_rng.binomial(size=(v.get_value().shape[0],len(hbias_b.get_value())), n=1, p=ph_b, dtype=floatX)

    # equation 17
    pv_act = (1-beta) * (T.dot(ha_sample, W_a.T) + vbias_a) + \
                beta  * (T.dot(hb_sample, W_b.T) + vbias_b)
    pv = T.nnet.sigmoid(pv_act)
    v_sample = theano_rng.binomial(size=(v.get_value().shape[0],len(vbias_b.get_value())), n=1, p=pv, dtype=floatX)

    sampler = sampler_module.BlockGibbsSampler({v: v_sample}, n_steps=1)
    return sampler


class AIS(object):

    def fX(a):
        return numpy.array(a,dtype=theano.config.floatX)

    dflt_beta =  numpy.hstack((fX(numpy.linspace(0,0.5,1e3)),
                              fX(numpy.linspace(0.5,0.9,1e4)),
                              fX(numpy.linspace(0.9,1.0,1e4))))
    
    def __init__(self, sampler, energy_fn, beta, n_runs, log_int=500):
       
        # initialize log_za to partition function at infinite temperature
        state = sampler.get_state().values()
        self.log_ais_w = numpy.zeros(n_runs, dtype=theano.config.floatX)
        
        self.sampler = sampler
        self.beta = beta 
        self.energy_fn = energy_fn
        self.n_runs = n_runs
        self.log_int = log_int

        ais_w = T.vector()
        dlogz = T.log(T.mean(T.exp(ais_w - T.max(ais_w)))) + T.max(ais_w)
        self.log_mean = theano.function([ais_w], dlogz, allow_input_downcast=False)

    def set_betas(self, betas=None, key_betas=None):
        """
        :param betas: vector of temperatures specifying interpolating distributions :param
        key_betas: if specified (not None), specifies specific temperatures at which we want to
                   compute the AIS estimate. AIS.run will then return a vector, containing AIS
                   at each key_beta temperature, including the nominal temperature.

        """
        self.key_betas = None if key_betas is None else numpy.sort(key_betas) 

        betas = numpy.array(betas, dtype=theano.config.floatX) \
                if betas is not None else self.dflt_beta
        # insert key temperatures within
        if key_betas is not None:
            betas = numpy.hstack((betas, key_betas))
            betas.sort()
    
        self.betas = betas

    def run(self, n_steps=1):
        if not hasattr(self, 'betas'):
            self.set_betas()

        # We use slightly different notation (for indices) than in the AIS paper 
        # w^i = p1(v0)*p2(v1)*...*pk(v_{k-1}) / [p0(v0)*p1(v1)*...*p_{k-1}(vk-1)]
        #     = p1(v0)/p0(v0) * p2(v1)/p1(v1) * ... * pk(v_{k-1})/p_{k-1}(vk-1)
        # log_w^i = fe_0(v0) - fe_1(v0) + 
        #           fe_1(v1) - fe_2(v1) + ... +
        #           fe_{k-1}(v_{k-1}) - fe_{k}(v_{k-1})

        state = self.sampler.get_state().values()
        self.std_ais_w = []
        self.logz_beta = []
        self.var_logz_beta = []

        sampler_t = 0
        energy_t = 0
        ki = 0
        for i in range(len(self.betas)-1):

            bp, bp1 = self.betas[i], self.betas[i+1]

            t1 = time.time()

            # log-ratio of (free) energies for two nearby temperatures
            self.log_ais_w += self.energy_fn(bp, *state) - \
                              self.energy_fn(bp1, *state)
            energy_t = time.time() - t1

            if (i+1)%self.log_int == 0:
                m = numpy.max(self.log_ais_w)
                std_ais = numpy.log(numpy.std(numpy.exp(self.log_ais_w-m))) + m - \
                          numpy.log(self.n_runs)/2;
                self.std_ais_w.append(std_ais)

            if self.key_betas is not None and \
               ki < len(self.key_betas) and \
               bp1 == self.key_betas[ki]:

                log_ais_w_bi, var_log_ais_w_bi = \
                    self.estimate_from_weights(self.log_ais_w)
                self.logz_beta.insert(0, log_ais_w_bi)
                self.var_logz_beta.insert(0, var_log_ais_w_bi)
                ki += 1

            # adapt beta such that next sample is from "new model"
            self.beta.set_value(bp1)

            # get new state
            t1 = time.time()
            state = self.sampler.draw().values()
            sampler_t += time.time() - t1

    def estimate_from_weights(self, log_ais_w=None):
        log_ais_w = self.log_ais_w if log_ais_w is None else log_ais_w

        # estimate the log-mean of the AIS weights
        dlogz = self.log_mean(log_ais_w)

        # estimate log-variance of the AIS weights
        # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
        m = numpy.max(log_ais_w)
        var_dlogz = log_ais_w.shape[0] *\
                     numpy.sum(numpy.exp(2*(log_ais_w - m)))/\
                     numpy.sum(numpy.exp(log_ais_w - m))**2 - 1.
                 
        return dlogz, var_dlogz
