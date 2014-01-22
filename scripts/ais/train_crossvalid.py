# allows to plot figures without X11 (i.e on cluster)
import matplotlib; matplotlib.use("Agg")
import numpy
import pdb
import pickle
import time
import os
import shutil
import copy
import tables
import resource
import pylab as pl

import theano
import theano.tensor as T

from theano import ProfileMode
from theano.printing import Print
from LeDeepNet import trainer
from smlpt import tools, rbm_tools, optimization, tempered_rbm, ais
from jobman import make, make2
from pylearn.io.seriestables import ErrorSeries

floatX = theano.config.floatX
def shrdX(value, name):
    return theano.shared(numpy.asarray(value, dtype=floatX), name=name)

def cpu_time():
    return resource.getrusage(resource.RUSAGE_SELF)[0]

def plot_betas(beta_log):

    t_max = max([len(b) for b in beta_log])

    for bi, b in enumerate(beta_log):
        t = numpy.arange(t_max - len(b), t_max)
        pl.plot(t, b)

    pl.savefig('betas.pdf')
    pl.close()

def plot_fup(betas, fup, t, pad=8):
    assert len(betas)==len(fup)
    fig = pl.figure()
    sp1 = fig.add_subplot(111)
    x1  = numpy.arange(len(betas))
    sp1.plot(x1, fup, 'b-.')
    sp1.set_xlabel('B [index]', color='b')
    sp1.set_xlim((0,len(betas)-1))
    sp1.set_ylabel('f_up')
    for tl in sp1.get_xticklabels():
        tl.set_color('b')

    sp2 = sp1.twiny()
    sp2.plot(betas, fup, 'r-^')
    sp2.set_xlabel('B', color='r')
    for tl in sp2.get_xticklabels():
        tl.set_color('r')

    fig_num = str(t)
    fig_num = '0'*(pad-len(fig_num)) + fig_num
    pl.title('fup @ %i'%t)
    pl.savefig('fup_update_%s.pdf'%fig_num)
    pl.close()

def plot_swap_stats(betas, rbm, t, pad=8):
    pl.plot(betas, rbm._swapstat.value[:len(betas)], 'b-o')
    pl.xlabel('B')
    pl.ylabel('Swap Prob.')
    pl.ylim((0,1))

    fig_num = str(t)
    fig_num = '0'*(pad-len(fig_num)) + fig_num
    pl.title('swapstat @ %i'%t)
    pl.savefig('swapstats_%s.pdf'%fig_num)
    pl.close()

def plot_Z(state, data=None):
    x = range(len(data['Z']))
    pl.plot(x, data['Z'], label='Z')

    if state.do_parallel_AIS:
        # parallel AIS estimate
        std = numpy.sqrt(numpy.array(data['pAIS_var']))
        #pl.errorbar(x, data['pAIS'], yerr=3*std, label='pAIS')
        pl.plot(x, data['pAIS'], label='pAIS')
    if state.rbm.serial_AIS_type:
        # serial AIS estimate
        std = numpy.sqrt(numpy.array(data['sAIS_var']))
        #pl.errorbar(x, data['sAIS'], yerr=3*std, label='sAIS')
        pl.plot(x, data['sAIS'], label='sAIS')
    if state.do_parallel_AIS and state.rbm.serial_AIS_type:
        # kalman filtered serial/parallel AIS estimate
        std = numpy.sqrt(numpy.array(data['fAIS_var']))
        #pl.errorbar(x, data['fAIS'], yerr=3*std, label='fAIS')
        pl.plot(x, data['fAIS'], label='fAIS')
    pl.legend()
    
    # plot log-likelihood on separate y-axis
    if state.compute_nll.vals!=-1:
        pl.twinx()
        x = range(len(data['ll']))
        pl.plot(x, data['ll'], label='NLL')
        pl.legend()

    title = 'nhid=%i lr=%.6f nbsize=%i sAIS=%s extraAIS=%i' %\
            (state.rbm.n_hid, state.rbm_init.lr, 
             state.rbm_init.nbsize, 
             state.rbm.serial_AIS_type,
             state.extra_AIS_sample)
    if hasattr(state.rbm_init, 'nbeta'):
        title += ' nbeta=%i' % state.rbm_init.nbeta
    pl.title(title)
    pl.savefig('AIS_' + title.replace(' ','_')+'.pdf')
    pl.close()

def experiment(state, channel):
    start_time = cpu_time()
    
    # Parse jobman parameters and set default values
    assert hasattr(state, 'base_lr')
    assert hasattr(state, 'lr_anneal_start')
    beta_lr  = state.get('beta_lr', state.base_lr)
    n_burnin = state.get('n_burnin', 1e3)
    momentum = state.get('momentum', 0.0)
    l1 = state.get('l1', 0.0)
    l2 = state.get('l2', 0.0)
    sparse_lambda = state.get('sparse_lambda', 0.0)
    sparse_p = state.get('sparse_p', 0.01)
    train_burnin = state.get('train_burnin', 0)
    walltime = state.get('walltime', None)
    min_p_acc = state.get('min_p_acc', 0.4)
    spawn_type = state.get('spawn_type', 'spawn_avg')
    state.rbm.tau = state.rbm.get('tau', None)
    spawn_burnin = state.get('spawn_burnin', 100)
    ais_runs = state.get('ais_runs', 100)
    polyak_start = state.get('polyak_start', -1)
    pos_batch_size = state.get('pos_batch_size', 100)
    debug = state.get('debug', False)
    logref = state.get('logref', 'time')
    assert spawn_type in ('spawn_avg')
    assert logref in ('time','updates')

    # Initialize param iterators
    _log_interval_ = make(state.log_interval)

    # make dataset
    dataset = make(state.dataset)

    # make RBM
    rbm = make(state.rbm)
    if rbm.n_beta.value == 1:
        state.spawn_beta = False
        state.adapt_beta = False

    if state.init_bias_to_mean:
        visb = -numpy.log(1./(numpy.mean(dataset.train.x, axis=0) + 1e-5) -1)
        rbm.vbias.value = copy.copy(visb) 

    ### define PT sampling function ###
    samples, sampling_updates = rbm.get_sampling_updates(k=1)
    sample_rbm = theano.function([], [], updates=sampling_updates)

    # define learning rate schedule
    iter = shrdX(0, name='iter')
    base_lr = numpy.asarray(state.base_lr, dtype=floatX)
    annealing_coef = T.switch(state.lr_anneal_start==-1, 1.0,
            T.clip(
               T.cast(state.lr_anneal_start / (iter+1.), floatX),
               0.0, base_lr))
    get_anneal_coef = theano.function([], annealing_coef)
    lr = base_lr * annealing_coef

    ###### DEFINE TRAINING FUNCTION ######
    # define gradients for SML-PT
    momentum = shrdX(momentum, name='momentum')
    learning_grads, sampling_updates = \
            rbm.get_learning_gradients(
                    k=state.k, 
                    l1=l1, l2=l2, 
                    sparse_lambda=sparse_lambda, 
                    sparse_p=sparse_p)

    # define learning updates
    learning_updates = optimization.sgd(learning_grads, lr, momentum)

    # use polyak averaging 
    polyak = optimization.PolyakAveraging(start=polyak_start)
    polyak_updates = polyak.config_updates(rbm.params, lr)
    model_averaging = theano.function([], [], updates=polyak_updates)
    
    # define free-energy function
    x = T.matrix('x')
    free_energy_x = tempered_rbm.free_energy(x, params=polyak.params)
    avg_model_free_energy = theano.function([x], free_energy_x)

    # define training function
    train_updates = {iter: iter + 1}
    train_updates.update(sampling_updates)
    train_updates.update(learning_updates)

    run_mode = state.get('run_mode', 'normal')
    if run_mode == 'profile':
        profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
        train_rbm = theano.function([rbm.input], [], updates=train_updates, mode=profmode)
    elif run_mode == 'pydot_profile':
        profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
        train_rbm = theano.function([rbm.input], [], updates=train_updates, mode=profmode)
        for i in range(1000):
            train_rbm(numpy.array(numpy.random.rand(100,784), dtype='float32'))
        theano.printing.pydotprint(train_rbm, mode=profmode, outfile='trainrbm.png', 
                                   compact=True, format='png', with_ids=False)
        profmode.print_summary()
        Quit
    else:
        train_rbm = theano.function([rbm.input], [], updates=train_updates)
    ###### END TRAINING FUNCTION ######

    # define functions to update temperature
    rbm.config_beta_updates()

    # the trainer takes care of handling the dataset
    t = trainer.Trainer(dataset, batch_size=pos_batch_size)
    t.set_active_dataset('train', randomize=False)

    # logging of various time series
    beta_log = [ [] for i in xrange(rbm.n_beta.value)]
    logger = tools.HDF5Logger("outputs.h5")
    state.nstamps = ""
    state.tstamps = ""
    state.train_nlls = ""
    state.valid_nlls = ""
    state.test_nlls  = ""
    state.logz  = ""
    state.logz_std  = ""
    state.logz_upper  = ""
    state.logz_lower  = ""

    rbm.best_valid_nll = -numpy.Inf

    def log_ais(n, train_time):

        #### compute AIS estimate of logZ ####
        log_z, (logz_std, logz_lower,logz_upper) = \
                ais.rbm_ais(rbm.param_vals, n_runs=ais_runs, rng=rbm.rng,
                            data=numpy.array(dataset.train.x, dtype=floatX))

        preproc = getattr(dataset, 'preprocess', None)
        train_nll = rbm_tools.compute_nll(avg_model_free_energy, dataset.train.x, log_z, preproc=preproc)
        valid_nll = rbm_tools.compute_nll(avg_model_free_energy, dataset.valid.x, log_z, preproc=preproc)
        test_nll  = rbm_tools.compute_nll(avg_model_free_energy, dataset.test.x,  log_z, preproc=preproc)

        # log history of likelihood scores in string format
        semicol = ':' if len(state.nstamps) else ''
        state.nstamps += semicol + "%i"   % n
        state.tstamps += semicol + "%.3g" % train_time
        state.train_nlls += semicol + "%.2f" % train_nll
        state.valid_nlls += semicol + "%.2f" % valid_nll
        state.test_nlls  += semicol + "%.2f" % test_nll
        
        state.logz       += semicol + "%.2f" % log_z
        state.logz_std   += semicol + "%.2f" % logz_std
        state.logz_lower += semicol + "%.2f" % logz_lower
        state.logz_upper += semicol + "%.2f" % logz_upper

        # store latest value seperately for easy querying
        state.n = n
        state.t = train_time
        state.train_nll = train_nll
        state.valid_nll = valid_nll
        state.test_nll  = test_nll

        logger.log('logz', n, log_z)
        logger.log('logz_std', n, logz_std)
        logger.log('logz_lower', n, logz_lower)
        logger.log('logz_upper', n, logz_upper)
        logger.log('train_nll', n, train_nll)
        logger.log('valid_nll', n, valid_nll)
        logger.log('test_nll', n, test_nll)

        print 'update %i/%i:' % (n, state.nupdates)
        print '   NLL=%f log_z=%f (+/- %f)'  %\
                (numpy.mean(train_nll), log_z, logz_upper-logz_lower)

        if valid_nll > rbm.best_valid_nll:
            save_model()
            rbm.best_valid_nll = valid_nll
      
    def log_stuff(n, train_time):

        print 'update %i/%i:' % (n, state.nupdates)

        est_rtime = tempered_rbm.rtime_deo(min_p_acc, rbm.n_beta.value)

        logger.log('t', n, train_time)
        logger.log('rtime', n, rbm.avg_rtime.value)
        logger.log('avgswap', n, numpy.mean(rbm._swapstat.value[:rbm.n_beta.value-1]))
        logger.log('nbeta', n, rbm.n_beta.value)
        logger.log('rtime2', n, rbm.avg_rtime.value/est_rtime)

        if rbm.n_beta.value > 1:
            fup = rbm.get_fup()
            new_betas = rbm._beta.value[rbm._mixstat.value[0,:]][:rbm.n_beta.value]
            plot_fup(new_betas, fup, n)
            plot_swap_stats(new_betas, rbm, n)

            # log graph of inverse temperatures
            for i, b in enumerate(betas):
                beta_log[i].append(b)
            plot_betas(beta_log)

            print '   fup : ', fup
            print '   betas : ', new_betas
            print '   swapstat : ', rbm._swapstat.value[:rbm.n_beta.value]
            print '   tau: ', rbm.get_tau()


    def save_model(postfix=''):
        # save both the averaged and non-averaged model
        fname = 'model%s.pkl' % postfix
        rbm.save(fname=fname)
        fname = 'polyak_model%s.pkl' % postfix
        polyak.save(fname=fname)

    train_time = 0
    spawn_time = 0

    for i in range(n_burnin):
        sample_rbm()

    if rbm.n_beta.value > 1:
        betas = rbm.get_betas()
    
    log_stuff(0, 0.)
    if not debug: 
        log_ais(0, 0.)

    ### START LEARNING ###
    for n in xrange(state.nupdates):

        if walltime and train_time > walltime:
            print '***** REACHED WALLTIME. GIVING UP *****'
            break

        t1 = cpu_time()
        t2 = time.time()

        # retrieve next data point
        (x,y) = t.next_batch()
        x *= 1.0

        ##### BEGIN TRAINING PHASE ####
        # train and optionally burn-in markov chain
        train_rbm(x)
        import pdb; pdb.set_trace()
        model_averaging()
        import pdb; pdb.set_trace()

        for i in range(train_burnin):
            sample_rbm()

        # check to see if we need to spawn more chains
        if state.spawn_beta and \
           rbm.check_spawn(min_p_acc, spawn_type=spawn_type):

            pos = rbm.get_spawn_loc()
            success = rbm.spawn_beta(pos)

            # if spawned successfully ...
            if success:
                beta_log.insert(pos+1, [])
                # mandatory burn-in period to recover from spawn
                _t = cpu_time()
                for i in range(spawn_burnin):
                    betas = rbm.grad_update_betas(beta_lr * get_anneal_coef())
                    sample_rbm()
                spawn_time += cpu_time() - _t
            else:
                # memory allocation failed, prevent further spawning
                state.spawn_beta = False

        # adapt temperatures
        if state.adapt_beta:
            betas = rbm.grad_update_betas(beta_lr * get_anneal_coef())
        ##### END TRAINING PHASE ####

        train_time += cpu_time() - t1

        if (logref=='time' and _log_interval_.next(train_time)) or \
           (logref=='updates' and _log_interval_.next(n)):
            if not debug: 
                log_ais(n+1, train_time)
            log_stuff(n+1, train_time)
            channel.save()

        # log timing info
        state.train_time = train_time
        state.spawn_time = spawn_time
        state.running_time = cpu_time() - start_time

    end_time = cpu_time()

    ###### SAVE STUFF BEFORE EXITING #####
    pickle.dump(beta_log, open('beta_log.pkl','w'), protocol=pickle.HIGHEST_PROTOCOL)
    logger.close()

    if run_mode == 'profile':
        profmode.print_summary()

    # log running time
    state.run_time = end_time - start_time
    print '**** Total running time is %fs ****' % state.run_time
    state.train_time = train_time
    print '**** Total training time is %fs ****' % state.train_time
    state.spawn_time = spawn_time
    print '**** Total spawn burn-in time is %fs ****' % state.spawn_time

    return channel.COMPLETE

