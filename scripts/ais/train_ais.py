# allows to plot figures without X11 (i.e on cluster)
import matplotlib
matplotlib.use("Agg")

import numpy, pdb, pickle, time, os, shutil, copy, tables
import pylab as pl

import theano
import theano.tensor as T
from theano.printing import Print
from LeDeepNet import trainer
from smlpt import tools, rbm_tools, optimization, tempered_rbm, ais
from jobman import make, make2
from pylearn.io.seriestables import ErrorSeries

def shrdX(value, name):
    return theano.shared(numpy.asarray(value, dtype=theano.config.floatX), name=name)

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
    start_time = time.time()

    # Parse jobman parameters and set default values
    assert hasattr(state, 'lr')
    lr_start  = state.get('lr_start', None)
    beta_lr  = state.get('beta_lr', state.lr.lr0)
    n_burnin = state.get('n_burnin', 1e3)
    momentum = state.get('momentum', [(0,0.0)])
    l1 = state.get('l1', 0.0)
    l2 = state.get('l2', 0.0)
    waste_cutoff = state.get('waste_cutoff', 1e-3)
    train_burnin = state.get('train_burnin', 0)
    walltime = state.get('walltime', None)
    min_p_acc = state.get('min_p_acc', 0.4)
    spawn_type = state.get('spawn_type', 'spawn_avg')
    spawn_rtime_multiplier = state.get('spawn_rtime_multiplier', 1.)
    state.rbm.tau = state.rbm.get('tau', None)
    spawn_burnin = state.get('spawn_burnin', 100)
    ais_runs = state.get('ais_runs', 100)
    polyak_start = state.get('polyak_start', -1)
    debug = state.get('debug', False)
    assert spawn_type in ('spawn_min', 'spawn_avg', 'spawn_rtime')

    # Initialize param iterators
    _lr_ = make(state.lr)
    _compute_nll_ = make(state.compute_nll)
    _log_interval_ = make(state.log_interval)
    _momentum_ = make(state.momentum)

    # make dataset
    dataset = make(state.dataset)

    # make RBM
    rbm = make(state.rbm)

    if state.init_bias_to_mean:
        visb = -numpy.log(1./(numpy.mean(traindata, axis=0) + 1e-5) -1)
        rbm.vbias.value = copy.copy(visb) 

    ### define PT sampling function ###
    samples, sampling_updates = rbm.get_sampling_updates(k=1)
    sample_rbm = theano.function([], [], updates=sampling_updates)

    ###### DEFINE TRAINING FUNCTION ######
    # define gradients for SML-PT
    lr = shrdX(_lr_.value, name='lr')
    momentum = shrdX(_momentum_.value, name='momentum')
    learning_grads, sampling_updates = rbm.get_learning_gradients(k=state.k, 
            l1=l1, l2=l2, waste_cutoff=waste_cutoff)

    # define learning updates
    learning_updates = optimization.sgd(learning_grads, lr, momentum)
    # use polyak averaging 
    polyak = optimization.PolyakAveraging(rbm.params, start=polyak_start)
    polyak_updates = polyak.get_updates(learning_updates)

    # define training function
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)
    updates.update(polyak_updates)
    train_rbm = theano.function([rbm.input], [], updates = updates)
    ###### END TRAINING FUNCTION ######

    # define functions to update temperature
    rbm.config_beta_updates()

    # the trainer takes care of handling the dataset
    t = trainer.Trainer(dataset, batch_size=rbm.batch_size)
    t.set_active_dataset('train', randomize=False)

    # logging of various time series
    logger = tools.HDF5Logger("outputs.h5")
    beta_log = [ [] for i in xrange(rbm.n_beta.value)]
    state.nstamps = ""
    state.tstamps = ""
    state.train_nlls = ""
    state.valid_nlls = ""
    state.test_nlls  = ""

    def log_ais(n, train_time):

        #### compute AIS estimate of logZ ####
        ais_log_z, (lower,upper) = ais.rbm_ais(polyak.param_vals, n_runs=ais_runs,
                                               rng=rbm.numpy_rng)
        rbm.log_Z = ais_log_z
        train_nll = numpy.mean(rbm_tools.compute_nll(rbm, dataset.train.x, skip_partition=True))
        valid_nll = numpy.mean(rbm_tools.compute_nll(rbm, dataset.valid.x, skip_partition=True))
        test_nll  = numpy.mean(rbm_tools.compute_nll(rbm, dataset.test.x, skip_partition=True))

        # log history of likelihood scores in string format
        state.nstamps += "%i :" % n
        state.tstamps += "%.3g :" % train_time
        state.train_nlls += "%.2f :" % train_nll
        state.valid_nlls += "%.2f :" % valid_nll
        state.test_nlls  += "%.2f :" % test_nll

        # store latest value seperately for easy querying
        state.n = n
        state.t = train_time
        state.train_nll = train_nll
        state.valid_nll = valid_nll
        state.test_nll  = test_nll

        logger.log('logz', n, ais_log_z)
        logger.log('logz_std', n, (upper-lower))
        logger.log('train_nll', n, train_nll)
        logger.log('valid_nll', n, valid_nll)
        logger.log('test_nll', n, test_nll)

        print 'update %i/%i:' % (n, state.nupdates)
        print '   NLL=%f log_Z=%f (+/- %f)'  %\
                (numpy.mean(train_nll), ais_log_z, upper-lower)
      
    def log_stuff(n, train_time):

        logger.log('t', n, train_time)
        logger.log('rtime', n, rbm.avg_rtime.value)
        logger.log('nbsize', n, rbm.waste_reduction_nbsize.value)
        logger.log('avgswap', n, numpy.mean(rbm._swapstat.value[:rbm.n_beta.value-1]))
       
        est_rtime = tempered_rbm.rtime_deo(min_p_acc, rbm.n_beta.value)
        logger.log('rtime2', n, rbm.avg_rtime.value/est_rtime)

        print 'update %i/%i:' % (n, state.nupdates)

        if rbm.n_beta.value > 1:
            fup = rbm.get_fup()
            new_betas = rbm._beta.value[rbm._mixstat.value[0,:]][:rbm.n_beta.value]
            plot_fup(new_betas, fup, n+1)
            plot_swap_stats(new_betas, rbm, n+1)
            print '   fup : ', fup
            print '   betas : ', new_betas
            print '   swapstat : ', rbm._swapstat.value[:rbm.n_beta.value]
            print '   tau: ', rbm.get_tau()

    def log_betas():
        for i, b in enumerate(betas):
            beta_log[i].append(b)
    
    def save_model():
        rbm.save(fname='model.pkl')
        polyak.save_and_replace('model.pkl', ofname='polyak_model.pkl')
        os.system('rm model.pkl')

    train_time = 0
    spawn_time = 0

    for i in range(n_burnin):
        sample_rbm()
    betas = rbm.get_betas()
    log_stuff(0, 0.)
    if not debug: log_ais(0, 0.)

    ### START LEARNING ###
    for n in xrange(1.2*state.nupdates):

        if time.time() - start_time > walltime:
            print '***** REACHED WALLTIME. GIVING UP *****'
            break

        if n > state.nupdates: lr.value = 0.0

        # handle hyper-parameters which are on a schedule
        lr.value = _lr_.next()
        momentum.value = _momentum_.next()
        compute_nll = _compute_nll_.next()
        log_interval = _log_interval_.next()

        t1 = time.time()

        # retrieve next data point
        (x,y) = t.next_batch()
        x *= 1.0

        ##### BEGIN TRAINING PHASE ###
        # train and optionally burn-in markov chain
        train_rbm(x)
        for i in range(train_burnin):
            sample_rbm()

        # check to see if we need to spawn more chains
        if state.spawn_beta and \
           rbm.check_spawn(min_p_acc, spawn_type=spawn_type,
                   spawn_rtime_multiplier=spawn_rtime_multiplier):

            pos = rbm.get_spawn_loc()
            success = rbm.spawn_beta(pos)

            # if spawned successfully ...
            if success:
                beta_log.insert(len(beta_log)-1, [])
                # mandatory burn-in period to recover from spawn
                _t = time.time()
                for i in range(spawn_burnin):
                    betas = rbm.grad_update_betas(beta_lr)
                    sample_rbm()
                spawn_time += time.time() - _t
            else:
                # memory allocation failed, prevent further spawning
                state.spawn_beta = False

        # adapt temperatures
        if state.adapt_beta and rbm.n_beta.value > 1:
            betas = rbm.grad_update_betas(beta_lr)
        ##### END TRAINING PHASE ####

        train_time += time.time() - t1

        if (n+1) % compute_nll == 0:
            save_model()
            if not debug: log_ais(n+1, train_time)
            channel.save()
        
        if (n+1) % log_interval == 0:
            save_model()
            log_stuff(n+1, train_time)
            log_betas()
	    plot_betas(beta_log)

        # log timing info
        state.train_time = train_time
        state.spawn_time = spawn_time
        state.running_time = time.time() - start_time

    end_time = time.time()

    ###### SAVE STUFF BEFORE EXITING #####
    save_model()
    log_ais(n+1, train_time)
    log_stuff(n+1, train_time)
    log_betas()
    plot_betas(beta_log)
    pickle.dump(beta_log, open('beta_log.pkl','w'), protocol=pickle.HIGHEST_PROTOCOL)
    logger.close()

    # log running time
    state.run_time = end_time - start_time
    print '**** Total running time is %fs ****' % state.run_time
    state.train_time = train_time
    print '**** Total training time is %fs ****' % state.train_time
    state.spawn_time = spawn_time
    print '**** Total spawn burn-in time is %fs ****' % state.spawn_time

    return channel.COMPLETE

