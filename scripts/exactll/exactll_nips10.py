# allows to plot figures without X11 (i.e on cluster)
import matplotlib
matplotlib.use("Agg")

import numpy, pdb, pickle, time, os, shutil, copy, tables
import pylab as pl

import theano
import theano.tensor as T
from theano.printing import Print
from LeDeepNet import trainer
from smlpt import rbm_tools, optimization, tempered_rbm
from jobman import make, make2
from pylearn.io.seriestables import ErrorSeries


def plot_betas(beta_log):

    t_max = max([len(b) for b in beta_log])

    for bi, b in enumerate(beta_log):
        t = numpy.arange(t_max - len(b), t_max)
        pl.plot(t, b)

    pl.savefig('betas.pdf')
    pl.close()

def plot_mode_ids(mode_histogram, mode_w, t, pad=8):
    x = numpy.arange(len(mode_histogram))

    fig = pl.figure()
    sp1 = fig.add_subplot(111)
    sp1.bar(x, mode_histogram, color='b')
    sp1.set_xlabel('mode id')
    sp1.set_ylabel('abs.', color='b')

    y = (mode_histogram/numpy.sum(mode_histogram)) / mode_w

    sp2 = sp1.twinx()
    sp2.bar(len(mode_histogram) + x, y, color='r')
    sp2.set_ylabel('rel.', color='r')
    
    fig_num = str(t)
    fig_num = '0'*(pad-len(fig_num)) + fig_num
    pl.title('Mode ID @ %i'%t)
    pl.savefig('modes_%s.pdf'%fig_num)
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

def copy_params(rbm, rbm_pl):
    # Compute exact partition function using plearn
    rbm_pl.connection.weights = rbm.w.copy()
    rbm_pl.visible_layer.bias = rbm.layers['vis'].b.copy()
    rbm_pl.hidden_layer.bias = rbm.layers['hid'].b.copy()
    rbm_pl.computePartitionFunction()

def make_testset(dataset, size=1e4):
    rval = numpy.zeros((size, dataset.img_size))
    for i in xrange(size):
        rval[i] = dataset.next()[0]
    return rval


def experiment(state, channel):

    n_burnin = state.get('n_burnin', 1e3)
    opt_method = state.get('optimization', 'sgd')
    momentum = state.get('momentum', 0.0)
    l1 = state.get('l1', 0.0)
    l2 = state.get('l2', 0.0)
    lr_method = state.get('lr_method', 'constant')
    is_online = state.get('is_online', False)
    waste_cutoff = state.get('waste_cutoff', 1e-3)
    train_burnin = state.get('train_burnin', 0)
    walltime = state.get('walltime', None)
    min_p_acc = state.get('min_p_acc', 0.4)
    spawn_type = state.get('spawn_type', 'spawn_rtime')
    spawn_rtime_multiplier = state.get('spawn_rtime_multiplier', 1.)
    state.rbm.rtime_a = state.rbm.get('rtime_a', 1)
    state.rbm.rtime_b = state.rbm.get('rtime_b', 100)
    state.rbm.n_rtime = state.rbm.get('n_rtime', 1)
    state.rbm.tau = state.rbm.get('tau', None)
    spawn_burnin = state.get('spawn_burnin', 1)
    beta_lr = state.get('beta_lr', 1)
    assert spawn_type in ('spawn_min', 'spawn_avg', 'spawn_rtime', 'spawn_auc')

    # make dataset
    dataset = make(state.dataset)

    # make RBM
    rbm = make2(state.rbm)
    if not is_online and state.init_bias_to_mean:
        visb = -numpy.log(1./(numpy.mean(dataset.train.x, axis=0) + 1e-5) -1)
        rbm.vbias.value = copy.copy(visb) 

    ### define PT sampling function ###
    samples, sampling_updates = rbm.get_sampling_updates(k=1)
    sample_rbm = theano.function([], [], updates=sampling_updates)

    ### define gradients for SML-PT ###
    lr = theano.shared(numpy.asarray(state.lr, dtype=theano.config.floatX), name='lr')
    momentum = theano.shared(numpy.asarray(momentum, dtype=theano.config.floatX), name='momentum')
    learning_grads, sampling_updates = rbm.get_learning_gradients(k=state.k, 
            l1=l1, l2=l2, waste_cutoff=waste_cutoff)

    ### choose optimization method ###
    if opt_method == 'sgd':
        # deal with optional weight decay parameters #
        learning_updates = optimization.sgd(learning_grads, lr, momentum)
    elif opt_method == 'sgd_polyak':
        learning_updates = optimization.polyak(learning_grads, lr)
    else:
        raise ValueError('Invalid optimization method')

    ### combine all update dictionary into a single training function ###
    updates = {}
    updates.update(sampling_updates)
    updates.update(learning_updates)

    #from theano import ProfileMode
    #profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
    #train_rbm = theano.function([rbm.input], [], updates = updates, mode=profmode)
    train_rbm = theano.function([rbm.input], [], updates = updates)

    # define functions to update temperature
    rbm.config_beta_updates(beta_lr=beta_lr)

    if not is_online:
        # the trainer takes care of handling the dataset
        t = trainer.Trainer(dataset, batch_size=rbm.batch_size)
        t.set_active_dataset('train', randomize=True)

    # logging of various time series
    tables_f = tables.openFile("outputs.h5","w")

    t_series      = ErrorSeries(index_names=('n',), error_name="t",      table_name="t",      hdf5_file=tables_f)
    nll_series    = ErrorSeries(index_names=('n',), error_name="nll",    table_name="nll",    hdf5_file=tables_f)
    test_nll_series = ErrorSeries(index_names=('n',), error_name="test_nll", table_name="test_nll",    hdf5_file=tables_f)
    logz_series   = ErrorSeries(index_names=('n',), error_name="logz",   table_name="logz",   hdf5_file=tables_f)
    rtime_series  = ErrorSeries(index_names=('n',), error_name="rtime",  table_name="rtime",  hdf5_file=tables_f)
    rtime2_series = ErrorSeries(index_names=('n',), error_name="rtime2", table_name="rtime2", hdf5_file=tables_f)
    t1frac_series = ErrorSeries(index_names=('n',), error_name="t1frac", table_name="t1frac", hdf5_file=tables_f)
    swap_series   = ErrorSeries(index_names=('n',), error_name="avgswap", table_name="avgswap", hdf5_file=tables_f)

    mode_series = []
    for i in range(dataset.n_modes):
        mode_series.append(ErrorSeries(index_names=('n',), error_name="mode%i"%i,
                                       table_name="mode%i"%i, hdf5_file=tables_f))

    nll_dataset = make_testset(dataset) if is_online else dataset.train.x

    beta_log = [ [] for i in xrange(rbm.n_beta.value)]
    mode_histogram = numpy.zeros(dataset.n_modes)
    mode_dist = numpy.zeros(dataset.n_modes)

    def log_stuff(n, train_time):
        nll = rbm_tools.compute_nll(rbm, nll_dataset)
        mean_nll = numpy.mean(nll)

        t_series.append([n], train_time)
        nll_series.append([n], mean_nll)
        logz_series.append([n], rbm.log_Z)
        rtime_series.append([n], rbm.avg_rtime.value)
        rtime2_series.append([n], rbm.avg_rtime.value/tempered_rbm.rtime_deo(min_p_acc, rbm.n_beta.value))
        t1frac_series.append([n], rbm.waste_reduction_t1frac.value)
        swap_series.append([n], numpy.mean(rbm._swapstat.value[:rbm.n_beta.value-1]))

        print 'update %i/%i:' % (n, state.nupdates)
        print '   NLL=%f log_Z=%f'  % (mean_nll, rbm.log_Z)

        if not is_online and state.compute_test_nll:
            test_nll = rbm_tools.compute_nll(rbm, dataset.test.x, skip_partition=True)
            test_nll_series.append([n], numpy.mean(test_nll))

        if rbm.n_beta.value > 1:
            fup = rbm.get_fup()
            new_betas = rbm._beta.value[rbm._mixstat.value[0,:]][:rbm.n_beta.value]
            plot_fup(new_betas, fup, n+1)
            plot_swap_stats(new_betas, rbm, n+1)
            print '   fup : ', fup
            print '   betas : ', new_betas
            #print '   optimal : ', rbm.get_optimal_betas()
            print '   swapstat : ', rbm._swapstat.value[:rbm.n_beta.value]
            print '   tau: ', rbm.get_tau()

            if n==0:
                os.system('pdfjoin fup.pdf fup_*')
                os.system('pdfjoin swapstat.pdf swapstats_*')
            else:
                os.system('pdfjoin _fup.pdf fup.pdf fup_*')
                os.system('pdfjoin _swapstat.pdf swapstat.pdf swapstats_*')
                os.system('mv _fup.pdf fup.pdf')
                os.system('mv _swapstat.pdf swapstat.pdf')


        plot_mode_ids(mode_histogram, dataset.w, n+1)
        if n==0:
            os.system('pdfjoin modes.pdf modes_*')
        else:
            os.system('pdfjoin _modes.pdf modes.pdf modes_*')
            os.system('mv _modes.pdf modes.pdf')
        os.system('rm modes_*')


        for i in range(dataset.n_modes):
            mode_series[i].append([n], mode_dist[i])

        log_betas()

    def log_betas():
        for i, b in enumerate(betas):
            beta_log[i].append(b)

    for i in range(n_burnin):
        sample_rbm()
    betas = rbm.grad_update_betas()
    log_stuff(0, 0.)

    start_time = time.time()
    train_time = 0
    spawn_time = 0

    ### START LEARNING ###
    for n in xrange(1.2*state.nupdates):

        if n > state.nupdates: lr.value = 0.0

        t1 = time.time()

        # retrieve next data point
        if is_online:
            x = dataset.next(batch_size=rbm.batch_size)
        else:
            (x,y) = t.next_batch()

        ##### begin training phase ###
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
                beta_log.insert(pos+1, [])
                # mandatory burn-in period to recover from spawn
                #for i in range(spawn_burnin * 1./rbm.get_tau()):
                _t = time.time()
                for i in range(100):
                    betas = rbm.grad_update_betas()
                    sample_rbm()
                spawn_time += time.time() - _t
            else:
                # memory allocation failed, prevent further spawning
                state.spawn_beta = False

        # adapt temperatures
        if state.adapt_beta and rbm.n_beta.value > 1:
            betas = rbm.grad_update_betas()
        ##### end training phase ####

        train_time += time.time() - t1

        if (n+1) % state.compute_nll == 0:
            log_stuff(n+1, train_time)
            mode_histogram.fill(0)
            mode_dist.fill(0)
        else:
            log_betas()

            samples = rbm._nvis.value[rbm._mixstat.value[:,0], :]
            for sample in samples:
                sample = numpy.tile(sample, (dataset.n_modes,1))
                dist = numpy.sum((dataset.modes - sample)**2, axis=1)
                mode_id = numpy.argmin(dist)
                mode_dist[mode_id] = (mode_histogram[mode_id] * mode_dist[mode_id] + dist[mode_id]) / (mode_histogram[mode_id]+1)
                mode_histogram[mode_id] += 1

        if walltime and train_time > walltime:
            print '***** REACHED WALLTIME. GIVING UP *****'
            log_stuff(n+1, train_time)
            break

    tables_f.close()
    
    end_time = time.time()
    state.run_time = end_time - start_time
    print '**** Total running time is %fs ****' % state.run_time
    state.train_time = train_time
    print '**** Total training time is %fs ****' % state.train_time
    state.spawn_time = spawn_time
    print '**** Total spawn burn-in time is %fs ****' % state.spawn_time

    plot_betas(beta_log)
    pickle.dump(beta_log, open('beta_log.pkl','w'), protocol=pickle.HIGHEST_PROTOCOL)

    #profmode.print_summary()
