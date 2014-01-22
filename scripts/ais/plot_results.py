from sqlalchemy.ext.sqlsoup import SqlSoup
from sqlalchemy.sql import text
import pylab as pl
import numpy

TABLE='smlpt_caltech3_view'

db = SqlSoup('postgres://desjagui:cr0quet@gershwin.iro.umontreal.ca/desjagui_db')

# retrieve the various timestamps
rp = db.bind.execute("select loginterval_vals from %(TABLE)s LIMIT 1" % locals())
tstamps = eval("%s" % rp.fetchone()[0])

# retrieve the various "n_beta" values
rp = db.bind.execute("select rbm_nbeta from %(TABLE)s group by rbm_nbeta" % locals())

y_mean = {}
y_std  = {}
data = {}
for row in rp.fetchall():
    y_mean[row['rbm_nbeta']] = []
    y_std[row['rbm_nbeta']]  = []
    data[row['rbm_nbeta']] = [ [] for t in tstamps ]

# retrieve the various timestamps
sqlcmd = text("""
CREATE OR REPLACE TEMP view ll_view as
select
    id,
    cast(regexp_split_to_table(tstamps, ':') as float) as tstamps,
    cast(regexp_split_to_table(validnlls, ':') as float) as validnlls,
    cast(regexp_split_to_table(testnlls, ':') as float) as testnlls
from 
    %(TABLE)s
where 
    tstamps!='' and validnlls!='' and testnlls!='';

CREATE OR REPLACE TEMP view estopper_view as
select 
    id, max(validnlls) as score
from 
    ll_view
where
    tstamps <= :tstamp
group by id;

CREATE OR REPLACE TEMP view avg_seed_view as
select
    rbm_nbeta, lr_lr0, l2, momentum_vals, rbm_tbatchsize, rbm_batchsize, rbm_betalbound, minpacc, betalr,
    count(ll.validnlls) as cnt,
    avg(ll.validnlls) as validnlls, 
    stddev(ll.validnlls)/sqrt(count(ll.validnlls)) as validnlls_std,
    avg(ll.testnlls) as testnlls, 
    stddev(ll.testnlls)/sqrt(count(ll.testnlls)) as testnlls_std
from
    %(TABLE)s, ll_view as ll, estopper_view
where
    ll.validnlls=estopper_view.score and
    ll.id=estopper_view.id and
    %(TABLE)s.id=estopper_view.id
group by
    rbm_nbeta, lr_lr0, l2, momentum_vals, rbm_tbatchsize, rbm_batchsize, rbm_betalbound, minpacc, betalr;

CREATE OR REPLACE TEMP view best_valid_view as
select
    rbm_nbeta, max(validnlls) as max_validnlls
from 
    avg_seed_view
group by
    rbm_nbeta;

select 
    as_view.rbm_nbeta as rbm_nbeta, 
    lr_lr0, betalr, rbm_betalbound, rbm_tbatchsize, minpacc, l2, momentum_vals,
    testnlls, testnlls_std
from
    avg_seed_view as as_view, 
    best_valid_view as bv_view
where
    bv_view.max_validnlls = as_view.validnlls and
    bv_view.rbm_nbeta = as_view.rbm_nbeta;
""" % locals())

hps = ['lr_lr0', 'betalr', 'rbm_betalbound', 'rbm_tbatchsize', 'minpacc', 'l2', 'momentum_vals']

import pickle;

for tstamp in tstamps:
    print 'Fetching results @%i' % tstamp
    rp = db.bind.execute(sqlcmd, tstamp=tstamp)
    for row in rp.fetchall():
        y_mean[row['rbm_nbeta']].append(row['testnlls'])
        if row['testnlls_std']:
            y_std[row['rbm_nbeta']].append(row['testnlls_std'])
        else:
            y_std[row['rbm_nbeta']].append(0)

        for i, hp in enumerate(hps):
            data[row['rbm_nbeta']][i].append( row[hp] if row[hp] is not None else 0 ) 

fig = pl.figure()
xticks = []
for i, tstamp in enumerate(tstamps):
    if i == 0:
        pl.bar(i*25,    -y_mean[1][i],  width=5, yerr=y_std[1][i],  color='r', ecolor='k', label='SML')
        pl.bar(i*25+5,  -y_mean[2][i],  width=5, yerr=y_std[2][i],  color='b', ecolor='k', label='SML-APT')
        #pl.bar(i*25+10, -y_mean[10][i], width=5, yerr=y_std[10][i], color='g', ecolor='k', label='SML-PT 10')
        #pl.bar(i*25+15, -y_mean[50][i], width=5, yerr=y_std[50][i], color='c', ecolor='k', label='SML-PT 50')
    else:
        pl.bar(i*25,    -y_mean[1][i],  width=5, yerr=y_std[1][i],  color='r', ecolor='k')
        pl.bar(i*25+5,  -y_mean[2][i],  width=5, yerr=y_std[2][i],  color='b', ecolor='k')
        #pl.bar(i*25+10, -y_mean[10][i], width=5, yerr=y_std[10][i], color='g', ecolor='k')
        #pl.bar(i*25+15, -y_mean[50][i], width=5, yerr=y_std[50][i], color='c', ecolor='k')
    xticks.append(tstamp)

pl.xticks(numpy.arange(len(tstamps)+1)*25 + 25/2., xticks)

y_min = numpy.floor(numpy.max(y_mean.values())/10.) * 10.
y_max = numpy.floor(numpy.min(y_mean.values())/10.) * 10.
pl.ylim(-y_min-10, -y_max)
#for label in fig.get_xticklabels():
    #label.set_rotation(30)

pl.legend(loc='upper right')
pl.savefig('results.pdf')

######### print best hyperparameters for each point #######

if 1:
    for n_beta, dat in data.iteritems():
        print 'n_beta = %i' % n_beta
        for i, hp in enumerate(hps):
            print '\t%s: \t' % hp,
            for k in data[n_beta][i]:
                print '%.4f ' % k,
            print ''
        print '\n'
