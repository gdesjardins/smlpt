from sqlalchemy.ext.sqlsoup import SqlSoup
from sqlalchemy.sql import text
import pylab as pl
import numpy

TABLE='smlpt_caltech1_view'

db = SqlSoup('postgres://desjagui:cr0quet@gershwin.iro.umontreal.ca/desjagui_db')

# retrieve the various timestamps
sqlcmd = text("""
CREATE OR REPLACE TEMP view ll_view as
select
    id,
    cast(regexp_split_to_table(validnlls, ':') as float) as validnlls
from 
    smlpt_caltech1_view
where 
    rbm_nbeta=2;

CREATE OR REPLACE TEMP view estopper_view as
select 
    id, max(validnlls) as score
from 
    ll_view
group by id;

CREATE OR REPLACE TEMP view avg_seed_view as
select
    rbm_nbeta, lr_lr0, l2, momentum_vals, rbm_tbatchsize, rbm_batchsize, rbm_betalbound,
    minpacc, betalr, ll.validnlls
from
    smlpt_caltech1_view, ll_view as ll, estopper_view
where
    ll.validnlls=estopper_view.score and
    ll.id=estopper_view.id and
    smlpt_caltech1_view.id=estopper_view.id;
""" % locals())
rp = db.bind.execute(sqlcmd)

sqltxt = """
select 
    $hp , avg(validnlls) as mean, stddev(validnlls) as std
from
    avg_seed_view as as_view
group by $hp
order by $hp asc;
""" % locals()

colors = ['b','g','r','c','m','y','k']

hps = ['lr_lr0', 'betalr', 'rbm_betalbound', 'rbm_tbatchsize', 'minpacc', 'l2', 'momentum_vals']

fig = pl.figure()

subfig1 = pl.subplot(1, len(hps), 1)

for i, hp in enumerate(hps):
    print 'Fetching results for hyper-parameter: %s' % hp
    sqlcmd = text(sqltxt.replace('$hp', hp))
    rp = db.bind.execute(sqlcmd)

    subfig = pl.subplot(1, len(hps), i+1, sharey=subfig1)

    # remove y-ticks for all but the first subplot
    if i==0:
        pl.ylabel('Test Likelihood')
    else:
        pl.setp(subfig.get_yticklabels(), visible=False)
    pl.xlabel(hp)

    xticks = []
    width = 0.35

    for j, row in enumerate(rp.fetchall()):
        subfig.bar(j*width, row['mean'], width=width, color=colors[j], yerr=row['std'], ecolor='k')
        xticks.append(row[hp] if row[hp] is not None else 0)

    pl.xticks(numpy.arange(j+1)*width + width/2., xticks)
    for label in subfig.get_xticklabels():
        label.set_rotation(30)

pl.savefig('hyperparams.pdf')
pl.close()
