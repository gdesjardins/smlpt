import os

import numpy
import pylab as pl

def plotmat(w, img_shape, grid_size=(10.,10.), normalize='global',
             fname='weights', postfix='', title='', 
             interpolation=None, eps=1e-8):
    """
    Function to plot data contained as 2D rasterized images (i.e weights).
    Can be used from python or using LeDeepNet/bin/plotweights executable.
    """
    assert normalize in ('global','filters','clip')
    bw = 3 # border width

    # determine number of figures per image
    ngrid = numpy.prod(grid_size)
    nfig = int(numpy.ceil(w.shape[0] / ngrid))

    # declare array which will ngrid subplots
    bigimg = numpy.zeros(((img_shape[0]+bw)*grid_size[0]+bw,
                        (img_shape[1]+bw)*grid_size[1]+bw))

    if normalize=='global':
        # configure luminance range over all w matrix
        (vmin,vmax) = (numpy.min(w), numpy.max(w))
    elif normalize=='clip':
        # clip everything to the [-1,1] interval
        (vmin,vmax) = (-1.0, 1.0)
        w = numpy.clip(w, vmin, vmax)
    elif normalize=='filters':
        # normalize each filter individually
        (vmin,vmax) = (0,1)

    fig = pl.figure()

    prefix = os.tmpnam()

    for n in range(nfig):
        (x,y) = (1,1)
        bigimg.fill(0.5)

        for i in range(grid_size[0]):
            x = 1

            for j in range(grid_size[1]):

                windex = n*ngrid + i*grid_size[1] + j
                if windex <= w.shape[0]:
                    w_i = w[windex,:]

                    if normalize=='filter':
                        w_i -= numpy.min(w_i)
                        w_i *= 1.0 / (numpy.max(w_i) + eps)

                    bigimg[y:y+img_shape[0],x:x+img_shape[1]] = w_i.reshape(img_shape)

                x += img_shape[1] + bw

            y += img_shape[0] + bw

        # do some renormalization
        temp_fname = prefix + '_fig%i_%s.pdf'%(n,postfix)
        pl.gray(); pl.axis('off'); pl.title(title);
        pl.imshow(bigimg, vmin=vmin, vmax=vmax, interpolation=interpolation)
        fig.savefig(temp_fname)
        pl.close(fig)
        fig = pl.figure()

    pl.close(fig)

    os.system('pdfjoin %s.pdf %s' % (fname,prefix+'_*'))
    os.system('rm %s' % prefix+'_*')
