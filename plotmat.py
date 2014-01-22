import os
import numpy

def plot_mat(w, img_shape, grid_size=(10.,10.), normalize='filter',
             save_dir='weights', fname='weights',
             postfix='', title='', interpolation=None):
    """
    Function to plot data contained as 2D rasterized images (i.e weights).
    Can be used from python or using LeDeepNet/bin/plotweights executable.
    """
    assert normalize in ('global','filter')
    bw = 1 # border width

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # determine number of figures per image
    ngrid = N.prod(grid_size)
    nfig = int(numpy.ceil(w.shape[0] / ngrid))

    # declare array which will ngrid subplots
    bigimg = numpy.zeros(((img_shape[0]+bw)*grid_size[0]+bw,
                        (img_shape[1]+bw)*grid_size[1]+bw))
    # configure luminance range over all w matrix
    if normalize=='global':
        (vmin,vmax) = (numpy.min(w), numpy.max(w))
    else:
        (vmin,vmax) = (0,1)

    fig = pl.figure()

    for n in range(nfig):
        (x,y) = (1,1)
        bigimg.fill(0.5)

        for i in range(grid_size[0]):
            x = 1

            for j in range(grid_size[1]):

                windex = n*ngrid + i*grid_size[1] + j
                if windex <= w.shape[0]:
                    w_i = w[windex,:].reshape(img_shape)

                    if normalize=='filter':
                        w_i -= numpy.expand_dims(numpy.min(w_i, axis=1), 1)
                        w_i = w_i / numpy.expand_dims(numpy.max(w_i, axis=1), 1)

                    bigimg[y:y+img_shape[0],x:x+img_shape[1]] = w_i

                x += img_shape[1] + bw

            y += img_shape[0] + bw

        # do some renormalization
        temp_fname = os.path.join(save_dir,'%s%i%s.pdf'%(fname,n,postfix))
        pl.gray(); pl.axis('off'); pl.title(title);
        pl.imshow(bigimg, vmin=vmin, vmax=vmax, interpolation=interpolation)
        fig.savefig(temp_fname)
        pl.close(fig)
        fig = pl.figure()

    pl.close(fig)
