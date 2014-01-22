import theano
import theano.tensor as T
import numpy

class BlockGibbsSampler(object):

    def __init__(self, block_updates, n_steps=1):
        """
        :param block_updates: dictionary whose keys are conditionally independent (theano
        shared) variables, and whose values are the update expression to use for block gibbs
        sampling
        :param n_steps: number of block Gibbs steps to perform
        """
        self.block_updates = block_updates
        self.n_steps = n_steps

        self.sample_block = {}
        for i, (k,v) in enumerate(block_updates.iteritems()):
            self.sample_block[k] = theano.function([],[],
                    updates={k:v},allow_input_downcast = False)

    def simulate(self, n_steps=None):
        n_steps = n_steps if n_steps else self.n_steps
        for n in xrange(n_steps):
            for fn in self.sample_block.itervalues():
                fn()

    def get_state(self):
       state = {}
       for v in self.block_updates.iterkeys():
           state[v] = v.value
       return state

    def draw(self, n_steps=None):
        self.simulate(n_steps=n_steps)
        return self.get_state()
