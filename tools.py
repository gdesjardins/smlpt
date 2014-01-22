import numpy

import theano
from theano import tensor as T

 
class HyperParamIterator():

    def __init__(self, vals):
        self.i = 0 
        self.n = 0
        if not isinstance(vals, list):
            self.vals = [(0,vals)]
        else:
            self.vals = vals

        self.value = self.vals[self.i][1]

    def next(self):
        self.n += 1

        if self.i+1 != len(self.vals) and self.n > self.vals[self.i+1][0]:
            self.i += 1
            self.value = self.vals[self.i][1]

        return self.value

    def __iterator__(self):
        return self

class EventScheduler():

    def __init__(self, vals):
        self.i = 0 
        if not isinstance(vals, list):
            self.vals = [vals]
        else:
            self.vals = vals

        self.is_event = False
        self.value = None

    def next(self, timestamp):

        if self.i < len(self.vals) and timestamp >= self.vals[self.i]:
            self.value = self.vals[self.i]
            self.i += 1
            self.is_event = True
        else:
            self.is_event = False

        return self.is_event

    def __iterator__(self):
        return self

class DelayedOneOverT():
    """
    Implements a delayed 1/t schedule. We maintain a fixed learning rate lr0 for the first
    "start" iterations, and then decrease it using a factor of "start/t". Using start=None
    yields a constant learning rate.
    """

    def __init__(self, lr0, start):
        self.lr0 = lr0
        self.start = start
        self.n = 0
        self.value = lr0

    def next(self):
        self.n += 1
        if self.start:
            self.value = self.lr0 * min(1, self.start/(self.n+1.))
        else:
            self.value = self.lr0
        return self.value
    
    def __iterator__(self):
        return self

import tables
from pylearn.io.seriestables import ErrorSeries

class HDF5Logger():
    def __init__(self, fname):
        self.fname = fname
        self.fp = tables.openFile(fname, "w") 
        self.entries = {}

    def log(self, name, index, value, index_names=('n',)):
        if not self.entries.has_key(name):
           self.entries[name] = ErrorSeries(index_names=index_names, 
                   error_name=name, table_name=name, hdf5_file=self.fp)

        self.entries[name].append([index], value)

    def close(self):
        self.fp.close()


