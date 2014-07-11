#!/usr/bin/python
"""Utilities to facilitate out-of-core learning in sklearn"""

import numpy as np
import scipy.sparse
import sklearn
import random

def fit(estimator, data_sequence, batch_size=100, max_steps=None):
    '''Fit a model to a generator stream.
    
    :parameters:
      - estimator : sklearn.base.BaseEstimator
        The model object.  Must implement ``partial_fit()``
      
      - data_sequence : generator
        A generator that yields samples
    
      - batch_size : int
        Maximum number of samples to buffer before updating the model
      
      - max_steps : int or None
        If ``None``, run until the stream is exhausted.
        Otherwise, run until at most ``max_steps`` examples have been processed.
    '''
    
    # Is this a supervised or unsupervised learner?
    supervised = isinstance(estimator, sklearn.base.ClassifierMixin)
    
    # Does the learner support partial fit?
    assert(hasattr(estimator, 'partial_fit'))
    
    def _matrixify(data):
        """Determine whether the data is sparse or not, act accordingly"""

        if scipy.sparse.issparse(data[0]):
            n = len(data)
            d = np.prod(data[0].shape)
    
            data_s = scipy.sparse.lil_matrix((n, d), dtype=data[0].dtype)
    
            for i in range(len(data)):
                idx = data[i].indices
                data_s[i, idx] = data[i][:, idx]

            return data_s.tocsr()
        else:
            return np.asarray(data)

    def _run(data, supervised):
        """Wrapper function to partial_fit()"""

        if supervised:
            args = map(_matrixify, zip(*data))
        else:
            args = [_matrixify(data)]

        estimator.partial_fit(*args)
            
    buf = []
    for i, x_new in enumerate(data_sequence):
        buf.append(x_new)
        
        # We've run too far, stop
        if max_steps is not None and i > max_steps:
            break
        
        # Buffer is full, do an update
        if len(buf) == batch_size:
            _run(buf, supervised)
            buf = []
    
    # Update on whatever's left over
    if len(buf) > 0:
        _run(buf, supervised)

def mux(generators):
    '''Randomly multiplex over a list of generators until they are all exhausted.
    
    :parameters:
      - generators : list
        List of generators to randomly multiplex
        
    :yields:
      - iterates from each of the input generators
    '''
    
    # Loop until the generators list is empty
    while len(generators) > 0:
        # Pick a random generator
        i = random.randint(0, len(generators) - 1)
        
        try:
            # Pull from the i'th generator
            yield generators[i].next()
            
        except StopIteration:
            # This one's done.  Remove it from the list.
            generators.pop(i)

            # TODO:   2014-07-03 17:18:36 by Brian McFee <brm2132@columbia.edu>
            # get a new generator here 


def mux_bounded(sample_generator, containers, working_size=10, max_iter=1, shuffle=True, **kwargs):
    '''Generate a sequence by multiplexing a generator function over subsets of data.
    
    1. Select a subset of ``working_size`` from containers
    2. Apply ``sample_generator`` to each item ``c`` in the subset: ``sample_generator(c, **kwargs)``
    3. Multiplex the results of each generator
    
    :parameters:
      - sample_generator : function
        A generator which takes as input a container (see below), and yields samples
      - containers: list-like
        A list of container objects
      - working_size : int > 0
        The maximum number of working containers to keep active at any time
      - max_iter : int > 0
        How many passes through the container set to take
      - shuffle : boolean
        Permute the containers at each iterations
      - kwargs : additional keyword arguments
        Parameters to pass throuh to sample_generator
      
    :yields:
      - sequence of items generated by multiplexing ``sample_generator()``
    '''
    
    idx = range(len(containers))
    
    for _ in range(max_iter):
        
        if shuffle:
            random.shuffle(idx)
            
        for i in range(0, len(idx), working_size):
            # Instantiate a new set of generators
            generators = [sample_generator(containers[j], **kwargs) for j in idx[i:i+working_size]]
    
            # Exhaust the generators
            for x in mux(generators):
                yield x

