#!/usr/bin/env python
"""Learn a whitening transformation and VQ codebook from a gordon stream"""

# Core dependencies
import argparse, sys
import numpy as np
import cPickle as pickle

import seymour
import seymour_analyzers.librosa_lowlevel as lowlevel
import seymour_analyzers.librosa_midlevel as midlevel

# Learn OOC is the glue that allows sequential learning (sklearn) from generator functions
# Whitening, Hartigan, and VQ are all sklearn modules for whitening transformations, 
# online clustering, and vector quantization
import pescador
from seymour_analyzers.Whitening          import Whitening
from seymour_analyzers.HartiganOnline     import HartiganOnline
from seymour_analyzers.VectorQuantizer    import VectorQuantizer


def feature_stream(track_id, n=50, transform=None):
    """The datastream object's feature generator. 
    Takes a track id, and generates samples of its feature content"""
    
    # Get the features
    features = midlevel.delta_features(seymour.get_analysis(track_id, lowlevel.__description__))
    
    if transform is not None:
        features = transform.transform(features.T).T
    
    for i in range(n):
        t = np.random.randint(0, features.shape[1])
        yield features[:, t]

def get_parameters():

    parser = argparse.ArgumentParser(description='Gordon VQ codebook learning')

    parser.add_argument('collection',
                        action  =   'store',
                        help    =   'Gordon collection to use for training')

    parser.add_argument('output_file',
                        action  =   'store',
                        help    =   'Path to save the model')
    
    parser.add_argument('-k',
                        '--num-codewords',
                        type    =   int,
                        default =   1024,
                        dest    =   'n_codewords',
                        help    =   'Size of the codebook')

    parser.add_argument('-w',
                        '--working-size',
                        type    =   int,
                        default =   64,
                        dest    =   'working_size',
                        help    =   'Maximum number of files to process at once')

    parser.add_argument('-m',
                        '--max-iter',
                        type    =   int,
                        default =   1,
                        dest    =   'max_iter',
                        help    =   'Number of passes through the data stream')

    parser.add_argument('-n',
                        '--num-samples',
                        type    =   int,
                        default =   64,
                        dest    =   'n_samples',
                        help    =   'Number of samples to draw (with replacement) from each file')

    parser.add_argument('-b',
                        '--buffer-size',
                        type    =   int,
                        default =   8192,
                        dest    =   'buffer_size',
                        help    =   'Number of samples to buffer for learning') 

    return vars(parser.parse_args(sys.argv[1:]))

def save_results(outfile, transformer, encoder, args):
    """Save the result objects to the specified output file"""

    with open(outfile, 'w') as f:
        pickle.dump({'transformer': transformer,
                     'encoder': encoder,
                     'args': args}, f, protocol=-1)

def learn_codebook(collection, n_codewords, working_size, max_iter, n_samples, buffer_size):
    """Learn the feature transformation"""

    # Get the collection's tracks
    tracks = seymour.get_collection_tracks(collection)

    print 'Learning from collection [%s], %d tracks' % (collection, len(tracks))

    print 'Learning the feature scaling... '
    # Create a data stream to learn a whitening transformer
    seeds = [pescador.Streamer(feature_stream, t) for t in tracks]
    mux_stream = pescador.mux(seeds, max_iter, working_size, lam=n_samples)

    # Build the whitening transform
    transformer = pescador.StreamLearner(Whitening(), batch_size=buffer_size)
    transformer.iter_fit(mux_stream)

    print 'Learning the codebook... '
    # Create a new data stream that uses the whitener prior to running k-means
    # This could also be done with a sklearn.pipeline, probably?
    seeds = [pescador.Streamer(feature_stream, t, transform=transformer) for t in tracks]
    mux_stream = pescador.mux(seeds, max_iter, working_size, lam=n_samples)

    # Build the codebook estimator. 
    encoder_ = VectorQuantizer(clusterer=HartiganOnline(n_clusters=n_codewords))
    encoder = pescador.StreamLearner(encoder_, batch_size=buffer_size)
    encoder.iter_fit(mux_stream)
    
    return transformer, encoder

if __name__ == '__main__':
    args = get_parameters()
    transformer, encoder = learn_codebook(args['collection'], 
                                          n_codewords   = args['n_codewords'], 
                                          working_size  = args['working_size'],
                                          max_iter      = args['max_iter'],
                                          n_samples     = args['n_samples'],
                                          buffer_size   = args['buffer_size'])

    save_results(args['output_file'], transformer, encoder, args)
