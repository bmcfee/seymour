#!/usr/bin/env python
"""Learn a whitening transformation and VQ codebook from a gordon stream"""

# Core dependencies
import argparse, sys
import numpy as np
import scipy
import cPickle as pickle

import librosa
import seymour

# Learn OOC is the glue that allows sequential learning (sklearn) from generator functions
# Whitening, Hartigan, and VQ are all sklearn modules for whitening transformations, 
# online clustering, and vector quantization
import learn_ooc
from Whitening import Whitening
from HartiganOnline import HartiganOnline
from VectorQuantizer import VectorQuantizer

def extract_features(track_id, widths=[3,7,15]):
    '''Get local contrast spectrogram features from a track id'''

    # Build the local difference mapping
    def make_contrast_features(X, widths=widths):
        """helper function to build contrast features"""
        Xhat = X
    
        for w in widths:
            Xhat = np.vstack((Xhat, scipy.signal.lfilter(np.ones(w) / w, 1, X, axis=-1)))
        
        return Xhat
    
    # Get the mel spectrogram, convert to log power
    S = seymour.get_analysis(track_id)['mel_spectrogram']
    S = librosa.logamplitude(S, ref_power=S.max())
    
    return make_contrast_features(S)

def feature_stream(track_id, n=50, transform=None):
    """The datastream object's feature generator. 
    Takes a track id, and generates samples of its feature content"""
    
    # Get the features
    features = extract_features(track_id)
    
    if transform is not None:
        features = transform.transform(features.T).T
    
    for i in range(n):
        t = np.random.randint(0, features.shape[1])
        yield features[:, t]

def learn_codebook(collection, n_codewords=2048, working_size=256, max_iter=1, n_samples=64, buffer_size=8192):
    """Learn the feature transformation"""

    # Get the collection's tracks
    tracks = seymour.get_collection_tracks(collection)

    print 'Learning from collection [%s], %d tracks' % (collection, len(tracks))

    print 'Learning the feature scaling... '
    # Create a data stream to learn a whitening transformer
    data_stream = learn_ooc.mux_bounded(feature_stream, 
                                        [t for t in tracks], 
                                        working_size=working_size, 
                                        max_iter=max_iter, 
                                        n=n_samples)

    # Build the whitening transform
    transformer = Whitening()
    learn_ooc.fit(transformer, data_stream, batch_size=buffer_size)

    print 'Learning the codebook... '
    # Create a new data stream that uses the whitener prior to running k-means
    # This could also be done with a sklearn.pipeline, probably?
    data_stream = learn_ooc.mux_bounded(feature_stream, 
                                        [t for t in tracks], 
                                        working_size=working_size, 
                                        max_iter=max_iter, 
                                        n=n_samples, 
                                        transform=transformer)

    # Build the codebook estimator. 
    encoder = VectorQuantizer(clusterer=HartiganOnline(n_clusters=n_codewords))
    learn_ooc.fit(encoder, data_stream, batch_size=buffer_size)

    return transformer, encoder

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

if __name__ == '__main__':
    args = get_parameters()
    transformer, encoder = learn_codebook(args['collection'], 
                                          n_codewords   = args['n_codewords'], 
                                          working_size  = args['working_size'],
                                          max_iter      = args['max_iter'],
                                          n_samples     = args['n_samples'],
                                          buffer_size   = args['buffer_size'])

    save_results(args['output_file'], transformer, encoder, args)
