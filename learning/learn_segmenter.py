#!/usr/bin/env python
# CREATED:2014-02-11 12:07:54 by Brian McFee <brm2132@columbia.edu>
#  Gordon-backed implementation of OLDA learning

import argparse
import sys
from joblib import Parallel, delayed
import mir_eval
import numpy as np

import OLDA
import seymour
import seymour_analyzers.librosa_midlevel as midlevel

def process_arguments(args):
    '''Argparser'''

    parser = argparse.ArgumentParser(description='Segmentation learning for seymour')

    parser.add_argument(    'model_file',
                            action  =   'store',
                            type    =   str,
                            help    =   'path to store the learned model file (.npy)')

    parser.add_argument(    '-j',
                            '--num-jobs',
                            dest    =   'num_jobs',
                            action  =   'store',
                            type    =   int,
                            default =   4,
                            help    =   'number of threads')

    parser.add_argument(    'collections',
                            action  =   'store',
                            type    =   str,
                            nargs   =   '+',
                            help    =   'one or more collections to train from')

    parser.add_argument(    '--sigma-min',
                            action  =   'store',
                            type    =   int,
                            default =   0,
                            dest    =   'sigma_min',
                            help    =   'Minimum power value of the regularization term')

    parser.add_argument(    '--sigma-max',
                            action  =   'store',
                            type    =   int,
                            default =   9,
                            dest    =   'sigma_max',
                            help    =   'Maximum power value of the regularization term')

    return vars(parser.parse_args(args))

def get_track_data(t_id):
    '''Get the data for a given track'''
    
    # Get the ground-truth segments
    segments, labels    = seymour.get_annotation(t_id, annotation='segments')
    
    # Get the mid-level analysis
    analysis_mid        = seymour.get_analysis(t_id, analysis=midlevel.__description__)

    # Flatten segment start/ends to boundaries
    boundary_times      = np.unique(segments.ravel())

    # Get the feature stack
    features            = midlevel.get_segment_features(analysis_mid, 
                                                        {'duration': boundary_times[-1]}, 
                                                        None)

    # Get k_min, k_max for this track
    k_min, k_max        = midlevel.get_segment_range(boundary_times[-1],
                                                     analysis_mid['PARAMETERS']['segments']['min_seg'],
                                                     analysis_mid['PARAMETERS']['segments']['max_seg'])

    boundary_beats      = []
    for time in boundary_times:
        boundary_beats.append(  np.argmin( (boundary_times - time)**2) )

    boundary_beats      = np.unique(boundary_beats)

    return {'features': features,
            'k_min':    k_min,
            'k_max':    k_max,
            'boundary_times': boundary_times,
            'boundary_beats': boundary_beats}
    
def get_training_data(collection_names):
    '''Construct training data from one or more collections'''

    track_ids = []
    for c in collection_names:
        track_ids.extend(seymour.get_collection_tracks(c))

    # Now, for each training example, get the features, timings, and ground-truth labels
    data = {}
    for t_id in track_ids:
        data[t_id] = get_track_data(t_id)

    return data

def learn_segmenter(model_file=None, num_jobs=1, collections=None, sigma_min=0, sigma_max=9):
    '''Outer layer of the segment learner'''

    # Step 1: build the data
    print 'Building training data... '
    train_data = get_training_data(collections)

    # Step 2: learn the model
    print 'Fitting the model... '

    # Step 3: save the results
    print 'Saving to ', model_file

    pass

if __name__ == '__main__':
    parameters = process_arguments(sys.argv[1:])

