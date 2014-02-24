#!/usr/bin/env python
# CREATED:2014-02-11 12:07:54 by Brian McFee <brm2132@columbia.edu>
#  Gordon-backed implementation of OLDA learning

import librosa
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

    # Construct beat intervals
    beat_intervals = np.asarray(zip(analysis_mid['beat_times'][:-1], 
                                    analysis_mid['beat_times'][1:]))

    # Match beat intervals to segments
    beat_segment_ids = librosa.util.match_intervals(beat_intervals, segments)
        
    boundary_beats      = []
    for i, seg in enumerate(segments):
        # Find the first beat that maps to this segment
        hits = np.argwhere(beat_segment_ids == i)
        if len(hits) > 0:
            boundary_beats.append( int(hits[0]) )

    boundary_beats = np.unique(boundary_beats)

    return {'features': features,
            'k_min':    k_min,
            'k_max':    k_max,
            'beat_times':     analysis_mid['beat_times'],
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

def score_example(W, features=None, k_min=None, k_max=None, beat_times=None, boundary_times=None, boundary_beats=None):
    '''Compute the boundary detection F-measure score for an example'''

    # Do the full prediction
    seg_predict_tree, best_idx = midlevel.get_segments(W.dot(features), k_min, k_max)

    # Make sure the beat times cover the entire range, pad
    beat_times = np.unique(np.concatenate([beat_times, [boundary_times[-1]]]))

    # Tack on the track duration to the end
    predicted_times = beat_times[seg_predict_tree[best_idx]]

    # Convert to intervals
    ref_intervals = mir_eval.util.boundaries_to_intervals(boundary_times)[0]
    est_intervals = mir_eval.util.boundaries_to_intervals(predicted_times)[0]

    est_intervals = mir_eval.util.adjust_intervals(est_intervals, 
                                                   t_min=0.0,
                                                   t_max=ref_intervals[-1, -1])[0]

    # Compute the score, pull off the f-measure
    score = mir_eval.segment.boundary_detection(ref_intervals, est_intervals)[-1]
    return score

def fit_model(train_data, num_jobs=1, sigma_min=0, sigma_max=9):
    '''Learning code for segmentation'''

    # Stack the features into a list
    features            = []
    boundary_beats      = []

    for t_id in train_data:
        features.append(train_data[t_id]['features'])
        boundary_beats.append(train_data[t_id]['boundary_beats'])

    # Parameter sweep range
    sigma_values = 10.0**np.arange(sigma_min, sigma_max+1)

    best_score  =   -np.inf
    best_sigma  =   None
    best_model  =   None

    for sigma in sigma_values:
        # Construct the model
        olda_model = OLDA.OLDA(sigma=sigma)

        # Fit the model
        olda_model.fit(features, boundary_beats)

        # Evaluate on the training set
        scores = Parallel(n_jobs=num_jobs)(delayed(score_example)(olda_model.components_, **example) 
                                            for _, example in train_data.iteritems())

        mean_score = np.mean(scores)
        print '\tsigma=%.2e, score=%.3f' % (sigma, mean_score)

        if mean_score > best_score:
            best_score  = mean_score
            best_sigma  = sigma
            best_model  = olda_model.components_

    print 'Best sigma: %.2e' % best_sigma
    return best_model

def learn_segmenter(model_file=None, num_jobs=1, collections=None, sigma_min=0, sigma_max=9):
    '''Outer layer of the segment learner'''

    # Step 1: build the data
    print 'Building training data... '
    train_data = get_training_data(collections)

    # Step 2: learn the model
    print 'Fitting the model on %d tracks ' % len(train_data)
    model   =   fit_model(train_data, num_jobs=num_jobs, sigma_min=sigma_min, sigma_max=sigma_max)

    # Step 3: save the results
    print 'Saving to ', model_file
    np.save(model_file, model)


if __name__ == '__main__':
    parameters = process_arguments(sys.argv[1:])

    learn_segmenter(**parameters)
