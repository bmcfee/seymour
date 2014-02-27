#!/usr/bin/env python
"""Learn a chord HMM model from a gordon collection"""

import argparse
import librosa
import mir_eval
import sys
import numpy as np
import sklearn.metrics
import sklearn.cross_validation
import cPickle as pickle
import seymour

def alphabet_minmaj():
    alphabet = ['N']
    for root in 'ABCDEFG':
        if root in 'CF':
            flats = ['']
        else:
            flats = ['', 'b']
            
        for flat in flats:
            for quality in ['maj', 'min']:
                alphabet.append('%s%s:%s' % (root, flat, quality))
                
    return alphabet

def midlevel_to_beat_chroma(midlevel_file):

    with open(midlevel_file, 'r') as f:
        analysis = pickle.load(f)

    # Get the beat-sync cqt
    cqt = analysis['beat_sync_cqt']

    # Wrap to chroma
    cq_to_chroma = librosa.filters.cq_to_chroma(cqt.shape[0])

    chroma = librosa.util.normalize(cq_to_chroma.dot(cqt))

    # Compute log-chroma
    log_chroma = librosa.logamplitude(chroma, ref_power=chroma.max())

    # Return the beat times
    beat_intervals = np.asarray(zip(analysis['beat_times'][:-1],
                                    analysis['beat_times'][1:]))

    return log_chroma, beat_intervals

def make_training_data(collection):
    '''get training data from a collection'''

    tracks = seymour.get_collection_tracks(collection)

    obs         = []
    times       = []
    labs        = []
    ref_times   = []
    ref_labs    = []
    names       = []

    for t in tracks:
        track = seymour.get_track(t)

        if not ('librosa:mid-level' in track.annotation_dict and 'chords' in track.annotation_dict):
            continue

        # Get the features
        chroma, beat_times = midlevel_to_beat_chroma(track.annotation_dict['librosa:mid-level'])

        # Get the labels
        chord_times, chord_labels = mir_eval.io.load_annotation(track.annotation_dict['chords'])
        chord_labels = mir_eval.chords.reduce_chords(chord_labels, 'minmaj')

        # Propagate chord labels to beat intervals
        beat_labels = librosa.chord.beats_to_chords(beat_times, chord_times, chord_labels)

        obs.append(chroma.T)
        times.append(beat_times)
        labs.append(beat_labels)
        ref_times.append(chord_times)
        ref_labs.append(chord_labels)
        names.append(track.ofilename)

    return obs, times, labs, ref_times, ref_labs, names

def process_arguments(args):

    parser = argparse.ArgumentParser(description='Learn a chord model through seymour')
    
    parser.add_argument('collection',
                        action  =   'store',
                        type    =   unicode,
                        help    =   'Gordon collection to use for training')

    parser.add_argument('model_file',
                        action  =   'store',
                        help    =   'Path to store the model file')

    parser.add_argument('-n', '--num-folds',
                        dest    =   'num_folds',
                        type    =   int,
                        default =   5,
                        help    =   'Number of validation folds')

    parser.add_argument('-p', '--prior',
                        dest    =   'prior',
                        type    =   float,
                        default =   1e-2,
                        help    =   'Covariance prior')

    parser.add_argument('-e', '--emission',
                        dest    =   'emission_model',
                        type    =   str,
                        choices =   ['full', 'spherical', 'diag', 'tied'],
                        default =   'full',
                        help    =   'Gaussian emission model')

    return vars(parser.parse_args(args))

def save_model(chord_hmm, model_file):

    with open(model_file, 'w') as f:
        pickle.dump(chord_hmm, f, protocol=-1)

def test(chord_hmm, obs, labs):

    # Evaluate how well we did
    y_pred = []
    y_true = []
    for o, l in zip(obs, labs):
        for y_t, y_p in zip(l, chord_hmm.predict_chords(o)):
            y_true.append(chord_hmm.chord_to_id_[y_t])
            y_pred.append(chord_hmm.chord_to_id_[y_p])

    return sklearn.metrics.accuracy_score(y_true, y_pred)

def train(alphabet, obs, labs, num_folds=5, emission_model=None, prior=None):

    # Cross-validation
    for fold, (idx_train, idx_test) in enumerate(sklearn.cross_validation.KFold(len(obs), n_folds=num_folds)):

        # Slice the training data
        obs_train   = [obs[i]   for i in idx_train]
        labs_train  = [labs[i]  for i in idx_train]

        chord_hmm = librosa.chord.ChordHMM(alphabet, covariance_type=emission_model, covars_prior=prior)
        chord_hmm.fit(obs_train, labs_train)
        
        print 'Fold: ', fold
        print '\t Train: %.3f' % test(chord_hmm, obs_train, labs_train)

        obs_test    = [obs[i]   for i in idx_test]
        labs_test   = [labs[i]  for i in idx_test]

        print '\t Test:  %.3f' % test(chord_hmm, obs_test, labs_test)

    chord_hmm = librosa.chord.ChordHMM(alphabet, covariance_type=emission_model)
    chord_hmm.fit(obs, labs)
    print 'Full: '
    print '\t Train: %.3f' % test(chord_hmm, obs, labs)
    return chord_hmm

def save_predictions(chord_hmm, obs, times, labs, ref_times, ref_labs, names):

    predictions = []
    for i in range(len(obs)):
        est = chord_hmm.predict_chords(obs[i])
        predictions.append({'name': names[i], 
                            'beat_times': times[i], 
                            'true_labels': labs[i], 
                            'estimated_labels': est})

    data = {'model': chord_hmm, 'predictions': predictions}

    with open('diagnostics.pickle', 'w') as f:
        pickle.dump(data, f)

def build_model(collection=None, model_file=None, num_folds=None, emission_model=None, prior=None):

    # 1: get the training data
    print '[1/3] Building the training data... '
    obs, times, labs, ref_times, ref_labs, names = make_training_data(collection)

    # 2: train the model
    print '[2/3] Training the model... '
    alphabet = alphabet_minmaj()
    
    chord_hmm = train(alphabet, obs, labs, num_folds=num_folds, emission_model=emission_model, prior=prior)

    # 3: save the model
    print '[3/3] Saving the model.'
    save_model(chord_hmm, model_file)

    print '[4/3] Saving diagnostic predictions'
    save_predictions(chord_hmm, obs, times, labs, ref_times, ref_labs, names)

if __name__ == '__main__':
    args = process_arguments(sys.argv[1:])
    
    build_model(**args)
