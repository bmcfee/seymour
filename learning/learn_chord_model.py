#!/usr/bin/env python
"""Learn a chord HMM model from a gordon collection"""

import argparse
import librosa
import mir_eval
import sys
import numpy as np
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

    chroma = librosa.util.normalize(cq_to_chroma.dot(cqt**2))

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
    
    for t in tracks:
        track = seymour.get_track(t)

        if not ('librosa:mid_level' in track.annotation_dict and
            'chords' in track.annotation_dict):
            continue

        # Get the features
        chroma, beat_times = midlevel_to_beat_chroma(track.annotation_dict['librosa:mid_level'])

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

    return obs, times, labs, ref_times, ref_labs

def process_arguments(args):

    parser = argparse.ArgumentParser(description='Learn a chord model through seymour')
    
    parser.add_argument('collection',
                        action  =   'store',
                        help    =   'Gordon collection to use for training')

    parser.add_argument('model_file',
                        action  =   'store',
                        help    =   'Path to store the model file')

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



def build_model(collection=None, model_file=None, emission_model=None):

    # 1: get the training data
    print '[1/3] Building the training data... '
    obs, times, labs, ref_times, ref_labs = make_training_data(collection)

    # 2: train the model
    print '[2/3] Training the model... '
    alphabet = alphabet_minmaj()
    
    chord_hmm = librosa.chord.ChordHMM(alphabet, covariance_type=emission_model)
    chord_hmm.train(obs, labs)

    # 3: save the model
    print '[3/3] Saving the model.'
    save_model(chord_hmm, model_file)


if __name__ == '__main__':
    args = process_arguments(sys.argv[1:])
    
    build_model(**args)
