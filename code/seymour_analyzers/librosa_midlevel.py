#!/usr/bin/env python

# CREATED:2014-02-08 13:11:54 by Brian McFee <brm2132@columbia.edu>
# core librosa feature extractor to integrate with gordon/seymour

import numpy as np
import scipy.signal
import librosa
import cPickle as pickle
import os
import git

__version__     = '0.1-dev'
__description__ = 'librosa:mid-level'

def get_git_revision(filename):
    '''Walk up a path name until we find a git dir, then grab its current revision.
    
    If the file is not under any git repository, returns ''.
    '''
    
    filename = os.path.realpath(filename)

    while filename != '/':
        try:
            g = git.Git(filename)
            rev = g.log(['-n 1', '--format=%H'])
            return rev
        except:
            filename = os.path.dirname(filename)
    return ''

# Environment settings
ENVIRONMENT = {'librosa':    {'version':      librosa.__version__,
                              'git-revision': get_git_revision(librosa.__file__),
                              'timestamp':    os.path.getmtime(librosa.__file__)
                              },
                'analyzer':  {'version':        __version__,
                              'git-revision':   get_git_revision(__file__),
                              'timestamp':      os.path.getmtime(__file__)
                },
               }

#-- Audio analysis guts
def get_feature_names():
    '''Construct a list of the low-level analyzer feature names'''
    return ['beat_sync', 
            'onset_sync', 
            'loudness',
            'beat_neighbors',
            'onset_neighbors',
            'segment_times']

def get_sync_features(lowlevel, frames):
    '''Compute synchronized features relative to a list of frames'''

    mfcc    = librosa.feature.sync(lowlevel['mfcc'],            frames) 
    melspec = librosa.feature.sync(lowlevel['mel_spectrogram'], frames)
    cqt     = librosa.feature.sync(lowlevel['cqt'],             frames, aggregate=np.median)
    chroma  = librosa.feature.sync(lowlevel['chroma'],          frames, aggregate=np.median)

    return mfcc, melspec, cqt, chroma


def get_beat_features(lowlevel):
    '''Compute timing features for synchronous analysis'''
    
    beats       = np.unique(np.concatenate([ [0.0], lowlevel['beat_times'] ]))
    duration    = lowlevel['duration']
    beat_idx    = np.arange(float(len(beats)))

    return np.vstack([  beats, 
                        beats / duration, 
                        beat_idx, 
                        beat_idx / len(beat_idx)])

def analyze_features(input_file, features=None, analysis=None, PARAMETERS=None):
    '''Mid-level feature analysis'''

    with open(input_file, 'r') as f:
        lowlevel = pickle.load(f)

    if analysis is None:
        analysis = {}
    
    if features is None:
        features = set(get_feature_names())

    # Compute beat-sync features
    beat_frames = librosa.frames_to_time(lowlevel['beat_times'],
                                         sr=lowlevel['PARAMETERS']['load']['sr'],
                                         hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])
    if 'beat_sync' in features:
        (analysis['beat_sync_mfcc'], 
         analysis['beat_sync_mel_spectrogram'], 
         analysis['beat_sync_cqt'], 
         analysis['beat_sync_chroma']) = get_sync_features(lowlevel, beat_frames)
                                                
                                                
    
    # Compute onset-sync features
    onset_frames = librosa.frames_to_time(lowlevel['onsets'],
                                         sr=lowlevel['PARAMETERS']['load']['sr'],
                                         hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])

    if 'onset_sync' in features:
        (analysis['beat_sync_mfcc'], 
         analysis['beat_sync_mel_spectrogram'], 
         analysis['beat_sync_cqt'], 
         analysis['beat_sync_chroma']) = get_sync_features(lowlevel, onset_frames)
                                                


    PREV = analysis.get('PREVIOUS', {})

    if 'computed_features' in analysis:
        PREV['computed_features'] = analysis['computed_features']

    analysis['computed_features'] = features

    if 'PARAMETERS' in analysis:
        analysis['PREVIOUS'] = {'PARAMETERS':   analysis['PARAMETERS'],
                                'ENVIRONMENT':  analysis['ENVIRONMENT'],
                                'PREVIOUS':     PREV}

    # We're done with harmonics now
    analysis['PARAMETERS']  = PARAMETERS
    analysis['ENVIRONMENT'] = ENVIRONMENT

    return analysis
