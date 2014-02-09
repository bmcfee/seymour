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

def get_feature_names():
    '''Construct a list of the low-level analyzer feature names'''
    return ['beat_sync', 
            'onset_sync', 
            'loudness',
            'beat_neighbors',
            'onset_neighbors',
            'repetition_mfcc',
            'repetition_chroma',
            'segment_times']

#-- Feature analysis guts
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

def get_neighbors(X, k, width, metric):
    '''Get the k-nearest-neighbors for each column of X, not counting the immediate vicinity'''
    
    k = min(k, X.shape[1])

    R = librosa.segment.recurrence_matrix(X, k=k, width=width, metric=metric)

    return R.argsort(axis=1)[:, -k:]

def compress_features(X, k):
    '''Compress the columns of X down to k dimensions'''
    
    sigma = np.cov(X)
    e_vals, e_vecs = scipy.linalg.eig(sigma)
        
    e_vals = np.maximum(0.0, np.real(e_vals))
    e_vecs = np.real(e_vecs)
        
    idx = np.argsort(e_vals)[::-1]
        
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]
    
    # Truncate to k dimensions
    if k < len(e_vals):
        e_vals = e_vals[:k]
        e_vecs = e_vecs[:, :k]
    else:
        # Pad out to k dimensions
        e_vals = np.pad(e_vals, (0, k - len(e_vals)), mode='constant')
        e_vecs = np.pad(e_vecs, [(0, 0), (0, k - e_vecs.shape[1])], mode='constant')
        
    # Normalize by the leading singular value of X
    Z = np.sqrt(e_vals.max())
        
    if Z > 0:
        e_vecs = e_vecs / Z
    
    return e_vecs.T.dot(X)

def get_repetition_features(X, n_steps, metric, width, kernel_size, n_factors):
    '''Construct latent repetition features'''

    # First, stack up by history
    X = librosa.segment.stack_memory(X, n_steps=n_steps)

    # Then build the recurrence matrix
    R = librosa.segment.recurrence_matrix(X, 
                                          width=width,
                                          metric=metric)

    # Skew it
    L = librosa.segment.structure_feature(R).astype(np.float32)

    # Filter it
    L = scipy.signal.medfilt2d(L, kernel_size=[1, kernel_size])

    # Discard empty rows
    L = L[L.any(axis=1)]

    return compress_features(L, n_factors)

#-- Segmentation guts
def __gaussian_cost(X):
    '''Return the average log-likelihood of data under a standard normal'''
    
    d, n = X.shape
    
    if n < 2:
        return 0
    
    sigma = np.var(X, axis=1, ddof=1)
    
    cost =  -0.5 * d * n * np.log(2. * np.pi) - 0.5 * (n - 1.) * np.sum(sigma) 
    return cost

def __clustering_cost(X, boundaries):
    '''Compute the cost of a clustering + AIC correction'''

    # Boundaries include beginning and end frames, so k is one less
    k = len(boundaries) - 1
    
    d, n = map(float, X.shape)
    
    # Compute the average log-likelihood of each cluster
    cost = [__gaussian_cost(X[:, start:end]) for (start, end) in zip(boundaries[:-1], 
                                                                    boundaries[1:])]
    
    cost = - 2 * np.sum(cost) / n + 2 * ( d * k )

    return cost

def __get_k_segments(X, k):
    
    # Step 1: run ward
    boundaries = librosa.segment.agglomerative(X, k)
    
    boundaries = np.unique(np.concatenate(([0], boundaries, [X.shape[1]])))
    
    # Step 2: compute cost
    cost = __clustering_cost(X, boundaries)
        
    return boundaries, cost

def get_segments(X, k_min, k_max):
    '''Build segmentations for each value of k between k_min and k_max (inclusive)'''

    boundaries  = []
    costs       = []

    for k in range(k_min, k_max+1):
        b_k, cost_k = __get_k_segments(X, k)
        boundaries.append(b_k)
        costs.append(cost_k)
            
    best_idx = np.argmin(costs)

    return boundaries, best_idx

def get_segment_range(duration, min_seg, max_seg):
    '''Get the range of reasonable values for the segment tree pruning'''

    k_min = max(1, np.floor(float(duration) / max_seg))
    k_max = max(2, np.ceil(float(duration) / min_seg))

    return int(k_min), int(k_max)

def get_segment_features(lowlevel, transformation_path):
    '''Construct the feature matrix for segmentation'''
    
    X = np.vstack([ lowlevel['beat_sync_mfcc'], 
                    lowlevel['repetition_mfcc'],
                    lowlevel['repetition_chroma'],
                    get_beat_features(lowlevel) ])

    if transformation_path is not None:
        W = np.load(transformation_path)
        return W.dot(X)
    
    return X

#-- Full feature extractor
def analyze_features(input_file, features=None, analysis=None, PARAMETERS=None):
    '''Mid-level feature analysis'''

    with open(input_file, 'r') as f:
        lowlevel = pickle.load(f)

    if analysis is None:
        analysis = {}
    
    if features is None:
        features = set(get_feature_names())

    # Compute beat-sync features
    if 'beat_sync' in features:
        beat_frames = librosa.frames_to_time(lowlevel['beat_times'],
                                             sr=lowlevel['PARAMETERS']['load']['sr'],
                                             hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])
        (analysis['beat_sync_mfcc'], 
         analysis['beat_sync_mel_spectrogram'], 
         analysis['beat_sync_cqt'], 
         analysis['beat_sync_chroma']) = get_sync_features(lowlevel, beat_frames)
                                                
                                                
    
    # Compute onset-sync features
    if 'onset_sync' in features:
        onset_frames = librosa.frames_to_time(lowlevel['onsets'],
                                             sr=lowlevel['PARAMETERS']['load']['sr'],
                                             hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])

        (analysis['beat_sync_mfcc'], 
         analysis['beat_sync_mel_spectrogram'], 
         analysis['beat_sync_cqt'], 
         analysis['beat_sync_chroma']) = get_sync_features(lowlevel, onset_frames)


    if 'repetition_mfcc' in features:
        analysis['repetition_mfcc'] = get_repetition_features(analysis['beat_sync_mfcc'], 
                                                              PARAMETERS['repetition_mfcc']['n_history'],
                                                              PARAMETERS['repetition_mfcc']['metric'],
                                                              PARAMETERS['repetition_mfcc']['kernel_size'],
                                                              PARAMETERS['repetition_mfcc']['n_factors'])

    if 'repetition_chroma' in features:
        analysis['repetition_chroma'] = get_repetition_features(analysis['beat_sync_chroma'], 
                                                              PARAMETERS['repetition_chroma']['n_history'],
                                                              PARAMETERS['repetition_chroma']['metric'],
                                                              PARAMETERS['repetition_chroma']['kernel_size'],
                                                              PARAMETERS['repetition_chroma']['n_factors'])
    if 'beat_neighbors' in features:
        analysis['mfcc_neighbors_beat']   = get_neighbors(analysis['beat_sync_mfcc'], 
                                                          PARAMETERS['beat_neighbors']['mfcc']['k'],
                                                          PARAMETERS['beat_neighbors']['mfcc']['width'],
                                                          PARAMETERS['beat_neighbors']['mfcc']['metric'])

        analysis['chroma_neighbors_beat'] = get_neighbors(analysis['beat_sync_chroma'], 
                                                          PARAMETERS['beat_neighbors']['chroma']['k'],
                                                          PARAMETERS['beat_neighbors']['chroma']['width'],
                                                          PARAMETERS['beat_neighbors']['chroma']['metric'])

    if 'segments' in features:
        # Get the min and max number of segments
        k_min, k_max = get_segment_range(lowlevel['duration'], 
                                         PARAMETERS['segments']['min_seg'], 
                                         PARAMETERS['segments']['max_seg'])

        # Build the feature stack
        X_segment = get_segment_features(lowlevel, PARAMETERS['segments']['transformation'])

        # Get the segment boundaries for each k in the range
        segment_boundaries, analysis['segments_best'] = get_segments(X_segment, k_min, k_max)

        # Convert back to boundary times
        analysis['segment_tree'] = []
        for level, bounds in enumerate(segment_boundaries):
            analysis['segment_tree'].append(lowlevel['beat_times'][bounds])

        # Just to make it easy, copy over the best segmentation
        analysis['segments'] = analysis['segment_tree'][analysis['segments_best']]


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
