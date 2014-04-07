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
            'repetition_mfcc',
            'repetition_chroma',
            'vq',
            'segments']

#-- Feature analysis guts
def get_sync_features(lowlevel, frames):
    '''Compute synchronized features relative to a list of frames'''

    mfcc    = librosa.feature.sync(lowlevel['mfcc'],            frames) 
    melspec = librosa.feature.sync(lowlevel['mel_spectrogram'], frames)
    cqt     = librosa.feature.sync(lowlevel['cqt'],             frames, aggregate=np.median)
    chroma  = librosa.feature.sync(lowlevel['chroma'],          frames, aggregate=np.median)

    return mfcc, melspec, cqt, chroma

def get_beat_features(duration, beat_times):
    '''Compute timing features for synchronous analysis'''
    
    # The "beat times" correspond to the timings which arise from using beat-synchronous feature aggregation
    # This will usually stick a phantom 0 on the beginning of the beat time
    beats       = np.unique(np.concatenate([ [0.0], beat_times ]))
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
    good_rows = L.any(axis=1)
    if sum(good_rows) >= n_factors:
        L = L[good_rows]

    return compress_features(L, n_factors)

def delta_features(lowlevel):
    '''Log-mel power delta features'''

    M0 = librosa.logamplitude(lowlevel['mel_spectrogram'])
    M1 = librosa.feature.delta(M0)
    M2 = librosa.feature.delta(M1)

    return np.vstack([M0, M1, M2])

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
        b_k, c_k = __get_k_segments(X, k)
        boundaries.append(b_k)
        costs.append(c_k)
            
    best_idx = np.argmin(costs)

    return boundaries, best_idx

def get_segment_range(duration, min_seg, max_seg):
    '''Get the range of reasonable values for the segment tree pruning'''

    k_min = max(1, np.floor(float(duration) / max_seg))
    k_max = max(2, np.ceil(float(duration) / min_seg))

    return int(k_min), int(k_max)

def get_segment_features(analysis, lowlevel, transformation_path):
    '''Construct the feature matrix for segmentation'''
    
    # Trim out any trailing beat features
    bf = get_beat_features(lowlevel['duration'], analysis['beat_times'])

    X = np.vstack([ analysis['beat_sync_mfcc'], 
                    analysis['repetition_mfcc'],
                    analysis['repetition_chroma'],
                    bf ])

    if transformation_path is None:
        return X

    W = np.load(transformation_path)
    return W.dot(X)

#-- Vector quantizer guts
def encoder_model(model_path, n_quantizers):

    with open(model_path, 'r') as f:
        data = pickle.load(f)


    whitener    = data['transformer']
    encoder     = data['encoder']
    parameters  = data['args']
    parameters['n_quantizers'] = n_quantizers

    encoder.n_quantizers = n_quantizers

    return whitener, encoder, parameters

def encode_features(features, whitener, encoder):
    return encoder.transform(whitener.transform(features.T)).T

#-- Full feature extractor
def analyze_features(input_file, features=None, analysis=None, PARAMETERS=None):
    '''Mid-level feature analysis'''

    with open(input_file, 'r') as f:
        lowlevel = pickle.load(f)

    if analysis is None:
        analysis = {}
    
    if features is None:
        features = set(get_feature_names())


    # Beats might occur after the last hop
    # We'll clip anything that's too big
    beat_frames = librosa.time_to_frames(lowlevel['beat_times'],
                                            sr=lowlevel['PARAMETERS']['load']['sr'],
                                            hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])

    beat_frames = np.clip(beat_frames, 0, lowlevel['mfcc'].shape[1]-1)

    # Pad on a phantom 0 here
    beat_frames = np.unique(np.concatenate([[0], beat_frames]))

    analysis['beat_times'] = librosa.frames_to_time(beat_frames, 
                                                    sr=lowlevel['PARAMETERS']['load']['sr'],
                                                    hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])

    # Compute beat-sync features
    if 'beat_sync' in features:
        (analysis['beat_sync_mfcc'], 
         analysis['beat_sync_mel_spectrogram'], 
         analysis['beat_sync_cqt'], 
         analysis['beat_sync_chroma']) = get_sync_features(lowlevel, beat_frames)
                                                
                                                
    
    onset_frames = librosa.time_to_frames(lowlevel['onsets'],
                                          sr=lowlevel['PARAMETERS']['load']['sr'],
                                          hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])

    onset_frames = np.clip(onset_frames, 0, lowlevel['mfcc'].shape[1]-1)
    onset_frames = np.unique(np.concatenate([[0], onset_frames]))

    analysis['onset_times'] = librosa.frames_to_time(onset_frames, 
                                                    sr=lowlevel['PARAMETERS']['load']['sr'],
                                                    hop_length=lowlevel['PARAMETERS']['stft']['hop_length'])
    # Compute onset-sync features
    if 'onset_sync' in features:

        (analysis['onset_sync_mfcc'], 
         analysis['onset_sync_mel_spectrogram'], 
         analysis['onset_sync_cqt'], 
         analysis['onset_sync_chroma']) = get_sync_features(lowlevel, onset_frames)


    if 'repetition_mfcc' in features:
        analysis['repetition_mfcc'] = get_repetition_features(analysis['beat_sync_mfcc'], 
                                                              PARAMETERS['repetition']['mfcc']['n_history'],
                                                              PARAMETERS['repetition']['mfcc']['metric'],
                                                              PARAMETERS['repetition']['mfcc']['width'],
                                                              PARAMETERS['repetition']['mfcc']['kernel_size'],
                                                              PARAMETERS['repetition']['mfcc']['n_factors'])

    if 'repetition_chroma' in features:
        analysis['repetition_chroma'] = get_repetition_features(analysis['beat_sync_chroma'], 
                                                              PARAMETERS['repetition']['chroma']['n_history'],
                                                              PARAMETERS['repetition']['chroma']['metric'],
                                                              PARAMETERS['repetition']['chroma']['width'],
                                                              PARAMETERS['repetition']['chroma']['kernel_size'],
                                                              PARAMETERS['repetition']['chroma']['n_factors'])
    if 'beat_neighbors' in features:
        analysis['mfcc_neighbors_beat']   = get_neighbors(analysis['beat_sync_mfcc'], 
                                                          PARAMETERS['beat_neighbors']['k'],
                                                          PARAMETERS['repetition']['mfcc']['width'],
                                                          PARAMETERS['repetition']['mfcc']['metric'])

        analysis['chroma_neighbors_beat'] = get_neighbors(analysis['beat_sync_chroma'], 
                                                          PARAMETERS['beat_neighbors']['k'],
                                                          PARAMETERS['repetition']['chroma']['width'],
                                                          PARAMETERS['repetition']['chroma']['metric'])

    if 'segments' in features:
        # Get the min and max number of segments
        k_min, k_max = get_segment_range(lowlevel['duration'], 
                                         PARAMETERS['segments']['min_seg'], 
                                         PARAMETERS['segments']['max_seg'])

        # Build the feature stack
        X_segment = get_segment_features(analysis, lowlevel, PARAMETERS['segments']['transformation'])

        # Get the segment boundaries for each k in the range
        segment_boundaries, analysis['segments_best'] = get_segments(X_segment, k_min, k_max)

        # Convert back to boundary times
        analysis['segment_time_tree']  = []
        analysis['segment_beat_tree'] = []

        # Pad the beat times so that we include all points of aggregation
        beat_times = np.unique(np.concatenate([analysis['beat_times'], [lowlevel['duration']]]))

        for level, bounds in enumerate(segment_boundaries):
            analysis['segment_beat_tree'].append(bounds)
            analysis['segment_time_tree'].append(beat_times[bounds])

        # Just to make it easy, copy over the best segmentation
        analysis['segment_times'] = analysis['segment_time_tree'][analysis['segments_best']]

    if 'vq' in features:
        # Load the transformer
        whitener, encoder, args     = encoder_model(PARAMETERS['encoder']['transformation'], 
                                                    PARAMETERS['encoder']['n_quantizers'])

        lmdeltas                    = delta_features(lowlevel)
        analysis['frame_vq']        = encode_features(lmdeltas, whitener, encoder)
        analysis['vq_parameters']   = args
        dense_code                  = analysis['frame_vq'].toarray().astype(np.float32)
        analysis['onset_sync_vq']   = librosa.feature.sync(dense_code, onset_frames).astype(np.float32)
        analysis['beat_sync_vq']    = librosa.feature.sync(dense_code, beat_frames).astype(np.float32)
        analysis['track_vq']        = np.mean(dense_code, axis=1).astype(np.float32)

        
    # Construct a dense representation for summarization purposes


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
