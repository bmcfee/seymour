#!/usr/bin/env python

# CREATED:2014-02-08 13:11:54 by Brian McFee <brm2132@columbia.edu>
# core librosa feature extractor to integrate with gordon/seymour

import numpy as np
import scipy.signal
import librosa
import mutagen
import os
import git

__version__     = '0.1-dev'
__description__ = 'librosa:low-level'

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
def get_hpss(y, PARAMETERS):
    '''Separate harmonic and percussive audio time series'''   
    # Get the STFT
    D = librosa.stft(y, **PARAMETERS['stft'])
    
    
    # Get the HPSS
    D_h, D_p = librosa.decompose.hpss(D, **PARAMETERS['hpss'])

    y_h = librosa.istft(D_h, hop_length=PARAMETERS['stft']['hop_length'])
    y_p = librosa.istft(D_p, hop_length=PARAMETERS['stft']['hop_length'])
    
    return y_h, y_p

def get_melspec(y, PARAMETERS):
    '''Get a mel power spectrogram'''   
    S = librosa.feature.melspectrogram(y=y, 
                                       sr=PARAMETERS['load']['sr'], 
                                       n_fft=PARAMETERS['stft']['n_fft'], 
                                       hop_length=PARAMETERS['stft']['hop_length'], 
                                       n_mels=PARAMETERS['mel']['n_mels'], 
                                       fmax=PARAMETERS['mel']['fmax']).astype(np.float32)
    
    return S
    
def get_mfcc(S, PARAMETERS):
    '''Get MFCCs from the mel power spectrogram'''
    M = librosa.feature.mfcc(S=librosa.logamplitude(S, ref_power=np.abs(S).max()), 
                             n_mfcc=PARAMETERS['mfcc']['n_mfcc'])
    return M

def get_tuning(y, PARAMETERS):
    '''Estimate tuning'''
    return librosa.feature.estimate_tuning( y=y, 
                                            sr=PARAMETERS['load']['sr'], 
                                            n_fft=PARAMETERS['stft']['n_fft'])

def get_beat(y, PARAMETERS):
    '''Estimate beat times and tempo'''
    # Compute a log-power mel spectrogram on the percussive component
    S_p = librosa.feature.melspectrogram(y=y, 
                                         sr=PARAMETERS['load']['sr'], 
                                         n_fft=PARAMETERS['stft']['n_fft'], 
                                         hop_length=PARAMETERS['beat']['hop_length'],
                                         n_mels=PARAMETERS['mel']['n_mels'],
                                         fmax=PARAMETERS['mel']['fmax'])
    
    S_p = librosa.logamplitude(S_p, ref_power=S_p.max())
    
    # Compute the median onset aggregation
    odf = librosa.onset.onset_strength(S=S_p, aggregate=np.median)
    
    # Get beats
    tempo, beats = librosa.beat.beat_track(onsets=odf, 
                                           sr=PARAMETERS['load']['sr'], 
                                           hop_length=PARAMETERS['beat']['hop_length'])
      
    beat_times = librosa.frames_to_time(beats, 
                                        sr=PARAMETERS['load']['sr'], 
                                        hop_length=PARAMETERS['beat']['hop_length'])
    
    return tempo, beat_times, odf
    
def get_chroma(y, PARAMETERS):
    '''STFT-chromagram'''
    C = librosa.feature.chromagram(y=y, 
                                   sr=PARAMETERS['load']['sr'], 
                                   hop_length=PARAMETERS['stft']['hop_length'], 
                                   **PARAMETERS['chroma']).astype(np.float32)
    
    return C
    
def get_cqt(y, PARAMETERS):
    '''Constant-Q transform, energy-only'''
    CQT = np.abs(librosa.cqt(y, 
                      sr=PARAMETERS['load']['sr'],
                      hop_length=PARAMETERS['stft']['hop_length'], 
                      **PARAMETERS['cqt']))
    
    return CQT

def get_feature_names():
    '''Construct a list of the low-level analyzer feature names'''
    return ['duration', 
            'signal', 
            'mel_spectrogram', 
            'mfcc', 
            'beats', 
            'tuning', 
            'chroma', 
            'cqt']

def analyze_audio(input_file, features=None, analysis=None, PARAMETERS=None):
    '''Full audio analysis'''

    if analysis is None:
        analysis = {}
    
    if features is None:
        features = set(get_feature_names())

    # Import metadata, if we can
    try:
        analysis['metadata'] = dict(mutagen.File(input_file, easy=True))
    except:
        analysis['metadata'] = {}
    
    # Load the input file
    y, sr = librosa.load(input_file, **PARAMETERS['load'])
    
    # Compute track duration
    if 'duration' in features:
        analysis['duration'] = float(len(y)) / sr
    
    # Pre-compute a downsampled time series for vis purposes
    if 'signal' in features:
        analysis['signal']   = scipy.signal.decimate(y, len(y)/1024).astype(np.float32)
    
    # Get a mel power spectrogram
    if 'mel_spectrogram' in features:
        analysis['mel_spectrogram'] = get_melspec(y, PARAMETERS)
    
    # Get the mfcc's
    if 'mfcc' in features:
        analysis['mfcc'] = get_mfcc(analysis['mel_spectrogram'], PARAMETERS)

    # Do HPSS
    y_h, y_p = get_hpss(y, PARAMETERS)
    
    # We're done with raw audio
    del y
    
    # Get beats and tempo
    if 'beats' in features:
        analysis['tempo'], analysis['beat_times'], analysis['onset_strength'] = get_beat(y_p, PARAMETERS)
    
    # We're done with percussion now
    del y_p
    
    # Get tuning
    if 'tuning' in features:
        analysis['tuning'] = get_tuning(y_h, PARAMETERS)
    
    # Get the chroma
    if 'chroma' in features:
        analysis['chroma'] = get_chroma(y_h, PARAMETERS)
    
    # Get the CQT
    if 'cqt' in features:
        analysis['cqt']    = get_cqt(y_h, PARAMETERS)
    
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
