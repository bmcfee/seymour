#!/usr/bin/env python

# CREATED:2013-12-19 19:33:28 by Brian McFee <brm2132@columbia.edu>
# core librosa feature extractor to integrate with gordon 

import numpy as np
import scipy.signal
import librosa
import mutagen
import os
import git

import json
import cPickle as pickle

from joblib import Parallel, delayed

import sys
import argparse

import gordon

__version__     = '0.1-dev'
ANALYSIS_NAME   = 'librosa:low-level'

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
def analyze_hpss(y, PARAMETERS):
    '''Separate harmonic and percussive audio time series'''   
    # Get the STFT
    D = librosa.stft(y, **PARAMETERS['stft'])
    
    
    # Get the HPSS
    D_h, D_p = librosa.decompose.hpss(D, **PARAMETERS['hpss'])

    y_h = librosa.istft(D_h, hop_length=PARAMETERS['stft']['hop_length'])
    y_p = librosa.istft(D_p, hop_length=PARAMETERS['stft']['hop_length'])
    
    return y_h, y_p

def analyze_melspec(y, PARAMETERS):
    '''Get a mel power spectrogram'''   
    S = librosa.feature.melspectrogram(y=y, 
                                       sr=PARAMETERS['load']['sr'], 
                                       n_fft=PARAMETERS['stft']['n_fft'], 
                                       hop_length=PARAMETERS['stft']['hop_length'], 
                                       n_mels=PARAMETERS['mel']['n_mels'], 
                                       fmax=PARAMETERS['mel']['fmax']).astype(np.float32)
    
    return S
    
def analyze_mfcc(S, PARAMETERS):
    '''Get MFCCs from the mel power spectrogram'''
    M = librosa.feature.mfcc(S=librosa.logamplitude(S, ref_power=np.abs(S).max()), 
                             n_mfcc=PARAMETERS['mfcc']['n_mfcc'])
    return M

def analyze_tuning(y, PARAMETERS):
    '''Estimate tuning'''
    return librosa.feature.estimate_tuning( y=y, 
                                            sr=PARAMETERS['load']['sr'], 
                                            n_fft=PARAMETERS['stft']['n_fft'])

def analyze_beat(y, PARAMETERS):
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
    
def analyze_chroma(y, PARAMETERS):
    '''STFT-chromagram'''
    C = librosa.feature.chromagram(y=y, 
                                   sr=PARAMETERS['load']['sr'], 
                                   hop_length=PARAMETERS['stft']['hop_length'], 
                                   **PARAMETERS['chroma']).astype(np.float32)
    
    return C
    
def analyze_cqt(y, PARAMETERS):
    '''Constant-Q transform, energy-only'''
    CQT = np.abs(librosa.cqt(y, 
                      sr=PARAMETERS['load']['sr'],
                      hop_length=PARAMETERS['stft']['hop_length'], 
                      **PARAMETERS['cqt']))
    
    return CQT

def analyze_audio(input_file, PARAMETERS):
    '''Full audio analysis'''
    analysis = {}
    
    # Import metadata, if we can
    try:
        analysis['metadata'] = dict(mutagen.File(input_file, easy=True))
    except:
        analysis['metadata'] = {}
    
    # Load the input file
    y, sr = librosa.load(input_file, **PARAMETERS['load'])
    
    # Compute track duration
    analysis['duration'] = float(len(y)) / sr
    
    # Pre-compute a downsampled time series for vis purposes
    analysis['signal']   = scipy.signal.decimate(y, 
                                                 len(y)/1024, 
                                                 ftype='fir').astype(np.float32)
    
    # Get a mel power spectrogram
    analysis['mel_spectrogram'] = analyze_melspec(y, PARAMETERS)
    
    # Get the mfcc's
    analysis['mfcc'] = analyze_mfcc(analysis['mel_spectrogram'], PARAMETERS)

    # Do HPSS
    y_h, y_p = analyze_hpss(y, PARAMETERS)
    
    # We're done with raw audio
    del y
    
    # Get beats and tempo
    analysis['tempo'], analysis['beat_times'], analysis['onset_strength'] = analyze_beat(y_p, PARAMETERS)
    
    # We're done with percussion now
    del y_p
    
    # Get tuning
    analysis['tuning'] = analyze_tuning(y_h, PARAMETERS)
    
    # Get the chroma
    analysis['chroma'] = analyze_chroma(y_h, PARAMETERS)
    
    # Get the CQT
    analysis['cqt']    = analyze_cqt(y_h, PARAMETERS)
    
    # We're done with harmonics now
    analysis['PARAMETERS'] = PARAMETERS
    analysis['ENVIRONMENT'] = ENVIRONMENT
    return analysis

#-- interface guts
def get_output_file(path, ext='pickle'):

    drop_ext = os.path.splitext(path)[0]

    return os.path.extsep.join([drop_ext, ext])

def process_arguments():

    parser = argparse.ArgumentParser(description='Librosa-Gordon feature analysis')

    parser.add_argument(    'feature_directory',
                            action  =   'store',
                            help    =   'directory to store feature output')

    parser.add_argument(    '-p',
                            '--parameters',
                            dest    =   'parameter_path',
                            action  =   'store',
                            type    =   str,
                            default =   './parameters.json',
                            help    =   'path to parameters json object')

    parser.add_argument(    '-j',
                            '--num-jobs',
                            dest    =   'num_jobs',
                            action  =   'store',
                            type    =   int,
                            default =   2,
                            help    =   'number of parallel jobs')

    parser.add_argument(    '-v',
                            '--verbose',
                            dest    =   'verbose',
                            action  =   'store',
                            type    =   int,
                            default =   1,
                            help    =   'verbosity')

    parser.add_argument(    '-r',
                            '--recompute',
                            dest    =   'recompute',
                            action  =   'store_true',
                            default =   False,
                            help    =   'Force recompute features')

    return vars(parser.parse_args(sys.argv[1:]))

def load_parameters(parameter_path):
    '''Load parameters from a json object'''
    with open(parameter_path, 'r') as f:
        PARAMETERS = json.load(f)

    return PARAMETERS

def save_analysis(track_path, analysis, feature_directory):

    output_file = os.path.join(feature_directory, get_output_file(track_path))
    try:
        os.makedirs(os.path.dirname(output_file))
    except:
        pass

    with open(output_file, 'w') as f:
        pickle.dump(analysis, f, protocol=-1)

    return output_file

def create_annotation(track_id, feature_directory, recompute, PARAMETERS):
    
    track = gordon.Track.query.get(track_id)

    output_file = os.path.join(feature_directory, get_output_file(track.path))

    if not recompute and os.path.exists(output_file):
        print 'Pre-computed: ', output_file
        return track_id, output_file

    print 'Analyzing ', track.path

    # Skip it if we've already got it cached
    if ANALYSIS_NAME in track.annotation_dict:
        return track_id, track.annotation_dict[ANALYSIS_NAME]

    analysis = analyze_audio(track.fn_audio, PARAMETERS)

    filename = save_analysis(track.path, analysis, feature_directory)
    return track_id, filename

def create_annotation_record(track_id, filename):

    track = gordon.Track.query.get(track_id)

    if ANALYSIS_NAME not in track.annotation_dict:
        track.annotations.append(gordon.Annotation(name=unicode(ANALYSIS_NAME), value=unicode(filename)))
        gordon.commit()
    
def main(feature_directory=None, parameter_path=None, num_jobs=None, verbose=1, recompute=False):
    
    PARAMETERS = load_parameters(parameter_path)

    def producer():
        tracks = gordon.Track.query.all()
        for track in tracks:
            yield track

    for (track_id, filename) in Parallel(n_jobs=num_jobs, verbose=verbose)(
        delayed(create_annotation)(track.id, feature_directory, recompute, PARAMETERS) for track in producer()):
        
        create_annotation_record(track_id, filename)

if __name__ == '__main__':
    args = process_arguments()
    main(**args)
