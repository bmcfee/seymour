#!/usr/bin/env python

# CREATED:2014-02-09 19:07:14 by Brian McFee <brm2132@columbia.edu>
# intermediate librosa feature analysis: structure

import argparse
import cPickle as pickle
import json
import os
import sys
from joblib import Parallel, delayed

import gordon

# Pull in the audio analyzer
from seymour_analyzers import librosa_midlevel
from seymour_analyzers import librosa_lowlevel

ANALYSIS_NAME = librosa_midlevel.__description__

#-- interface guts
def get_output_file(path, ext='pickle'):

    drop_ext = os.path.splitext(path)[0]

    return os.path.extsep.join([drop_ext, ext])

def process_arguments():

    parser = argparse.ArgumentParser(description='Librosa-Gordon intermediate feature analysis')

    parser.add_argument(    'feature_directory',
                            action  =   'store',
                            help    =   'directory to store feature output')

    parser.add_argument(    '-p',
                            '--parameters',
                            dest    =   'parameter_path',
                            action  =   'store',
                            type    =   str,
                            default =   './parameters-midlevel.json',
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

    if librosa_lowlevel.__description__ not in track.annotation_dict:
        print 'Missing low-level analysis for ', track_id
        return track_id, output_file

    print 'Analyzing ', track.path

    # Skip it if we've already got it cached
    if not recompute and ANALYSIS_NAME in track.annotation_dict:
        return track_id, track.annotation_dict[ANALYSIS_NAME]

    analysis = librosa_midlevel.analyze_audio(track.fn_audio, PARAMETERS=PARAMETERS)

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
