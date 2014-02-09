#!/usr/bin/env python

# CREATED:2014-01-02 13:07:49 by Brian McFee <brm2132@columbia.edu>
# revise/recompute existing feature analysis 

import argparse
import cPickle as pickle
import json
import os
import sys
from joblib import Parallel, delayed

import gordon

# Pull in the audio analyzer
from seymour_analyzers import librosa_lowlevel

ANALYSIS_NAME   = librosa_lowlevel.__description__

#-- interface guts
def get_output_file(path, ext='pickle'):

    drop_ext = os.path.splitext(path)[0]

    return os.path.extsep.join([drop_ext, ext])

def process_arguments():

    parser = argparse.ArgumentParser(description='Librosa-Gordon feature analysis updater')

    parser.add_argument(    'feature_directory',
                            action  =   'store',
                            help    =   'directory to store feature output')

    parser.add_argument(    'features',
                            nargs   =   '+',
                            action  =   'store',
                            type    =   str,
                            choices =   librosa_lowlevel.get_feature_names(),
                            help    =   'features to update')

    parser.add_argument(    '-p',
                            '--parameters',
                            dest    =   'parameter_path',
                            action  =   'store',
                            type    =   str,
                            default =   './parameters-lowlevel.json',
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

    return vars(parser.parse_args(sys.argv[1:]))

def load_parameters(parameter_path):
    '''Load parameters from a json object'''
    with open(parameter_path, 'r') as f:
        PARAMETERS = json.load(f)

    return PARAMETERS

def load_analysis(analysis_file):

    with open(analysis_file, 'r') as f:
        analysis = pickle.load(f)

    return analysis

def save_analysis(track_path, analysis, feature_directory):

    output_file = os.path.join(feature_directory, get_output_file(track_path))
    try:
        os.makedirs(os.path.dirname(output_file))
    except:
        pass

    with open(output_file, 'w') as f:
        pickle.dump(analysis, f, protocol=-1)

    return output_file

def update_annotation(track_id, feature_directory, features, PARAMETERS):
    
    track = gordon.Track.query.get(track_id)

    output_file = os.path.join(feature_directory, get_output_file(track.path))

    if not os.path.exists(output_file) or ANALYSIS_NAME not in track.annotation_dict:
        print 'No existing analysis for ', track_id
        return track_id, output_file

    print 'Updating ', track.path
    
    analysis = load_analysis(output_file)
    analysis = librosa_lowlevel.analyze_audio(track.fn_audio, 
                                              features=set(features), 
                                              analysis=analysis, 
                                              PARAMETERS=PARAMETERS)

    filename = save_analysis(track.path, analysis, feature_directory)
    return track_id, filename

def create_annotation_record(track_id, filename):

    track = gordon.Track.query.get(track_id)

    if ANALYSIS_NAME not in track.annotation_dict:
        track.annotations.append(gordon.Annotation(name=unicode(ANALYSIS_NAME), value=unicode(filename)))
        gordon.commit()
    
def main(feature_directory=None, parameter_path=None, num_jobs=None, verbose=1, features=None):
    
    PARAMETERS = load_parameters(parameter_path)

    def producer():
        tracks = gordon.Track.query.all()
        for track in tracks:
            yield track

    for (track_id, filename) in Parallel(n_jobs=num_jobs, verbose=verbose)(
        delayed(update_annotation)(track.id, feature_directory, features, PARAMETERS) for track in producer()):
        pass

if __name__ == '__main__':
    args = process_arguments()
    main(**args)
