#!/usr/bin/env python
"""Module to simplify interaction with gordon"""

import mir_eval
import gordon
import cPickle as pickle

def get_collection_tracks(name):
    """Get all the tracks matching the given collection name"""

    C = gordon.Collection.query.filter_by(name=name).limit(1)
    C = C.all()
    if len(C) == 0:
        return []

    return [t.id for t in C[0].tracks]

def get_track(t_id):
    """Get the track object given the id"""
    return gordon.Track.query.get(t_id)

def get_annotation(track_id, annotation='segments', **kwargs):
    '''get annotation data for a given track'''

    track = get_track(track_id)

    annotation_file = track.annotation_dict[annotation]

    return mir_eval.io.load_annotation(annotation_file, **kwargs)

def get_analysis(track_id, analysis='librosa:low-level'):
    '''get the analysis data for a track by id'''
    
    track = get_track(track_id)
    
    analysis_file = track.annotation_dict[analysis]
    
    with open(analysis_file, 'r') as f:
        data = pickle.load(f)
        
    return data


