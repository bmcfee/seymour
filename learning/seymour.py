#!/usr/bin/env python
"""Module to simplify interaction with gordon"""

import gordon
import cPickle as pickle

def get_collection_tracks(name):
    """Get all the tracks matching the given collection name"""

    C = gordon.Collection.query.filter(gordon.Collection.name==name).limit(1)
    C = C.all()
    if len(C) == 0:
        return []

    return [t.id for t in C[0].tracks]

def get_track(t_id):
    """Get the track object given the id"""
    t = gordon.Track.query.filter(gordon.Track.id==t_id).limit(1)
    t = t.all()
    return t[0]


def get_analysis(track_id, analysis='librosa:low-level'):
    '''get the analysis data for a track by id'''
    
    track = get_track(track_id)
    
    analysis_file = track.annotation_dict[analysis]
    
    with open(analysis_file, 'r') as f:
        data = pickle.load(f)
        
    return data


