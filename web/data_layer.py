#!/usr/bin/env python

import gordon
import numpy as np
import cPickle as pickle

#-- collection functions
def get_collections():
    return [{'collection_id': c.id, 
             'name': c.name,
             'track_count': c.trackcount} for c in gordon.Collection.query.all()]

def get_collection_info(collection_id):
    c = gordon.Collection.query.get(collection_id)
    return {'name': c.name, 'track_count': c.trackcount}

def get_tracks(collection_id, offset=0, limit=10):

    # Maybe easier with just collection objects
    C = gordon.Collection.query.get(collection_id)
    
    query = gordon.Track.query.\
                        filter(gordon.Track.collections.contains(C)).\
                        offset(offset).\
                        limit(limit)
    
    return [{'track_id':    t.id, 
             'title':       t.title, 
             'artist':      t.artist,
             'album':       t.album} for t in query.all()]

def search_tracks(collection_id=None, rawstr=None, artist=None, title=None, album=None, 
                  offset=0, limit=10):

    # Initialize the query
    query = gordon.Track.query

    # Do we have a collection?
    if collection_id is not None:
        C = gordon.Collection.query.get(collection_id)

        query = query.filter(gordon.Track.collections.contains(C))

    if rawstr is None:
        if artist is not None:
            query = query.filter(gordon.Track.artist.like(u'%%%s%%' % artist))
        if title is not None:
            query = query.filter(gordon.Track.title.like(u'%%%s%%' % title))
        if album is not None:
            query = query.filter(gordon.Track.album.like(u'%%%s%%' % title))
    else:
        query = query.filter(gordon.Track.artist.like(u'%%%s%%' % rawstr) | \
                             gordon.Track.title.like(u'%%%s%%' % rawstr) | 
                             gordon.Track.album.like(u'%%%s%%' % rawstr))

    query = query.offset(offset).limit(limit)

    return [{'track_id':    t.id, 
             'title':       t.title, 
             'artist':      t.artist,
             'album':       t.album} for t in query.all()]

#-- track functions
def get_track_audio(track_id):
    
    track = gordon.Track.query.get(track_id)

    return track.fn_audio


def __get_track_lowlevel(track):

    if 'librosa:low-level' not in track.annotation_dict:
        return {}

    with open(track.annotation_dict['librosa:low-level'], 'r') as f:
        analysis = pickle.load(f)

    data = {}

    data['signal']      = analysis['signal'].tolist()
    data['tempo']       = analysis['tempo']
    data['tuning']      = analysis['tuning']
    data['duration']    = analysis['duration']

    return data

def __get_track_midlevel(track):
    
    if 'librosa:mid-level' not in track.annotation_dict:
        return {}

    with open(track.annotation_dict['librosa:mid-level'], 'r') as f:
        analysis = pickle.load(f)

    data = {}
    data['beats']       = analysis['beat_times'].tolist()
    data['links']       = analysis['mfcc_neighbors_beat'].tolist()
    data['segment_tree']= [z.tolist() for z in analysis['segment_beat_tree']]
    data['segments']    = analysis['segment_beat_tree'][analysis['segments_best']].tolist()
    data['cqt']         = (analysis['beat_sync_cqt'] ** (1./3)).T.tolist()
    data['chroma']      = np.roll(analysis['beat_sync_chroma'], -3, axis=0).T.tolist()

    return data

def get_track_analysis(track_id):

    track = gordon.Track.query.get(track_id)

    analysis = { 'track_id':    track_id,
                 'title':       track.title,
                 'artist':      track.artist,
                 'album':       track.album}

    analysis.update(__get_track_lowlevel(track))
    analysis.update(__get_track_midlevel(track))

    return analysis
