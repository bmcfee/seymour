#!/usr/bin/env python

import gordon
import numpy as np
import cPickle as pickle

import whoosh
import whoosh.index
import whoosh.qparser
import whoosh.query

index = None

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

def open_index(index_path):

    global index
    index = whoosh.index.open_dir(index_path)
    pass

def whoosh_search(qstr, offset=0, limit=10):

    global index
    
    n = 0
    search_results = []

#     TODO:   2014-08-28 14:02:01 by Brian McFee <brm2132@columbia.edu>
# add collection name search 

    parser = whoosh.qparser.MultifieldParser(['title', 
                                              'artist', 
                                              'album',
                                              'collection'],
                                             index.schema,
                                             group=whoosh.qparser.OrGroup)
    parser.add_plugin(whoosh.qparser.EveryPlugin())

    query = parser.parse(qstr)
    
    if qstr:
        with index.searcher() as searcher:
            
            rs = searcher.search(query, limit=offset+limit)
            n = rs.estimated_length()
            for i, result in enumerate(rs):
                if i < offset:
                    continue
    
                if i >= limit+offset:
                    break
    
                sr = {'title': '', 'album': '', 'artist': '', 'track_id': ''}

                sr.update(dict(result))
                sr['collections'] = [sr['collection_id']]
                del sr['collection_id']
                del sr['collection']
                
                search_results.append(sr)

    return {'count': n, 
            'offset': offset,
            'limit': limit, 
            'results': search_results}


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
