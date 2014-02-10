#!/usr/bin/env python

import gordon
import cPickle as pickle
import numpy as np
import ujson as json

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
