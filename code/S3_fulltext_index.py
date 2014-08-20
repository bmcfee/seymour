#!/usr/bin/env python
# CREATED:2014-08-20 12:01:45 by Brian McFee <brm2132@columbia.edu>
#  construct a full-text index of a gordon database (metadata only)
#  index entries contain:
#       track id
#       title
#       artist
#       album
#       collection_name
#       collection_id

import argparse
import logging
import os
import sys

import whoosh
import whoosh.analysis
import whoosh.index
import whoosh.fields

from whoosh.support.charset import accent_map

import gordon

log = logging.getLogger('gordon.fulltext_index')

def create_index_writer(index_path):
    '''Create a new whoosh index in the given directory path.
    
    Input: directory in which to create the index
    
    Output: `whoosh.index` writer object
    '''
    

    if not os.path.exists(index_path):
        os.mkdir(index_path)

    analyzer = (whoosh.analysis.StemmingAnalyzer() | 
                whoosh.analysis.CharsetFilter(accent_map))

    schema = whoosh.fields.Schema(track_id=whoosh.fields.STORED,
                                  title=whoosh.fields.TEXT(stored=True, analyzer=analyzer),
                                  artist=whoosh.fields.TEXT(stored=True, analyzer=analyzer),
                                  album=whoosh.fields.TEXT(stored=True, analyzer=analyzer),
                                  collection=whoosh.fields.KEYWORD(stored=True),
                                  collection_id=whoosh.fields.NUMERIC(stored=True))

    index = whoosh.index.create_in(index_path, schema)

    return index.writer()


def build_index(output_dir=None):
    
    # 1. Build the parser
    # 2. Iterate over tracks
    # 3. Done

    writer = create_index_writer(output_dir)

    for track in gordon.Track.query.all():
        datum = {'track_id': track.id,
                 'title': track.title,
                 'artist': track.artist,
                 'album': track.album,
                 'collection_id': track.collections[0].id,
                 'collection': track.collections[0].name}
        
        for k, v in datum.iteritems():
            if not v:
                datum[k] = None

        writer.add_document(**datum)

    writer.commit()

    pass

def process_arguments(args):

    parser = argparse.ArgumentParser(description='Construct a Gordon fulltext index')

    parser.add_argument('output_dir',
                        action  = 'store',
                        help    = 'path to store the index structure')

    return vars(parser.parse_args(args))

if __name__ == '__main__':

    args = process_arguments(sys.argv[1:])

    log.info('Building fulltext index in %s', args['output_dir'])

    build_index(**args)
