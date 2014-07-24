#!/usr/bin/env python
# CREATED:2014-07-24 13:14:22 by Brian McFee <brm2132@columbia.edu>
#  Build an instrument annotation dataset using remapped annotation data

import argparse
import sys

import seymour

import cPickle as pickle

def process_arguments(args):
    '''Argparser'''

    parser = argparse.ArgumentParser(description='Instrument dataset construction')

    parser.add_argument(    'data_file',
                            action  =   'store',
                            type    =   str,
                            help    =   'path to store the data file (.npy)')

    parser.add_argument(    'collection',
                            action  =   'store',
                            type    =   str,
                            help    =   'collection to operate on')

    parser.add_argument(    'tag_map',
                            action  =   'store',
                            type    =   str,
                            help    =   'pickle file containing the tag mapping')

    return vars(parser.parse_args(args))

def load_tag_mapping(tag_map_file):

    tag_map = {}
    with open(tag_map_file, 'r') as f:
        tag_map = pickle.load(f)

    return tag_map

def get_track_data(t_id, tag_map):

    vq = seymour.get_analysis(t_id, analysis='librosa:mid-level')['track_vq']

    raw_tags = seymour.get_tags(t_id, annotation='tags')

    # Apply the tag mapping
    inst_tags = {}

    for tag in raw_tags:
        if tag not in tag_map:
            continue
        
        new_tags, positive = tag_map[tag]
        
        for new_tag in new_tags:
            inst_tags[new_tag] = positive

    return vq, raw_tags, inst_tags

def save_data(data_file, ids, names, features, raw, insts):

    with open(data_file, 'w') as f:
        pickle.dump({'ids': ids,
                     'names': names,
                     'features': features,
                     'tag_raw': raw,
                     'tag_inst': insts}, f)

def build_instrument_data(data_file=None, collection=None, tag_map=None):

    inst_tags = load_tag_mapping(tag_map)

    track_ids = seymour.get_collection_tracks(collection, inst_tags)
    
    track_features  = []
    track_names     = []
    track_raw_tags  = []
    track_inst_tags = []

    for t_id in track_ids:
        track = seymour.get_track(t_id)
        t_f, t_raw, t_inst = get_track_data(t_id, inst_tags)
        
        track_names.append( (track.artist, track.title) )

        track_features.append(t_f)
        track_raw_tags.append(t_raw)
        track_inst_tags.append(t_inst)

    save_data(data_file, track_ids, track_names, track_features, track_raw_tags, track_inst_tags)
    pass

if __name__ == '__main__':
    parameters = process_arguments(sys.argv[1:])

    build_instrument_data(**parameters)
