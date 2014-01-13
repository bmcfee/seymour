#!/usr/bin/env python
# CREATED:2014-01-13 14:40:12 by Brian McFee <brm2132@columbia.edu>
# build a json-structured tag index for cal10k 

import argparse
import os
import sys
import ujson as json

def parse_arguments():
    parser  =   argparse.ArgumentParser(description='Build an index of cal10k filenames to annotations')

    parser.add_argument('data_dir',
                        action  =   'store',
                        help    =   'Path to cal10k')

    parser.add_argument('annotation_dir',
                        action  =   'store',
                        help    =   'Path to store annotation data')

    parser.add_argument('output_file',
                        action  =   'store',
                        help    =   'Path to store the index file')

    return vars(parser.parse_args(sys.argv[1:]))

def load_annotations(data_dir):
    # Import the song mapping
    song_index = {}
    with open(os.path.join(data_dir, 'songList.tab'), 'r') as f:
        for line in f:
            song_id, artist, title, filepath = line.strip().split('\t', 4)

            song_index[int(song_id)] = filepath

    # Import the annotations
    tag_mtx = {}
    with open(os.path.join(data_dir, 'PandoraTagSong.tab'), 'r') as f:
        for line in f:
            A = line.strip().split('\t')
            tag_name = A[0]
            for song_id in map(int, A[1::2]):
                if song_id not in tag_mtx:
                    tag_mtx[song_id] = {}
                tag_mtx[song_id][tag_name] = 1

    annotations = []
    songs       = []
    for song_id in song_index:
        songs.append(song_index[song_id])
        if song_id in tag_mtx:
            annotations.append(tag_mtx[song_id])
        else:
            annotations.append({})


    return songs, annotations

def save_annotations(annotation_dir, audio_files, annotations):

    try:
        os.makedirs(annotation_dir)
    except: 
        pass

    out_files = []
    for aud_f, ann in zip(audio_files, annotations):
        output_file = os.path.basename(aud_f)
        output_file = os.path.splitext(output_file)[0]
        output_file = os.path.extsep.join([output_file, 'json'])
        output_file = os.path.join(annotation_dir, output_file)

        with open(output_file, 'w') as f:
            json.dump(ann, f)
        out_files.append(output_file)

    return out_files

def build_index(data_dir=None, annotation_dir=None, output_file=None):

    annotation_dir = os.path.abspath(annotation_dir)

    audio_files, annotations = load_annotations(data_dir)

    # Store the annotations as json files
    ann_files   = save_annotations(annotation_dir, audio_files, annotations)

    if len(audio_files) != len(ann_files):
        raise Exception('Audio and annotations do not have the same number of files')

    with open(output_file, 'w') as f:
        for aud_f, ann_f in zip(audio_files, ann_files):
            f.write('%s\t%s\n' % (aud_f, ann_f))

if __name__ == '__main__':
    args = parse_arguments()

    build_index(**args)
