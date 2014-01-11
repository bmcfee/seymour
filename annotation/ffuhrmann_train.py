#!/usr/bin/env python
# CREATED:2014-01-10 11:15:33 by Brian McFee <brm2132@columbia.edu>
# Create an index file matching ffuhrmann-train annotation files 

import argparse
import re
import os
import sys
import librosa
import ujson as json

def parse_arguments():
    parser  =   argparse.ArgumentParser(description='Build an index of ffuhrmann filenames to annotations')

    parser.add_argument('-e',
                        '--audio-extension',
                        dest    =   'audio_ext',
                        default =   'mp3',
                        help    =   'File extension for audio data')

    parser.add_argument('data_dir',
                        action  =   'store',
                        help    =   'Path to audio data')

    parser.add_argument('annotation_dir',
                        action  =   'store',
                        help    =   'Path to store annotation data')

    parser.add_argument('output_file',
                        action  =   'store',
                        help    =   'Path to store the index file')

    return vars(parser.parse_args(sys.argv[1:]))

def generate_annotations(audio_files):

    # mapping
    inst_map = {'cel': 'cello', 
                'cla': 'clarinet', 
                'flu': 'flute', 
                'gac': 'guitar-acoustic',
                'gel': 'guitar-electric',
                'org': 'organ',
                'pia': 'piano',
                'sax': 'saxophone',
                'tru': 'trumpet',
                'vio': 'violin',
                'voi': 'vocals'}

    genre_map = {'cla': 'classical',
                 'cou_fol': 'country-folk',
                 'jaz_blu': 'jazz-blues',
                 'lat_sou': 'latin-soul',
                 'pop_roc': 'pop-rock'}

    pattern = re.compile('.*\[(?P<instrument>.*)\]\[(?P<drums>.*)\]\[(?P<genre>.*)\].*')

    annotations = []
    for aud_f in audio_files:
        aud_f   = os.path.basename(aud_f)
        result  = pattern.match(aud_f)
        
        tags = {}
        tags[inst_map[result.group('instrument')]] = 1
        if result.group('drums') == 'dru':
            tags['drums'] = 1

        tags[genre_map[result.group('genre')]] = 1
        annotations.append(tags)

    return annotations

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

def build_index(data_dir=None, annotation_dir=None, output_file=None, audio_ext=None):

    annotation_dir = os.path.abspath(annotation_dir)

    audio_files = librosa.util.get_audio_files(data_dir , ext=audio_ext)
    annotations = generate_annotations(audio_files)

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
