#!/usr/bin/env python
# CREATED:2014-01-13 17:33:16 by Brian McFee <brm2132@columbia.edu>
# salami annotation import 


import argparse
import os
import sys
import librosa

def parse_arguments():
    parser  =   argparse.ArgumentParser(description='Build an index of salami filenames to annotations')

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

def load_annotations(data_dir, audio_files):
    
    annotations = []
    good_files  = []

    for aud_f in audio_files:
        id = os.path.splitext(os.path.basename(aud_f))[0]

        ann_f = os.path.join(data_dir, 'data', str(id), 'parsed', 'textfile1_functions.txt')
        if os.path.exists(ann_f):
            annotations.append(ann_f)
            good_files.append(aud_f)

    return good_files, annotations

def triplize(ann_orig, ann_new):

    boundaries = []
    labels = []

    with open(ann_orig, 'r') as f:
        for line in f:
            b, l = line.strip().split('\t', 2)
            boundaries.append(float(b))
            labels.append(l)

    segments = zip(boundaries[:-1], boundaries[1:])
    labels = labels[:-1]

    with open(ann_new, 'w') as f:
        for s, l in zip(segments, labels):
            f.write('%f\t%f\t%s\n' % (s[0], s[1], l))

def translate_annotations(annotation_dir, audio_files, ann_files_orig):

    ann_files = []

    try:
        os.makedirs(annotation_dir)
    except:
        pass

    for aud_f, afo in zip(audio_files, ann_files_orig):
        # load the file
        # convert to triples
        # save it to annotation_dir
        # add to ann_files
        afn = os.path.splitext(os.path.basename(aud_f))[0]
        afn = os.path.extsep.join([afn, 'lab'])
        afn = os.path.join(annotation_dir, afn)

        triplize(afo, afn)
        ann_files.append(afn)

    return ann_files

def build_index(data_dir=None, annotation_dir=None, output_file=None, audio_ext=None):

    audio_files = librosa.util.get_audio_files(os.path.join(data_dir, 'audio'), ext=audio_ext)
    audio_files, ann_files_orig   = load_annotations(data_dir, audio_files)

    if len(audio_files) != len(ann_files_orig):
        raise Exception('Audio and annotations do not have the same number of files')

    ann_files = translate_annotations(annotation_dir, audio_files, ann_files_orig)

    with open(output_file, 'w') as f:
        for aud_f, ann_f in zip(audio_files, ann_files):
            f.write('%s\t%s\n' % (aud_f, ann_f))

if __name__ == '__main__':
    args = parse_arguments()

    build_index(**args)
