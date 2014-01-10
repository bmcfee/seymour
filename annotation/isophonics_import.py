#!/usr/bin/env python
# CREATED:2014-01-10 11:15:33 by Brian McFee <brm2132@columbia.edu>
# Create an index file matching isophonics annotation files 

import argparse
import sys
import librosa

def parse_arguments():
    parser  =   argparse.ArgumentParser(description='Build an index of isophonics filenames to annotations')

    parser.add_argument('-e',
                        '--audio-extension',
                        dest    =   'audio_ext',
                        default =   'flac',
                        help    =   'File extension for audio data')

    parser.add_argument('-a',
                        '--annotation-extension',
                        dest    =   'ann_ext',
                        default =   'txt',
                        help    =   'File extension for annotations')

    parser.add_argument('audio_dir',
                        action  =   'store',
                        help    =   'Path to audio data')

    parser.add_argument('annotation_dir',
                        action  =   'store',
                        help    =   'Path to annotation data')

    parser.add_argument('output_file',
                        action  =   'store',
                        help    =   'Path to store the index file')

    return vars(parser.parse_args(sys.argv[1:]))

def build_index(audio_dir=None, annotation_dir=None, output_file=None, audio_ext=None, ann_ext=None):

    audio_files = librosa.util.get_audio_files(audio_dir, ext=audio_ext)
    ann_files  = librosa.util.get_audio_files(annotation_dir, ext=ann_ext)

    if len(audio_files) != len(ann_files):
        raise Exception('Audio and annotations do not have the same number of files')

    with open(output_file, 'w') as f:
        for aud_f, ann_f in zip(audio_files, ann_files):
            f.write('%s\t%s\n' % (aud_f, ann_f))

if __name__ == '__main__':
    args = parse_arguments()

    build_index(**args)
