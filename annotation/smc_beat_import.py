#!/usr/bin/env python
# CREATED:2014-01-10 11:15:33 by Brian McFee <brm2132@columbia.edu>
# Create an index file matching SMC songs to annotation files 

import os
import glob
import sys
import argparse

def parse_arguments():
    parser  =   argparse.ArgumentParser(description='Build an index of SMC song filenames to annotations')

    parser.add_argument('data_dir',
                        action  =   'store',
                        help    =   'Path to SMC data')

    parser.add_argument('output_file',
                        action  =   'store',
                        help    =   'Path to store the index file')

    return vars(parser.parse_args(sys.argv[1:]))

def build_index(data_dir=None, output_file=None):

    audio_files = sorted(glob.glob(os.path.join(data_dir, 'SMC_MIREX_Audio', '*.wav')))
    beat_files  = sorted(glob.glob(os.path.join(data_dir, 'SMC_MIREX_Annotations', '*.txt')))

    if len(audio_files) != len(beat_files):
        raise Exception('Audio and annotations do not have the same number of files')

    with open(output_file, 'w') as f:
        for aud_f, beat_f in zip(audio_files, beat_files):
            f.write('%s\t%s\n' % (aud_f, beat_f))

if __name__ == '__main__':
    args = parse_arguments()

    build_index(**args)
