#!/usr/bin/env python
# CREATED:2014-02-11 12:07:54 by Brian McFee <brm2132@columbia.edu>
#  Gordon-backed implementation of OLDA learning

import argparse
import sys
import gordon
import cPickle as pickle
from joblib import Parallel, delayed
import mir_eval

import OLDA
import seymour
import seymour_analyzers.librosa_midlevel as midlevel

def process_arguments(args):
    '''Argparser'''

    parser = argparse.ArgumentParser(description='Segmentation learning for seymour')

    parser.add_argument(    'model_file',
                            action  =   'store',
                            type    =   str,
                            help    =   'path to store the learned model file')

    parser.add_argument(    '-j',
                            '--num-jobs',
                            dest    =   'num_jobs',
                            action  =   'store',
                            type    =   int,
                            default =   4,
                            help    =   'number of threads')

    parser.add_argument(    'collections',
                            action  =   'store',
                            type    =   str,
                            nargs   =   '+',
                            help    =   'one or more collections to train from')

    return vars(parser.parse_args(args))



if __name__ == '__main__':
    parameters = process_arguments(sys.argv[1:])

