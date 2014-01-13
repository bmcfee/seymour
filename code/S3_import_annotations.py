#!/usr/bin/env python
# CREATED:2014-01-10 11:11:11 by Brian McFee <brm2132@columbia.edu>
# Import annotation files

import argparse
import os
import sys
import shutil

import gordon

def load_annotation_index(annotation_index):
    
    # match the string processing to the intake function

    mapping = {}
    with open(annotation_index, 'r') as f:
        for line in f:
            aud_f, beat_f = line.strip().split('\t', 2)
            aud_f = os.path.basename(aud_f)

            # FIXME:  2014-01-13 15:14:54 by Brian McFee <brm2132@columbia.edu>
            #  actually, the intake script should change to match this, not the other
            #  way
            # latin1 re-encoding to match intake script
            try:
                aud_f = aud_f.decode('utf-8')
            except:
                try: aud_f.decode('latin1')
                except: aud_f = 'unknown'

            mapping[os.path.basename(aud_f)] = beat_f

    return mapping

def get_collection_tracks(collection):

    C = gordon.Collection.query.filter_by(name=collection).limit(1).all()

    if len(C) == 0:
        raise ValueError('Collection not found: %s' % collection)

    return C[0].tracks

def get_output_file(path, ext='txt'):

    drop_ext = os.path.splitext(path)[0]
    return os.path.extsep.join([drop_ext, ext])

def copy_annotation(input_file, output_file):

    try:
        os.makedirs(os.path.dirname(output_file))
    except:
        pass

    shutil.copyfile(input_file, output_file)


def import_annotations(collection=None, annotation_name=None, annotation_directory=None,
annotation_index=None, no_commit=False):

    # Start the session
    gordon.session.begin()

    # First, get the tracks for the collection
    tracks = get_collection_tracks(collection)

    # Then get the annotation mapping
    mapping = load_annotation_index(annotation_index)

    for t in tracks:
        if t.ofilename not in mapping:
            print 'Skipping [%s], not in annotation set' % t.ofilename
            continue

        annotation  = mapping[t.ofilename]
        output_file = os.path.join(annotation_directory, get_output_file(t.path))

        # Copy the annotation record
        if not no_commit:
            copy_annotation(annotation, output_file)

        if annotation_name in t.annotation_dict:
            print 'Updating [%s] annotation for %d' % (annotation_name, t.id)
            t.annotation_dict[annotation_name] = output_file
        else:
            print 'Adding [%s] annotation for %d' % (annotation_name, t.id)
            t.add_annotation(annotation_name, output_file)

    if no_commit:
        gordon.session.rollback()
    else:
        gordon.session.commit()

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Import annotations for a collection')

    parser.add_argument('-n',
                        '--no-commit',
                        dest    =   'no_commit',
                        action  =   'store_true',
                        default =   False,
                        help    =   'Do not commit changes to the database')

    parser.add_argument('collection',
                        action  =   'store',
                        help    =   'Which gordon collection to annotate')
    
    parser.add_argument('annotation_name',
                        action  =   'store',
                        help    =   'Name for the annotation (e.g., beats or segments)')

    parser.add_argument('annotation_directory',
                        action  =   'store',
                        help    =   'directory to store annotation output')

    parser.add_argument('annotation_index',
                        action  =   'store',
                        help    =   'file mapping the song filenames to annotation files')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    args = parse_arguments()

    import_annotations(**args)
