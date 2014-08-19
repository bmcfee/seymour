#!/usr/bin/env python
# CHANGED:2014-01-10 22:36:21 by Brian McFee <brm2132@columbia.edu>
#  adapted from the CSV intake script by Ron Weiss

import argparse
import re
import sys
import ujson as json

def extract_metadata_from_filename(filename, pattern):
    tagdict = dict(title=None, artist=None, album=None, tracknum=-1,
                   compilation=False)
    
    m = re.match(unicode(pattern,  'utf-8', errors='ignore'), 
                 unicode(filename, 'utf-8', errors='ignore'))
    if m:
        for key, val in m.groupdict().iteritems():
            tagdict[key] = val

    return tagdict

def main(pattern=None, output_filename=None, filenames=None):
    keys = ('title', 'artist', 'album', 'tracknum', 'compilation')

    data = []
    for filename in filenames:
        print >> sys.stderr, 'Processing %s' % filename

        tagdict = extract_metadata_from_filename(filename, pattern)
        record = {'filepath': filename}

        for k in keys:
            record[k] = tagdict[k]

        data.append(record)

    with open(output_filename, 'w') as f:
        json.dump(data, f)
    
def process_arguments():

    usage = """Extracts metadata from each of the given filenames using the specified
        regexp pattern to generate a tracklist.  The pattern must name
        matching groups with the names "artist", "album", "title", "tracknum",
        or "compilation" to fill in the corresponding track metadata fields.
        
        For example, the pattern:
        '.*/(?P<artist>.*)-(?P<album>.*)-(?P<tracknum>[0-9]*)-(?P<title>.*)\.'
        will match filenames of the form:
        /path/to/artist name-album name-track number-title.mp3
        
        Note that this script *does not* support gordon annotations.  You must
        add them to the generated tracklist manually using e.g. sed.
        
        Example Usage:
        
        %(prog)s \\
            '.*/audio/(?P<artist>The Beatles)/.*_-_(?P<album>.*)/(?P<tracknum>[0-9]*)_-_(?P<title>.*)\.' \\
            tracklist.json \\
            ~/data/beatles/audio/The\ Beatles/*/*wav
        
        creates tracklist.json
        """

    parser = argparse.ArgumentParser(description='Generate a gordon intake file', 
                                     usage=usage)
    
    parser.add_argument('pattern',
                        action  =   'store',
                        help    =   'Regular expression to match')

    parser.add_argument('output_filename',
                        action  =   'store',
                        help    =   'Path to store the intake file (json)')

    parser.add_argument('filenames',
                        action  =   'store',
                        nargs   =   '+',
                        help    =   'One or more files to intake')
    
    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == "__main__":
    main(**process_arguments())
