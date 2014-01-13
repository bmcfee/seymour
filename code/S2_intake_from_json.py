#!/usr/bin/env python

# Copyright (C) 2010 Douglas Eck
#
# This file is part of Gordon.
#
# Gordon is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gordon is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Gordon.  If not, see <http://www.gnu.org/licenses/>.

'''
Functions for importing music to Gordon database

script usage:
python audio_intake_from_tracklist.py source csv [doit]
<source> is a name for the collection
<csv> is the collection physical location to browse
<doit> is an optional parameter; if False, no actual import takes
place, only verbose would-dos
'''

import os, collections, datetime, logging, stat, sys

import argparse

import ujson as json

from gordon.io import AudioFile
from gordon.db.model import add, commit, Album, Artist, Track, Collection, Annotation
from gordon.db.config import DEF_GORDON_DIR
from gordon.db.gordon_db import get_tidfilename, make_subdirs_and_copy, is_binary
from gordon.io.mp3_eyeD3 import isValidMP3, getAllTags


log = logging.getLogger('gordon.audio_intake_from_json')

def track_exists(fn_recoded, bytes):
    '''Check to see if a track already exists, using the filename and size'''

    T = Track.query.filter_by(ofilename=fn_recoded, bytes=bytes)
    return T.count() > 0

def add_track(trackpath, source=str(datetime.date.today()),
              gordonDir=DEF_GORDON_DIR, tag_dict=None, artist=None,
              album=None, fast_import=False, import_md=False):
    """Add track with given filename <trackpath> to database
    
         @param source: audio files data source (string)
         @param gordonDir: main Gordon directory
         @param tag_dict: dictionary of key,val tag pairs - See add_album(...).
         @param artist: The artist for this track. An instance of Artist. None if not present
         @param album: The album for this track. An instance of Album. None if not present
         @param fast_import: If true, do not calculate strip_zero length. Defaults to False
         @param import_md: use True to try to extract all metadata tags embedded in the auudio-file. Defaults to False
    """
    (path, filename) = os.path.split(trackpath) 
    (fname, ext) = os.path.splitext(filename)

    if tag_dict is None:
        tag_dict = {}

    log.debug('Adding file "%s" of "%s" album by %s', filename, album, artist)
    
    # validations
    if 'album' not in tag_dict:
        #todo: currently cannot add singleton files. Need an album which is defined in tag_dict
        log.error('Cannot add "%s" because it is not part of an album',
                  filename)
        return -1 # didn't add
    if not os.path.isfile(trackpath):
        log.info('Skipping %s because it is not a file', filename)
        return -1 # not a file

# FIXME:  2014-01-10 21:42:48 by Brian McFee <brm2132@columbia.edu>
#  gordon's AudioFile code doesn't handle unicode filenames correctly
#     try:
#         AudioFile(trackpath).read(tlen_sec=0.01)
#     except:
#         log.error('Skipping "%s" because it is not a valid audio file', filename)
#         return -1 # not an audio file

    # required data
    bytes = os.stat(trackpath)[stat.ST_SIZE]

    # FIXME:  2014-01-13 15:14:39 by Brian McFee <brm2132@columbia.edu>
    #  gaaaaah fn_recoded = 'unknown'?!  this is very wrong.
    # reencode name to latin1 !!!
    try:
        fn_recoded = filename.decode('utf-8')
    except:
        try: fn_recoded = filename.decode('latin1')
        except: fn_recoded = 'unknown'


    # First look for dupes
    if track_exists(fn_recoded, bytes):
        log.debug('Skipping "%s" because it has already been indexed', filename)
        return -1 # Already exists

    # prepare data
    
    if tag_dict[u'compilation'] not in [True, False, 'True', 'False'] :
        tag_dict[u'compilation'] = False

    track = Track(title = tag_dict[u'title'],
                  artist = tag_dict[u'artist'],
                  album = tag_dict[u'album'],
                  tracknum = tag_dict[u'tracknum'],
                  compilation = tag_dict[u'compilation'],
                  otitle = tag_dict[u'title'],
                  oartist = tag_dict[u'artist'],
                  oalbum = tag_dict[u'album'],
                  otracknum = tag_dict[u'tracknum'],
                  ofilename = fn_recoded,
                  source = unicode(source),
                  bytes = bytes)
    
    # add data
    add(track) # needed to get a track id
    commit() #to get our track id we need to write this record
    log.debug('Wrote track record %s to database', track.id)

    if fast_import :
        track.secs = -1
        track.zsecs = -1
    else :
        a = AudioFile(trackpath)
        [track.secs, track.zsecs] = a.get_secs_zsecs()
        
    track.path = u'%s' % get_tidfilename(track.id, ext[1:])

    # links track to artist & album in DB
    if artist:
        log.debug('Linking %s to artist %s', track, artist)
        track.artist = artist.name
        track.artists.append(artist)
    if album:
        log.debug('Linking %s to album %s', track, album)
        track.album = album.name
        track.albums.append(album)

    log.debug('Wrote album and artist additions to track into database')

    # copy the file to the Gordon audio/feature data directory
    tgt = os.path.join(gordonDir, 'audio', 'main', track.path)
    make_subdirs_and_copy(trackpath, tgt)
    log.debug('Copied "%s" to %s', trackpath, tgt)
    
    # add annotations
    
    del(tag_dict[u'title'])
    del(tag_dict[u'artist'])
    del(tag_dict[u'album'])
    del(tag_dict[u'tracknum'])
    del(tag_dict[u'compilation'])
    for tagkey, tagval in tag_dict.iteritems(): # create remaining annotations
        track.annotations.append(Annotation(type='text', name=tagkey, value=tagval))
    
    if import_md:
        #check if file is mp3. if so:
        if isValidMP3(trackpath):
            #extract all ID3 tags, store each tag value as an annotation type id3.[tagname]
            for tag in getAllTags(trackpath):
                track.annotations.append(Annotation(type='id3', name=tag[0], value=tag[1]))
        
        #todo: work with more metadata formats (use tagpy?)
    
    # Link the track to the collection object
    track.collections.append(get_or_create_collection(source))
    commit() # store the annotations
    log.debug('Added "%s"', trackpath)

def _read_json(cwd, csv=None):
    '''Reads a csv file containing track metadata and annotations (v 2.0)
    
    # may use py comments in metadata .csv file. Must include a header:
    filename, title, artist, album, tracknum, compilation, [optional1], [optional2]...
    and then corresponding values per line (see example metadata.csv file in project root)
    
    @return: a 2D dict in the form dict[<file-name>][<tag>] or False if an error occurs
    
    @param cwd: complete csv file-path (if no <csv> sent) or path to directory to work in  
    @param csv: csv file-name (in <cwd> dir). Defaults to None
    
    @todo: csv values (after header) may include "\embedded" to try getting it from the audio file.
    Currently only ID3 tags names understood by gordon.io.mp3_eyeD3.id3v2_getval_sub are usable in this manner.
    @todo: include other metadata formats (use tagpy?)'''

    # open csv file
    if csv is None:
        filename = cwd
    else:
        filename = os.path.join(cwd, csv)

    try:
        data = json.load(open(filename, 'r'))
    except IOError:
        log.error("  Couldn't open '%s'", csv)
        raise
    
    tags = dict()
    for line_num, line in enumerate(data): 
        # save title, artist, album, tracknum, compilation in tags[<file-name>]
        filepath = line.pop('filepath')

        # this deletes previous lines if filepath is repeated ...
        tags[filepath] = {'tracknum': 0, 'compilation': False}

        tags[filepath].update(line)

        for k, value in tags[filepath].iteritems():
            if not (isinstance(value, str) or isinstance(value, unicode)):
                continue

            if os.path.isfile(value):
                if not is_binary(value):
                    try:
                        txt = open(value)
                        tags[filepath][k] = unicode(txt.read())
                        txt.close()
                    except:
                        log.error('Error opening %s file %s at line %d', k, value, line_num)
                        tags[filepath][k] = unicode(value)
                else:
                    try:
                        tags[filepath][k] = u'%s' % value
                    except UnicodeDecodeError:
                        tags[filepath][k] = value.decode('utf-8')

    return tags

def add_album(album_name, tags_dicts, source=str(datetime.date.today()),
              gordonDir=DEF_GORDON_DIR, prompt_aname=False, import_md=False, fast_import=False):
    """Add an album from a list of metadata in <tags_dicts> v "1.0 CSV"
    """
    log.debug('Adding album "%s"', album_name)
    
    # create set of artists from tag_dicts
    
    artists = set()
    for track in tags_dicts.itervalues():
        artists.add(track['artist'])
    
    if len(artists) == 0:
        log.debug('Nothing to add')
        return # no songs
    else:
        log.debug('Found %d artists in directory: %s', len(artists), artists)
    
    #add album to Album table
    log.debug('Album has %d tracks', len(tags_dicts))
    albumrec = Album(name = album_name, trackcount = len(tags_dicts))

    #if we have an *exact* string match we will use the existing artist
    artist_dict = dict()
    for artist in artists:
        match = Artist.query.filter_by(name=artist)
        if match.count() == 1 :
            log.debug('Matched %s to %s in database', artist, match[0])
            artist_dict[artist] = match[0]
            #todo: (eckdoug) what happens if match.count()>1? This means we have multiple artists in db with same 
            # name. Do we try harder to resolve which one? Or just add a new one.  I added a new one (existing code)
            # but it seems risky.. like we will generate lots of new artists. 
            # Anyway, we resolve this in the musicbrainz resolver....
        else :
            # add our Artist to artist table
            newartist = Artist(name = artist)
            artist_dict[artist] = newartist

        #add artist to album (album_artist table)
        albumrec.artists.append(artist_dict[artist])

    # Commit these changes in order to have access to this album
    # record when adding tracks.
    commit()

    # Now add our tracks to album.
    for filename in sorted(tags_dicts.keys()):
        add_track(filename, source=source, gordonDir=gordonDir, tag_dict=tags_dicts[filename],
                  artist=artist_dict[tags_dicts[filename][u'artist']], album=albumrec,
                  fast_import=fast_import, import_md=import_md)

    #now update our track counts
    for aname, artist in artist_dict.iteritems() :
        artist.update_trackcount()
        log.debug('Updated trackcount for artist %s', artist)
    albumrec.update_trackcount()
    log.debug('Updated trackcount for album %s', albumrec)
    commit()

def get_or_create_collection(source):

    match = Collection.query.filter_by(name = unicode(source))

    if match.count() == 1:
        log.debug(' Matched source %s in database', match[0])
        return match[0]
    else:
        return Collection(name=unicode(source))

def add_collection_from_json_file(jsonfile, source=str(datetime.date.today()),
                                 prompt_incompletes=False, doit=False,
                                 gordonDir=DEF_GORDON_DIR, fast_import=False,
                                 import_md=False):
    """Adds tracks from a CSV (file) list of file-paths.
     
    Only imports if all songs actually have same album name. 
    With flag prompt_incompletes will prompt for incomplete albums as well

    Use doit=True to actually commit the addition of songs
    """
    metadata = _read_json(jsonfile)

    # Turn metadata into a list of albums:
    albums = collections.defaultdict(dict)
    for filename, x in metadata.iteritems():
        albums[x['album']][filename] = x

    ntracks = 1
    for albumname in sorted(albums):
        tracks = albums[albumname]
        # tracks is a 2D dict[<file-name>][<tag>] for a single album.
        if not doit:
            print 'Would import album "%s"' % albumname
            for track in sorted(tracks):
                print '  Would import file %d: "%s"' % (ntracks, track)
                for key, value in tracks[track].iteritems():
                    strvalue = '%s' % value
                    if '\n' in strvalue:
                        strvalue = '%s ...' % strvalue.split('\n')[0]
                    print '    %s: %s' % (key, strvalue)
                ntracks += 1
        else:
            add_album(  albumname, 
                        tracks, 
                        source=source, 
                        gordonDir=gordonDir, 
                        prompt_aname=prompt_incompletes, 
                        import_md=import_md,
                        fast_import=fast_import)
    
    print 'Finished'

def process_arguments():
    parser = argparse.ArgumentParser(description='Gordon audio intake from track list')

    parser.add_argument('source',
                        action  =   'store',
                        help    =   'name for the collection')

    parser.add_argument('jsonfile',
                        action  =   'store',
                        help    =   'path to the track list CSV file')

    parser.add_argument('-f', 
                        '--fast',
                        action  =   'store_true',
                        dest    =   'fast',
                        default =   False,
                        help    =   'imports without calculating zero-stripped track times')

    parser.add_argument('--no-prompt',
                        action  =   'store_false',
                        dest    =   'prompt_incompletes',
                        default =   True,
                        help    =   'Do not prompt for incomplete albums. See log for what we skipped.')

    parser.add_argument('-t', 
                        '--test',
                        action  =   'store_false',
                        dest    =   'doit',
                        default =   True,
                        help    =   'test the intake without modifying the database')
    parser.add_argument('-m',
                        '--metadata',
                        action  =   'store_true',
                        dest    =   'import_md',
                        default =   False,
                        help    =   'import all metadata flags from the audio file')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':

    args = process_arguments()

    log.info('Importing audio from tracklist %s (source=%s)', args['jsonfile'], args['source'])
    add_collection_from_json_file(  args['jsonfile'], 
                                    source=args['source'],
                                    prompt_incompletes=args['prompt_incompletes'],
                                    doit=args['doit'], 
                                    fast_import=args['fast'],
                                    import_md=args['import_md'])
    
