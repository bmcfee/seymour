#!/usr/bin/env python
# CREATED:2014-01-08 09:41:26 by Brian McFee <brm2132@columbia.edu>
#  normalize titles in gordon to remove underscores and use proper capitalization

import gordon

def normalize(s):

    # Replace underscores, if they exist
    if s.count('_') > 0:
        s = s.replace('_', ' ')

    # Capitalize, if it's all lower-case
    if s.islower():
        s = s.title()

    return s

def main():
    # First, do the artists
    print 'Correcting artists...'
    for artist in gordon.Artist.query.all():
        artist.name = normalize(artist.name)

    # Then albums
    print 'Correcting albums...'
    for album in gordon.Album.query.all():
        album.name = normalize(album.name)

    # Then tracks
    print 'Correcting tracks...'
    for track in gordon.Track.query.all():
        track.title = normalize(track.title)

    print 'Done.'
    gordon.commit()

if __name__ == '__main__':
    main()
