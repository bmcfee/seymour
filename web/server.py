#!/usr/bin/env python

import argparse
import flask
import ConfigParser
import os
import sys
import ujson as json

import data_layer

DEBUG = True

# construct application object
app = flask.Flask(__name__)
app.config.from_object(__name__)

def load_config(server_ini):
    P       = ConfigParser.RawConfigParser()

    P.opionxform    = str
    P.read(server_ini)

    CFG = {}
    for section in P.sections():
        CFG[section] = dict(P.items(section))

    for (k, v) in CFG['server'].iteritems():
        app.config[k] = v
    return CFG

def run(**kwargs):
    app.run(**kwargs)

@app.route('/audio/<int:track_id>')
def get_track_audio(track_id):
    return flask.send_file(data_layer.get_track_audio(track_id), cache_timeout=0)

@app.route('/analysis/<int:track_id>')
def get_track_analysis(track_id):
    return json.encode(data_layer.get_track_analysis(track_id))

@app.route('/track/<int:track_id>')
def get_track(track_id):
    return flask.render_template('track.html', track_id=track_id)

@app.route('/collection/<int:collection_id>')
def get_collection_info(collection_id):
    return json.encode(data_layer.get_collection_info(collection_id));

@app.route('/collections')
def get_collections():
    return json.encode(data_layer.get_collections())

@app.route('/tracks/<int:collection_id>', defaults={'offset': 0, 'limit': 12})
@app.route('/tracks/<int:collection_id>/<int:offset>', defaults={'limit': 12})
@app.route('/tracks/<int:collection_id>/<int:offset>/<int:limit>')
def get_tracks(collection_id, offset, limit):
    limit = min(limit, 48)

    return json.encode(data_layer.get_tracks(collection_id, offset=offset, limit=limit))

@app.route('/', methods=['GET'], defaults={'collection_id': 1})
@app.route('/<int:collection_id>')
def index(collection_id):
    '''Top-level web page'''
    return flask.render_template('index.html', collection_id=collection_id)


# Main block
def process_arguments():

    parser = argparse.ArgumentParser(description='Yankomatic web server')

    parser.add_argument(    '-i',
                            '--ini',
                            dest    =   'ini',
                            required=   False,
                            type    =   str,
                            default =   'server.ini',
                            help    =   'Path to server.ini file')

    parser.add_argument(    '-p',
                            '--port',
                            dest    =   'port',
                            required=   False,
                            type    =   int,
                            default =   5000,
                            help    =   'Port')

    parser.add_argument(    '--host',
                            dest    =   'host',
                            required=   False,
                            type    =   str,
                            default =   '0.0.0.0',
                            help    =   'host')

    return vars(parser.parse_args(sys.argv[1:]))
                            
if __name__ == '__main__':
    parameters = process_arguments()

    CFG = load_config(parameters['ini'])

    port = parameters['port']

    if os.environ.get('ENV') == 'production':
        port = int(os.environ.get('PORT'))

    run(host=parameters['host'], port=port, debug=DEBUG)


