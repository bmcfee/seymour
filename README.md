seymour
=======

Integrating gordon with librosa analysis


To use this package, do the following :

1. install the gordon branch from the bmcfee fork: https://github.com/bmcfee/gordon
1. Edit the gordon.ini.default, and either copy it to /etc, or set up an environment variable:
    `export GORDON_INI=/path/to/your/gordon.ini`
1. Run the scripts in `code/` in alphabetical order:
    * ./S0_init_gordon.py
    * ./S1_generate_tracklist.py REGEXP tracks.csv file1 [file2 ...]
    * ./S2_intake_from_tracklist.py "Collection name" tracks.csv
    * ./S3_analyze_librosa_lowlevel.py /path/to/analysis/output
