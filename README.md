seymour
=======

Integrating gordon with librosa analysis


To use this package, do the following :

1. install the gordon branch from the bmcfee fork: https://github.com/bmcfee/gordon
1. Edit the gordon.ini.default, and either copy it to /etc, or set up an environment variable:
    `export GORDON_INI=/path/to/your/gordon.ini`
1. Run the scripts in `code/` in alphabetical order:
    * `./S0_init_gordon.py`
    * `./S1_generate_index.py REGEXP tracks.json file1 [file2 ...]`
    * `./S2_intake_from_index.py "Collection name" tracks.json`
    * `./S3_analyze_librosa_lowlevel.py /path/to/low_level/analysis/output`
    * `./S3_title_cleanup.py`
    * `./S4_analyze_librosa_midlevel.py /path/to/mid_level/analysis/output`
    * To update analyses later, run `./S4_update_librosa_lowlevel.py /path/to/analysis/output feature1 [feature2 [ ... ] ]`
