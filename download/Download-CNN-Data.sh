#!/bin/bash

# List of Languages to download
LANGS=("azerbaijani" "turkish" "spanish")

# MMID CNN URL
URL_START="http://nlpgrid.seas.upenn.edu/MMID/"
URL_END=".tar.gz"

# MMID CNN File Name
FILE_EXTENSION=".tgz"

## Download Files
for lang in "${LANGS[@]}"; do
    if test -f "$lang$FILE_EXTENSION"; then
        echo "${lang^^} IS ALREADY DOWNLOADED"
        printf "\n"
    else
        echo "${lang^^} DOWNLOAD START"
        wget "$URL_START$lang$URL_END"
        echo "${lang^^} DOWNLOAD END"
        printf "\n"
    fi
done
