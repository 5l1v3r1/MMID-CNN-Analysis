#!/bin/bash

# Languages to Download
LANGS=("azerbaijani" "turkish" "spanish")


for lang in "${LANGS[@]}"; do
    if test -d "./CNN/$lang"; then
        echo "${lang^^} IS ALREADY UNZIPPED"
        printf "\n"
    else
        echo "${lang^^} UNZIP START"
        tar -zxf "$lang.tar.gz"
        echo "${lang^^} UNZIP END"
        printf "\n"
    fi
done
