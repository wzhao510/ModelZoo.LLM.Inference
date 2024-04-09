#!/bin/bash

declare -a processed_args

# convert arguments
for arg in "$@"; do
    if [[ -e "$arg" ]]; then  
        # convert to real path 
        processed_args+=("$(realpath "$arg")")
    else
        processed_args+=("$arg")
    fi
done

# call start.py
python ./Code/Start.py "${processed_args[@]}"



