#!/bin/bash

# $1 is the first argument of this method and is the superlevel directory that we call
# the script on.

echo $1

# Navigate to the directory.
cd $1
# Loop over the zipped folders in this directory.
for item in *; do
    if [ -n "$(file -b "$item" | grep -o 'Zip')" ]; then
	echo "Now unzipping and removing $item"
        unzip -q "$item" && rm "$item"
    fi
done
