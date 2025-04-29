#!/bin/bash

REPO_URL="git@github.com:SarathViswa09/aodt.git"
CLONE_DIR="/tmp/repo_clone"

# set path to adot_sim
TARGET_DIR="$HOME/aodt_1.2.0/backend/aodt_sim/"

echo "Cloning the repository from $REPO_URL..."
git clone "$REPO_URL" "$CLONE_DIR"

if [ $? -ne 0 ]; then
    echo "Failed to clone the repository."
    exit 1
fi

echo "Overwriting files in $TARGET_DIR with files from the repository..."
cp -rf "$CLONE_DIR/"* "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "Files successfully overwritten."
else
    echo "Failed to overwrite the files."
    exit 1
fi

rm -rf "$CLONE_DIR"
echo "Temporary repository cloned in $CLONE_DIR has been removed."
