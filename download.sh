#!/bin/bash

FILE=$1

if [ "$FILE" == "celeba" ]; then
    echo "🔽 Downloading CelebA dataset from Dropbox..."

    # Direct Dropbox download link
    URL="https://www.dropbox.com/scl/fi/75do2onlortal0ltwzoco/celeba.zip?rlkey=2z3i4pqbzzli1pkorm553cq9d&st=r0lx0quu&dl=1"
    ZIP_FILE=./data/celeba.zip

    # Create target folders
    mkdir -p ./data/images

    # Download ZIP file
    wget -N "$URL" -O "$ZIP_FILE"

    # Unzip to data/images
    echo "📦 Unzipping CelebA ZIP..."
    unzip -q "$ZIP_FILE" -d ./data/images

    # Remove ZIP
    rm "$ZIP_FILE"

    echo "✅ Done. Data is in ./data/images"
else
    echo "❌ Invalid argument. Usage: bash download.sh celeba"
    exit 1
fi
