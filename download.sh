#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define dataset name and Dropbox URL
DATASET_NAME="celeba"
# --- IMPORTANT: Replace dl=0 with dl=1 for direct download ---
DROPBOX_URL="https://www.dropbox.com/scl/fi/75do2onlortal0ltwzoco/celeba.zip?rlkey=2z3i4pqbzzli1pkorm553cq9d&st=tgk1qaob&dl=1"
ZIP_FILE="${DATASET_NAME}.zip"
TARGET_DIR="data/${DATASET_NAME}" # Target directory structure (data/celeba/)
IMAGE_DIR="${TARGET_DIR}/images"
ATTR_FILE="${TARGET_DIR}/list_attr_celeba.txt"

# Check if the correct argument is provided
if [ "$1" != "${DATASET_NAME}" ]; then
  echo "Usage: bash $0 ${DATASET_NAME}"
  exit 1
fi

# Check if data already exists
if [ -d "${IMAGE_DIR}" ] && [ -f "${ATTR_FILE}" ]; then
  echo "Dataset '${DATASET_NAME}' already found in ${TARGET_DIR}. Skipping download."
  exit 0
fi

# Create the target directory if it doesn't exist
echo "Creating directory ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"

# Download the dataset
echo "Downloading ${DATASET_NAME} dataset from Dropbox..."
# Use wget with -O to specify the output file path and name
wget -O "${TARGET_DIR}/${ZIP_FILE}" "${DROPBOX_URL}" --no-check-certificate

# Check if download was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to download ${ZIP_FILE}. Check URL and network connection."
  # Clean up potentially incomplete zip file
  rm -f "${TARGET_DIR}/${ZIP_FILE}"
  exit 1
fi
echo "Download complete: ${TARGET_DIR}/${ZIP_FILE}"

# Unzip the dataset into the target directory
echo "Extracting ${ZIP_FILE}..."
unzip -q "${TARGET_DIR}/${ZIP_FILE}" -d "${TARGET_DIR}"
# -q for quiet mode

# Check if unzip was successful (basic check: image dir and attr file exist)
# This assumes the zip extracts 'images' folder and 'list_attr_celeba.txt' directly into TARGET_DIR
if [ ! -d "${IMAGE_DIR}" ] || [ ! -f "${ATTR_FILE}" ]; then
   echo "Error: Failed to extract dataset correctly."
   echo "Check the structure of ${ZIP_FILE}."
   echo "Expected structure: images/ folder and list_attr_celeba.txt inside the zip."
   # Optional: List contents if unzip failed to verify structure
   # echo "Listing contents of ${TARGET_DIR} after extraction attempt:"
   # ls -lR "${TARGET_DIR}"
   # Clean up
   # rm -f "${TARGET_DIR}/${ZIP_FILE}" # Keep zip for debugging if extraction fails?
   exit 1
fi
echo "Extraction complete."


# Clean up the downloaded zip file
echo "Cleaning up ${ZIP_FILE}..."
rm "${TARGET_DIR}/${ZIP_FILE}"

echo "CelebA dataset downloaded and extracted successfully to ${TARGET_DIR}."
exit 0
