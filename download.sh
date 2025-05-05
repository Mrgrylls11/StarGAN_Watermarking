FILE=$1

if [ $FILE == "celeba" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/scl/fi/75do2onlortal0ltwzoco/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE
else
    echo "Available arguments are celeba, pretrained-celeba-128x128, pretrained-celeba-256x256."
    exit 1
fi
