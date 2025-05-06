if [ $FILE == "celeba" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/scl/fi/75do2onlortal0ltwzoco/celeba.zip?rlkey=2z3i4pqbzzli1pkorm553cq9d&st=r0lx0quu&dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/images
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/images
    rm $ZIP_FILE

else
    exit 1
fi
