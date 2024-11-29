# Check if GloVe embeddings exist, download if not
GLOVE_DIR="../glove"
GLOVE_FILE="glove.6B.50d.txt"

if [ ! -f "$GLOVE_DIR/$GLOVE_FILE" ]; then
    echo "GloVe embeddings not found. Downloading..."
    
    # Create the glove directory if it doesn't exist
    if [ ! -d "$GLOVE_DIR" ]; then
        mkdir "$GLOVE_DIR"
    fi

    # Download GloVe embeddings
    wget "http://nlp.stanford.edu/data/glove.6B.zip" -P "$GLOVE_DIR"
    
    # Unzip and keep only the 50d version
    unzip -q "$GLOVE_DIR/glove.6B.zip" -d "$GLOVE_DIR"
    rm "$GLOVE_DIR/glove.6B.zip"
    echo "Downloaded and extracted GloVe embeddings successfully."
else
    echo "GloVe embeddings already exist. Skipping download."
fi