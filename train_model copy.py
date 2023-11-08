from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense
from keras.models import save_model
from keras.callbacks import ModelCheckpoint  # Import ModelCheckpoint



def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = Sequential()

    # Embedding layer
    model.add(Embedding(input_dim=word_types, output_dim=32, input_length=6))
    model.add(Flatten())  # Flatten the output of the embedding layer

    # Hidden layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))

    # Output layer
    model.add(Dense(outputs, activation='softmax'))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                  loss='categorical_crossentropy')

    return model


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <input_data> <target_data> <output_model>")
        sys.exit(1)

    input_file = sys.argv[1]
    target_file = sys.argv[2]
    output_model_file = sys.argv[3]

    # Load training data matrices
    inputs = np.load(input_file)
    outputs = np.load(target_file)

    # Create a feature extractor
    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'
    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)

    # Build the model
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))

    # Define a checkpoint to save the best model during training
    checkpoint = ModelCheckpoint(output_model_file, monitor='val_loss', save_best_only=True)

    # Train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100, validation_split=0.1, callbacks=[checkpoint])

    # Save the trained model
    save_model(model, output_model_file)