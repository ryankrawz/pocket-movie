import os

import numpy as np
import tensorflow as tf


tf.compat.v1.enable_eager_execution()


BATCH_SIZE = 64
BUFFER_SIZE = 10000
CHARS_TO_GENERATE = 100
CHECKPOINT_DIR = './training_data/training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, 'cpt_{epoch}')
EMBEDDING_DIMENSIONS = 256
EPOCHS = 5
INPUT_LENGTH = 100
PATH_TO_TRAINING_SCRIPT = 'training_data/raw_text_scripts/all_scripts_continuous.txt'
RNN_UNITS = 1024
S_1, S_2 = 342, 400
TEMPERATURE = 1.0


def build_model(vocab_size, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIMENSIONS,
            batch_input_shape=[batch_size, None]
        ),
        tf.keras.layers.GRU(
            RNN_UNITS,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dense(vocab_size)
    ])


def get_script_text():
    return open(PATH_TO_TRAINING_SCRIPT, 'rb').read().decode(encoding='utf-8')


def initialize_model():
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not last_checkpoint:
        raise Exception('Checkpoint for restoring model does not exist in \'{}\''.format(
            CHECKPOINT_DIR
        ))
    vocab = sorted(set(get_script_text()))
    model = build_model(len(vocab), 1)
    model.load_weights(last_checkpoint).expect_partial()
    model.build(tf.TensorShape([1, None]))
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def map_chars(vocab):
    return {u: i for i, u in enumerate(vocab)}, np.array(vocab)


def split_input_target(sequence):
    return sequence[:-1], sequence[1:]


def train_rnn():
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if last_checkpoint:
        raise Exception('Training checkpoints exist. Clear \'{}\' if you wish to retrain'.format(
            CHECKPOINT_DIR
        ))
    # Read in continuous text file of all scripts
    training_text = get_script_text()
    print('Length of text: {}'.format(len(training_text)))
    # Determine language vocabulary
    vocab = sorted(set(training_text))
    print('Total unique characters: {}'.format(len(vocab)))
    # Convert text to vector on integer representations of each character
    char_to_index, index_to_char = map_chars(vocab)
    vectorized_text = np.array([char_to_index[c] for c in training_text])
    print('{} ----> {}'.format(repr(training_text[S_1:S_2]), vectorized_text[S_1:S_2]))
    # Create training inputs and targets
    char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_text)
    sequences = char_dataset.batch(INPUT_LENGTH + 1, drop_remainder=True)
    # Split training sequences into input and target text
    dataset = sequences.map(split_input_target)
    # Shuffle training data
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # Construct model
    model = build_model(len(vocab), BATCH_SIZE)
    # Configure training procedure
    model.compile(optimizer='adam', loss=loss)
    # Ensure checkpoints are stored
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PREFIX,
        save_weights_only=True
    )
    # Execute training
    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


def generate_text(start_string):
    model = initialize_model()
    vocab = sorted(set(get_script_text()))
    char_to_index, index_to_char = map_chars(vocab)
    input_vector = [char_to_index[c] for c in start_string]
    input_vector = tf.expand_dims(input_vector, 0)
    payload = ''
    model.reset_states()
    for _ in range(CHARS_TO_GENERATE):
        predictions = model(input_vector)
        predictions = tf.squeeze(predictions, 0) / TEMPERATURE
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_vector = tf.expand_dims([predicted_id], 0)
        payload += index_to_char[predicted_id]
    return payload
