import os

import numpy as np
import tensorflow as tf


tf.compat.v1.enable_eager_execution()


class SentenceGenerationRNN:
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
    TEMPERATURE = 1.0

    def __init__(self):
        # Spin up neural language generation model
        self.model = self.__initialize_model()
        # Read in continuous text file of all scripts
        self.training_text = self.__get_script_text()
        # Determine language vocabulary
        self.vocab = sorted(set(self.training_text))
        # Map characters to integers in order to generate vectors
        self.char_to_index, self.index_to_char = self.__map_chars(self.vocab)

    def __build_model(self, vocab_size, batch_size):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size,
                self.EMBEDDING_DIMENSIONS,
                batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.GRU(
                self.RNN_UNITS,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'
            ),
            tf.keras.layers.Dense(vocab_size)
        ])

    def __get_script_text(self):
        return open(self.PATH_TO_TRAINING_SCRIPT, 'rb').read().decode(encoding='utf-8')

    def __initialize_model(self):
        last_checkpoint = tf.train.latest_checkpoint(self.CHECKPOINT_DIR)
        if not last_checkpoint:
            return None
        vocab = sorted(set(self.__get_script_text()))
        model = self.__build_model(len(vocab), 1)
        model.load_weights(last_checkpoint).expect_partial()
        model.build(tf.TensorShape([1, None]))
        return model

    @staticmethod
    def __loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    @staticmethod
    def __map_chars(vocab):
        return {u: i for i, u in enumerate(vocab)}, np.array(vocab)

    @staticmethod
    def __split_input_target(sequence):
        return sequence[:-1], sequence[1:]

    def train_rnn(self):
        last_checkpoint = tf.train.latest_checkpoint(self.CHECKPOINT_DIR)
        if last_checkpoint:
            raise Exception('Training checkpoints exist. Clear \'{}\' if you wish to retrain'.format(
                self.CHECKPOINT_DIR
            ))
        # Convert text to vector on integer representations of each character
        vectorized_text = np.array([self.char_to_index[c] for c in self.training_text])
        # Create training inputs and targets
        char_dataset = tf.data.Dataset.from_tensor_slices(vectorized_text)
        sequences = char_dataset.batch(self.INPUT_LENGTH + 1, drop_remainder=True)
        # Split training sequences into input and target text
        dataset = sequences.map(self.__split_input_target)
        # Shuffle training data
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        # Construct model
        model = self.__build_model(len(self.vocab), self.BATCH_SIZE)
        # Configure training procedure
        model.compile(optimizer='adam', loss=self.__loss)
        # Ensure checkpoints are stored
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.CHECKPOINT_PREFIX,
            save_weights_only=True
        )
        # Execute training
        model.fit(dataset, epochs=self.EPOCHS, callbacks=[checkpoint_callback])

    def generate_text(self, start_string):
        if not self.model:
            raise Exception('Checkpoint for restoring model does not exist in \'{}\''.format(
                self.CHECKPOINT_DIR
            ))
        input_vector = [self.char_to_index[c] for c in start_string]
        input_vector = tf.expand_dims(input_vector, 0)
        payload = ''
        self.model.reset_states()
        for _ in range(self.CHARS_TO_GENERATE):
            predictions = self.model(input_vector)
            predictions = tf.squeeze(predictions, 0) / self.TEMPERATURE
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_vector = tf.expand_dims([predicted_id], 0)
            payload += self.index_to_char[predicted_id]
        return payload.strip()
