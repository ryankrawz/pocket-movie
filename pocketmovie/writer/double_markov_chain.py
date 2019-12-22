from random import choice
import sys

from nltk.metrics.distance import edit_distance

from pocketmovie.enums import SentenceContext
from reader.models import Sentence, StartSymbol
from writer.sentence_generation_model import SentenceGenerationRNN
import writer.models as w_models


class DoubleMarkov:
    # Limit for input size of neural language model
    RNN_INPUT_CEILING = 100

    # Initialize context/type ngrams relevant to genre
    def __init__(self, genre, title, author, characters, start_sentence, length):
        self.title = title.strip().upper()
        self.author = author.strip()
        self.characters = [character.strip().upper() for character in characters]
        self.start_sentence = start_sentence + ' '
        self.current_context_ngram = ()
        self.current_type_ngram = ()
        self.context_unigrams = w_models.ContextUnigramKeyValue.objects.filter(genre=genre)
        self.context_bigrams = w_models.ContextBigramKeyValue.objects.filter(genre=genre)
        self.context_trigrams = w_models.ContextTrigramKeyValue.objects.filter(genre=genre)
        self.type_unigrams = w_models.TypeUnigramKeyValue.objects.filter(genre=genre)
        self.type_bigrams = w_models.TypeBigramKeyValue.objects.filter(genre=genre)
        self.type_trigrams = w_models.TypeTrigramKeyValue.objects.filter(genre=genre)
        self.sentences = Sentence.objects.filter(genre=genre)
        self.rnn = SentenceGenerationRNN()
        self.context_count_ceiling = length

    # Query for sentence with corresponding context/type
    def _get_sentence(self, all_text, current_context, current_type, current_character):
        matching_sentences = self.sentences.filter(
            sentence_context=current_context,
            sentence_type=current_type
        ).values_list(
            'text',
            flat=True
        ).distinct()
        if matching_sentences:
            current_text = self._match_sentence_to_guide(all_text, matching_sentences)
            if current_context == str(SentenceContext.DIALOGUE) and current_text:
                next_character = choice(self.characters)
                while not len(self.characters) <= 1 and next_character == current_character:
                    next_character = choice(self.characters)
                current_character = next_character
                current_text = '\n\n{0}:\n\t"{1}"\n\n'.format(current_character, current_text)
            return current_text + ' ', current_character
        return '', current_character

    @staticmethod
    # Convert start symbol counts to probabilities for scalability, return chosen type
    def _get_start_type(context):
        start_symbols = StartSymbol.objects.filter(sentence_context=context)
        total = sum([start.count for start in start_symbols])
        type_list = []
        for start in start_symbols:
            type_list += [start.sentence_type] * int((start.count / total) * 100)
        return choice(type_list)

    # Identify sentence with lowest edit distance to guide sentence from neural model
    def _match_sentence_to_guide(self, all_text, matching_sentences):
        inference_text = all_text[-self.RNN_INPUT_CEILING:] if len(all_text) > self.RNN_INPUT_CEILING else all_text
        guide_text = self.rnn.generate_text(inference_text)
        current_text = ''
        current_distance = 0
        for sentence_text in matching_sentences:
            if sentence_text not in all_text:
                new_distance = edit_distance(guide_text, sentence_text)
                if not current_text or new_distance < current_distance:
                    current_text = sentence_text
                    current_distance = new_distance
        return current_text.strip()

    # Retrieve usable sentences for script from database
    def _produce_sentences(self, all_contexts):
        payload = self.start_sentence
        current_character = ''
        # Iterate through available context sequence and generate types considering identical subsequent contexts
        for index, context in enumerate(all_contexts):
            # Print progress bar to console
            sys.stdout.write('\r')
            complete = (index + 1) / self.context_count_ceiling
            sys.stdout.write('Generating script: [%-50s] %.1f%%' % ('=' * int(50 * complete), 100 * complete))
            if complete == 1:
                sys.stdout.write('\n')
            sys.stdout.flush()
            try:
                if index == 0 or context != all_contexts[index - 1]:
                    self.current_type_ngram = (self._get_start_type(context),)
                elif len(self.current_type_ngram) == 1:
                    target_type = self._weighted_random(self.type_bigrams.filter(
                        gram_1=self.current_type_ngram[0]
                    ))
                    self.current_type_ngram = (target_type.gram_1, target_type.gram_2)
                elif len(self.current_type_ngram) == 2:
                    target_type = self._weighted_random(self.type_trigrams.filter(
                        gram_1=self.current_type_ngram[0],
                        gram_2=self.current_type_ngram[1]
                    ))
                    self.current_type_ngram = (target_type.gram_1, target_type.gram_2, target_type.gram_3)
                elif len(self.current_type_ngram) == 3:
                    target_type = self._weighted_random(self.type_trigrams.filter(
                        gram_1=self.current_type_ngram[1],
                        gram_2=self.current_type_ngram[2]
                    ))
                    self.current_type_ngram = (target_type.gram_1, target_type.gram_2, target_type.gram_3)
                else:
                    target_type = self._weighted_random(self.type_unigrams)
                    self.current_type_ngram = (target_type.gram_1,)
            except IndexError:
                target_type = self._weighted_random(self.type_unigrams)
                self.current_type_ngram = (target_type.gram_1,)
            next_sentence, current_character = self._get_sentence(
                payload,
                context,
                self.current_type_ngram[-1],
                current_character
            )
            payload += next_sentence
        return payload

    @staticmethod
    # Produce context/type by ngram probabilities
    def _weighted_random(l):
        weighted_list = []
        for item in l:
            weighted_list += [item] * int(item.probability * 1000)
        return choice(weighted_list)

    # Draft movie script based on Markov chain probability
    def generate_output(self):
        full_context_sequence = []
        while len(full_context_sequence) < self.context_count_ceiling:
            try:
                if len(self.current_context_ngram) == 1:
                    target_context = self._weighted_random(self.context_bigrams.filter(
                        gram_1=self.current_context_ngram[0]
                    ))
                    self.current_context_ngram = (target_context.gram_1, target_context.gram_2)
                elif len(self.current_context_ngram) == 2:
                    target_context = self._weighted_random(self.context_trigrams.filter(
                        gram_1=self.current_context_ngram[0],
                        gram_2=self.current_context_ngram[1]
                    ))
                    self.current_context_ngram = (target_context.gram_1, target_context.gram_2, target_context.gram_3)
                elif len(self.current_context_ngram) == 3:
                    target_context = self._weighted_random(self.context_trigrams.filter(
                        gram_1=self.current_context_ngram[1],
                        gram_2=self.current_context_ngram[2]
                    ))
                    self.current_context_ngram = (target_context.gram_1, target_context.gram_2, target_context.gram_3)
                else:
                    target_context = self._weighted_random(self.context_unigrams)
                    self.current_context_ngram = (target_context.gram_1,)
            except IndexError:
                target_context = self._weighted_random(self.context_unigrams)
                self.current_context_ngram = (target_context.gram_1,)
            full_context_sequence.append(self.current_context_ngram[-1])
        return '{0}\n\nby: {1}\n\n\n\n\n\n-- Fade in from black --\n\n\n{2}\n\n\n-- End scene --\n'.format(
            self.title,
            self.author,
            self._produce_sentences(full_context_sequence)
        )
