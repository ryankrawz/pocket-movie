from random import choice

from pocketmovie.enums import SentenceContext
from reader.models import Sentence, StartSymbol
import writer.models as w_models


class DoubleMarkov:
    # Limit for length of movie script
    CONTEXT_COUNT_CEILING = 500

    # Initialize context/type ngrams relevant to genre
    def __init__(self, genre, title, author, characters):
        self.title = title.strip().upper()
        self.author = author.strip()
        self.characters = [character.strip().upper() for character in characters]
        self.current_context_ngram = ()
        self.current_type_ngram = ()
        self.context_unigrams = w_models.ContextUnigramKeyValue.objects.filter(genre=genre)
        self.context_bigrams = w_models.ContextBigramKeyValue.objects.filter(genre=genre)
        self.context_trigrams = w_models.ContextTrigramKeyValue.objects.filter(genre=genre)
        self.type_unigrams = w_models.TypeUnigramKeyValue.objects.filter(genre=genre)
        self.type_bigrams = w_models.TypeBigramKeyValue.objects.filter(genre=genre)
        self.type_trigrams = w_models.TypeTrigramKeyValue.objects.filter(genre=genre)
        self.sentences = Sentence.objects.filter(genre=genre)

    # Query for sentence with corresponding context/type
    def get_sentence(self, current_context, current_type):
        if current_context == SentenceContext.ACTOR_NAME:
            current_context = str(SentenceContext.DIALOGUE)
        matching_sentences = self.sentences.filter(
            sentence_context=current_context,
            sentence_type=current_type
        ).values_list(
            'text',
            flat=True
        ).distinct()
        if matching_sentences:
            current_text = choice(matching_sentences).strip()
            if current_context == str(SentenceContext.DIALOGUE) and current_text:
                current_text = '\n\n{0}:\n\t"{1}"\n\n'.format(choice(self.characters), current_text)
            return current_text + ' '
        return ''

    @staticmethod
    # Convert start symbol counts to probabilities for scalability, return chosen type
    def get_start_type(context):
        start_symbols = StartSymbol.objects.filter(sentence_context=context)
        total = sum([start.count for start in start_symbols])
        type_list = []
        for start in start_symbols:
            type_list += [start.sentence_type] * int((start.count / total) * 100)
        return choice(type_list)

    # Retrieve usable sentences for script from database
    def produce_sentences(self, all_contexts):
        payload = ''
        # Iterate through available context sequence and generate types considering identical subsequent contexts
        for index, context in enumerate(all_contexts):
            try:
                if index == 0 or context != all_contexts[index - 1]:
                    self.current_type_ngram = (self.get_start_type(context),)
                elif len(self.current_type_ngram) == 1:
                    target_type = self.weighted_random(self.type_bigrams.filter(
                        gram_1=self.current_type_ngram[0]
                    ))
                    self.current_type_ngram = (target_type.gram_1, target_type.gram_2)
                elif len(self.current_type_ngram) == 2:
                    target_type = self.weighted_random(self.type_trigrams.filter(
                        gram_1=self.current_type_ngram[0],
                        gram_2=self.current_type_ngram[1]
                    ))
                    self.current_type_ngram = (target_type.gram_1, target_type.gram_2, target_type.gram_3)
                elif len(self.current_type_ngram) == 3:
                    target_type = self.weighted_random(self.type_trigrams.filter(
                        gram_1=self.current_type_ngram[1],
                        gram_2=self.current_type_ngram[2]
                    ))
                    self.current_type_ngram = (target_type.gram_1, target_type.gram_2, target_type.gram_3)
                else:
                    target_type = self.weighted_random(self.type_unigrams)
                    self.current_type_ngram = (target_type.gram_1,)
            except IndexError:
                target_type = self.weighted_random(self.type_unigrams)
                self.current_type_ngram = (target_type.gram_1,)
            payload += self.get_sentence(context, self.current_type_ngram[-1])
        return payload

    @staticmethod
    # Produce context/type by ngram probabilities
    def weighted_random(l):
        weighted_list = []
        for item in l:
            weighted_list += [item] * int(item.probability * 1000)
        return choice(weighted_list)

    # Draft movie script based on Markov chain probability
    def generate_output(self):
        full_context_sequence = []
        while len(full_context_sequence) < self.CONTEXT_COUNT_CEILING:
            try:
                if len(self.current_context_ngram) == 1:
                    target_context = self.weighted_random(self.context_bigrams.filter(
                        gram_1=self.current_context_ngram[0]
                    ))
                    self.current_context_ngram = (target_context.gram_1, target_context.gram_2)
                elif len(self.current_context_ngram) == 2:
                    target_context = self.weighted_random(self.context_trigrams.filter(
                        gram_1=self.current_context_ngram[0],
                        gram_2=self.current_context_ngram[1]
                    ))
                    self.current_context_ngram = (target_context.gram_1, target_context.gram_2, target_context.gram_3)
                elif len(self.current_context_ngram) == 3:
                    target_context = self.weighted_random(self.context_trigrams.filter(
                        gram_1=self.current_context_ngram[1],
                        gram_2=self.current_context_ngram[2]
                    ))
                    self.current_context_ngram = (target_context.gram_1, target_context.gram_2, target_context.gram_3)
                else:
                    target_context = self.weighted_random(self.context_unigrams)
                    self.current_context_ngram = (target_context.gram_1,)
            except IndexError:
                target_context = self.weighted_random(self.context_unigrams)
                self.current_context_ngram = (target_context.gram_1,)
            full_context_sequence.append(self.current_context_ngram[-1])
        return '{0}\n\nby: {1}\n\n\n\n\n\n-- Fade in from black --\n\n\n{2}\n\n\n-- End scene --\n'.format(
            self.title,
            self.author,
            self.produce_sentences(full_context_sequence)
        )
