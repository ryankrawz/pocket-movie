from random import choice

from pocketmovie.enums import Genre, SentenceContext, SentenceType
from reader.models import Sentence
import writer.models as w_models


class DoubleMarkov:
    # Limit for length of movie script
    CEILING = 1500

    # Initialize context/type ngrams relevant to genre
    def __init__(self, genre):
        self.generated_script = ''
        self.current_context_ngram = ()
        self.current_type_ngram = ()
        self.context_unigrams = w_models.ContextUnigramKeyValue.objects.filter(genre=genre)
        self.context_bigrams = w_models.ContextBigramKeyValue.objects.filter(genre=genre)
        self.context_trigrams = w_models.ContextTrigramKeyValue.objects.filter(genre=genre)
        self.type_unigrams = w_models.TypeUnigramKeyValue.objects.filter(genre=genre)
        self.type_bigrams = w_models.TypeBigramKeyValue.objects.filter(genre=genre)
        self.type_trigrams = w_models.TypeTrigramKeyValue.objects.filter(genre=genre)
        self.sentences = Sentence.objects.filter(genre=genre)

    @staticmethod
    # Produce context/type by ngram probabilities
    def weighted_random(l):
        weighted_list = []
        for item in l:
            weighted_list += [item] * int(item.probability * 1000)
        return choice(weighted_list)

    # Draft movie script based on Markov chain probability
    def generate_output(self):
        while len(self.generated_script) < self.CEILING:
            if not self.current_context_ngram:
                target_context = self.weighted_random(self.context_unigrams)
                pass  # TODO
                self.current_context_ngram = (target_context.gram_1,)
            elif len(self.current_context_ngram) == 1:
                target_context = self.weighted_random(self.context_bigrams.filter(
                    gram_1=self.current_context_ngram[0]
                ))
                pass  # TODO
                self.current_context_ngram = (target_context.gram_1, target_context.gram_2)
            elif len(self.current_context_ngram) == 2:
                target_context = self.weighted_random(self.context_trigrams.filter(
                    gram_1=self.current_context_ngram[0],
                    gram_2=self.current_context_ngram[1]
                ))
                pass  # TODO
                self.current_context_ngram = (target_context.gram_2, target_context.gram_3)
