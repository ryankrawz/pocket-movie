from django.test import TestCase
import numpy as np

from pocketmovie import enums
from reader.models import Sentence, StartSymbol
from writer.double_markov_chain import DoubleMarkov
from writer.models import ContextUnigramKeyValue, TypeUnigramKeyValue
from writer.sentence_generation_model import SentenceGenerationRNN


class WriterTest(TestCase):
    def setUp(self):
        super().setUp()
        Sentence.objects.get_or_create(
            text='Death goes by many names.',
            genre=enums.Genre.HORROR,
            sentence_context=enums.SentenceContext.DIALOGUE,
            sentence_type=enums.SentenceType.DECLARATIVE,
        )
        StartSymbol.objects.get_or_create(
            sentence_context=enums.SentenceContext.DIALOGUE,
            sentence_type=enums.SentenceType.DECLARATIVE,
            count=1,
        )
        ContextUnigramKeyValue.objects.get_or_create(
            genre=enums.Genre.HORROR,
            gram_1=enums.SentenceContext.DIALOGUE,
            probability=1,
        )
        TypeUnigramKeyValue.objects.get_or_create(
            genre=enums.Genre.HORROR,
            gram_1=enums.SentenceType.DECLARATIVE,
            probability=1,
        )
        self.rnn = SentenceGenerationRNN()
        self.markov = DoubleMarkov(
            enums.Genre.HORROR,
            'Big Scary',
            'Unit Test',
            ['Gunther'],
            'It was a dark and stormy night.',
            1
        )

    def test_map_chars(self):
        vocab = sorted({'a', 'b', 'c'})
        result = self.rnn._map_chars(vocab)
        result = (result[0],) + (result[1].tolist(),)
        expected = ({'a': 0, 'b': 1, 'c': 2}, np.array(vocab).tolist())
        self.assertEqual(result, expected)

    def test_split_input_target(self):
        sequence = ['a', 'b', 'c', 'd', 'e']
        result = self.rnn._split_input_target(sequence)
        expected = (['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'])
        self.assertEqual(result, expected)

    def test_get_sentence(self):
        result = self.markov._get_sentence(
            'It was a dark and stormy night. ',
            str(enums.SentenceContext.DIALOGUE),
            enums.SentenceType.DECLARATIVE,
            'GUNTHER'
        )
        expected = ('\n\nGUNTHER:\n\t"Death goes by many names."\n\n ', 'GUNTHER')
        self.assertEqual(result, expected)

    def test_get_start_type(self):
        result = self.markov._get_start_type(enums.SentenceContext.DIALOGUE)
        expected = str(enums.SentenceType.DECLARATIVE)
        self.assertEqual(result, expected)

    def test_match_sentence_to_guide(self):
        result = self.markov._match_sentence_to_guide(
            'It was a dark and stormy night. ',
            ['Death goes by many names.']
        )
        expected = 'Death goes by many names.'
        self.assertEqual(result, expected)

    def test_produce_sentences(self):
        result = self.markov._produce_sentences([str(enums.SentenceContext.DIALOGUE)])
        expected = 'It was a dark and stormy night. \n\nGUNTHER:\n\t"Death goes by many names."\n\n '
        self.assertEqual(result, expected)

    def test_generate_output(self):
        result = self.markov.generate_output()
        expected = 'BIG SCARY\n\nby: Unit Test\n\n\n\n\n\n-- Fade in from black --\n\n\n' \
                   'It was a dark and stormy night. \n\nGUNTHER:\n\t"Death goes by many names."' \
                   '\n\n \n\n\n-- End scene --\n'
        self.assertEqual(result, expected)
