from copy import copy

from django.test import TestCase

import pocketmovie.enums as enums
import reader.sentence_population as sp


class ReaderTest(TestCase):
    def setUp(self):
        super().setUp()
        self.nlp = sp.load_spacy()
        self.us_names = sp.get_us_names()
        self.type_counts = {
            (enums.SentenceType.DECLARATIVE,): 2,
            (enums.SentenceType.DECLARATIVE, enums.SentenceType.IMPERATIVE): 4,
        }
        self.context_counts = {
            (enums.SentenceContext.DIRECTION,): 2,
            (enums.SentenceContext.DIRECTION, enums.SentenceContext.DIALOGUE): 4,
        }

    def test_back_off_ngram_exists(self):
        result = sp.back_off(self.type_counts, (enums.SentenceType.DECLARATIVE, enums.SentenceType.IMPERATIVE), 4)
        expected = 2
        self.assertEqual(result, expected)

    def test_back_off_once(self):
        result = sp.back_off(self.type_counts, (enums.SentenceType.DECLARATIVE, enums.SentenceType.EXCLAMATORY), 4)
        expected = 0.5
        self.assertEqual(result, expected)

    def test_back_off_no_ngram(self):
        result = sp.back_off(self.type_counts, (enums.SentenceType.EXCLAMATORY, enums.SentenceType.EXCLAMATORY), 4)
        expected = 0.25
        self.assertEqual(result, expected)

    def test_classify_type_interrogative(self):
        text = 'Are you going to the store?'
        result = sp.classify_type(text, self.nlp(text))
        expected = enums.SentenceType.INTERROGATIVE
        self.assertEqual(result, expected)

    def test_classify_type_exclamatory(self):
        text = 'I am going to the store!'
        result = sp.classify_type(text, self.nlp(text))
        expected = enums.SentenceType.EXCLAMATORY
        self.assertEqual(result, expected)

    def test_classify_type_imperative(self):
        text = 'Go to the store.'
        result = sp.classify_type(text, self.nlp(text))
        expected = enums.SentenceType.IMPERATIVE
        self.assertEqual(result, expected)

    def test_classify_type_declarative(self):
        text = 'I am going to the store.'
        result = sp.classify_type(text, self.nlp(text))
        expected = enums.SentenceType.DECLARATIVE
        self.assertEqual(result, expected)

    def test_contains_number_true(self):
        text = 'abc5de'
        self.assertTrue(sp.contains_number(text))

    def test_contains_number_false(self):
        text = 'abcde'
        self.assertFalse(sp.contains_number(text))

    def test_contains_name_true(self):
        text = 'No seriously, step up Kyle.'
        self.assertTrue(sp.contains_name(self.nlp(text), self.us_names))

    def test_contains_name_false(self):
        text = 'It is a human toe.'
        self.assertFalse(sp.contains_name(self.nlp(text), self.us_names))

    def test_contains_website_true(self):
        text = 'test.io'
        self.assertTrue(sp.contains_website(text))

    def test_contains_website_false(self):
        text = 'test'
        self.assertFalse(sp.contains_website(text))

    def test_count_ngrams_existing(self):
        type_counts = copy(self.type_counts)
        context_counts = copy(self.context_counts)
        sp.count_ngrams(
            type_counts,
            context_counts,
            (enums.SentenceType.DECLARATIVE,),
            (enums.SentenceContext.DIRECTION,),
            enums.SentenceType.IMPERATIVE,
            enums.SentenceContext.DIALOGUE
        )
        type_expected = {
            (enums.SentenceType.DECLARATIVE,): 3,
            (enums.SentenceType.DECLARATIVE, enums.SentenceType.IMPERATIVE): 5,
        }
        self.assertEqual(type_counts, type_expected)
        context_expected = self.context_counts = {
            (enums.SentenceContext.DIRECTION,): 3,
            (enums.SentenceContext.DIRECTION, enums.SentenceContext.DIALOGUE): 5,
        }
        self.assertEqual(context_counts, context_expected)

    def test_count_ngrams_not_existing(self):
        type_counts = copy(self.type_counts)
        context_counts = copy(self.context_counts)
        sp.count_ngrams(
            type_counts,
            context_counts,
            (enums.SentenceType.INTERROGATIVE,),
            (enums.SentenceContext.DIALOGUE,),
            enums.SentenceType.DECLARATIVE,
            enums.SentenceContext.DIRECTION
        )
        type_expected = {
            (enums.SentenceType.DECLARATIVE,): 2,
            (enums.SentenceType.DECLARATIVE, enums.SentenceType.IMPERATIVE): 4,
            (enums.SentenceType.INTERROGATIVE,): 1,
            (enums.SentenceType.INTERROGATIVE, enums.SentenceType.DECLARATIVE): 1,
        }
        self.assertEqual(type_counts, type_expected)
        context_expected = self.context_counts = {
            (enums.SentenceContext.DIRECTION,): 2,
            (enums.SentenceContext.DIRECTION, enums.SentenceContext.DIALOGUE): 4,
            (enums.SentenceContext.DIALOGUE,): 1,
            (enums.SentenceContext.DIALOGUE, enums.SentenceContext.DIRECTION): 1,
        }
        self.assertEqual(context_counts, context_expected)

    def test_has_direct_address_true(self):
        text = 'I think, therefore I am.'
        self.assertTrue(sp.has_direct_address(text))

    def test_has_direct_address_false(self):
        text = 'The cow jumped over the moon.'
        self.assertFalse(sp.has_direct_address(text))

    def test_is_actor_name_true(self):
        text = 'JOHN\n\nYou do not like flying, do you?'
        self.assertTrue(sp.is_actor_name(text))

    def test_is_actor_name_false(self):
        text = 'JOHN does not like flying.'
        self.assertFalse(sp.is_actor_name(text))

    def test_is_direction_true(self):
        text = 'JOHN does not like flying.'
        self.assertTrue(sp.is_direction(text))

    def test_is_direction_false(self):
        text = 'Someone does not like flying.'
        self.assertFalse(sp.is_direction(text))
