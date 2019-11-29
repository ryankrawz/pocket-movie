import os
import re
import sys

import nltk
import pandas as pd
import spacy

from pocketmovie.enums import Genre, SentenceContext, SentenceType
from reader.models import Sentence, StartSymbol
import writer.models as w_models


ACTOR_NAME_REGEX = re.compile(r'([A-Z]+)[\n\r]+(.*)')
NUMBER_REGEX = re.compile(r'\d')
UPPERCASE_REGEX = re.compile(r'[A-Z]{3,}')
WEBSITE_REGEX = re.compile(r'\.(com|org|net|io|co|us)', re.IGNORECASE)
PATH_TO_SCRIPTS = 'training_data/raw_text_scripts/'


# Back off ngram degrees until existing count is found
def back_off(counts, ngram, total):
    if len(ngram) == 1:
        if ngram in counts:
            return counts[ngram] / total
        else:
            return 1 / total
    elif ngram in counts and ngram[:-1] in counts:
        return counts[ngram] / counts[ngram[:-1]]
    else:
        return back_off(counts, ngram[:-1], total)


# Return sentence type predicted by expression characteristics
def classify_type(text, doc):
    if text.strip().endswith('?'):
        return SentenceType.INTERROGATIVE
    elif text.strip().endswith('!'):
        return SentenceType.EXCLAMATORY
    elif doc[0].pos_ == 'VERB':
        return SentenceType.IMPERATIVE
    else:
        return SentenceType.DECLARATIVE


# Search for number
def contains_number(text):
    return bool(NUMBER_REGEX.search(text))


# Check for human name in words
def contains_name(doc, all_names):
    return any([(token.pos_ == 'PROPN' or
                token.text.upper() in all_names)
                for token in doc])


# Search for a URL pattern
def contains_website(text):
    return bool(WEBSITE_REGEX.search(text))


# Store new counts for ngrams as dictionary values
def count_ngrams(type_counts, context_counts, type_ngram, context_ngram, sentence_type, sentence_context):
    type_ngram += (sentence_type,)
    context_ngram += (sentence_context,)
    type_ngram = type_ngram[-3:] if len(type_ngram) > 3 else type_ngram
    context_ngram = context_ngram[-3:] if len(context_ngram) > 3 else context_ngram
    for index in range(len(type_ngram)):
        if type_ngram[:index + 1] in type_counts:
            type_counts[type_ngram[:index + 1]] += 1
        else:
            type_counts[type_ngram[:index + 1]] = 1
    for index in range(len(context_ngram)):
        if context_ngram[:index + 1] in context_counts:
            context_counts[context_ngram[:index + 1]] += 1
        else:
            context_counts[context_ngram[:index + 1]] = 1
    return type_ngram, context_ngram


# Retrieve lexicon of 94,000 most common US names, courtesy Social Security database
def get_us_names():
    us_names = list(pd.DataFrame(pd.read_csv('reader/NationalNames.csv'))['Name'])
    for index, _ in enumerate(us_names):
        us_names[index] = us_names[index].upper()
    return us_names


def has_direct_address(text):
    return any([word in text for word in ['I ', 'you', 'You', 'your', 'Your', 'we',
                                          'We', '?', 'my', 'My', 'mine', 'Mine', 'our', 'Our']])


# Sentence contains entity that is an individual name
def is_actor_name(text):
    name_search = ACTOR_NAME_REGEX.search(text)
    if name_search:
        return not (is_direction(name_search.group(2)) or name_search.group(2).strip()[0] == '(')
    return False


# Customary for directions to be in all upper case
def is_direction(text):
    return bool(UPPERCASE_REGEX.search(text))


# Retrieve Spacy linguistics analysis package
def load_spacy():
    return spacy.load('en_core_web_sm')


# Include ngram counts in database as KeyValue objects
def unpack_counts(type_counts, context_counts, total, genre):
    # Crude smoothing
    for type_count in type_counts:
        type_counts[type_count] += 1
    # Calculate ngram probabilities for types
    for type_count in type_counts:
        if len(type_count) == 1:
            w_models.TypeUnigramKeyValue.objects.get_or_create(
                genre=genre,
                gram_1=type_count[0],
                probability=back_off(type_counts, type_count, total)
            )
        elif len(type_count) == 2:
            w_models.TypeBigramKeyValue.objects.get_or_create(
                genre=genre,
                gram_1=type_count[0],
                gram_2=type_count[1],
                probability=back_off(type_counts, type_count, total)
            )
        elif len(type_count) == 3:
            w_models.TypeTrigramKeyValue.objects.get_or_create(
                genre=genre,
                gram_1=type_count[0],
                gram_2=type_count[1],
                gram_3=type_count[2],
                probability=back_off(type_counts, type_count, total)
            )
    # Crude smoothing
    for context_count in context_counts:
        context_counts[context_count] += 1
    # Calculate ngram probabilities for contexts
    for context_count in context_counts:
        if len(context_count) == 1:
            w_models.ContextUnigramKeyValue.objects.get_or_create(
                genre=genre,
                gram_1=context_count[0],
                probability=back_off(context_counts, context_count, total)
            )
        elif len(context_count) == 2:
            w_models.ContextBigramKeyValue.objects.get_or_create(
                genre=genre,
                gram_1=context_count[0],
                gram_2=context_count[1],
                probability=back_off(context_counts, context_count, total)
            )
        elif len(context_count) == 3:
            w_models.ContextTrigramKeyValue.objects.get_or_create(
                genre=genre,
                gram_1=context_count[0],
                gram_2=context_count[1],
                gram_3=context_count[2],
                probability=back_off(context_counts, context_count, total)
            )


def populate_script_sentences():
    nlp = load_spacy()
    us_names = get_us_names()
    # Script directories by genre for all genres in genre enum
    for genre in Genre:
        type_counts = dict()
        context_counts = dict()
        total = 0
        root = PATH_TO_SCRIPTS + genre.value + '/'
        file_names = os.listdir(root)
        # Scripts in each genre directory
        for name in file_names:
            # Print current file to console
            sys.stdout.write('\r')
            sys.stdout.write('Parsing file (%s): %s%s' % (genre.value, name, 20 * ' '))
            sys.stdout.flush()
            with open(root + name, 'r') as f:
                script_text = f.read().strip()
            # Sentence tokens of script, excluding title and ending
            sentences = nltk.sent_tokenize(script_text)[1:-1]
            total += len(sentences)
            type_ngram = ()
            context_ngram = ()
            for sentence in sentences:
                # Avoid chronological discontinuity
                if not (contains_number(sentence) or contains_website(sentence)):
                    doc = nlp(sentence)
                    sentence_type = classify_type(sentence, doc)
                    current_context = None
                    if is_actor_name(sentence):
                        # Sentence is an actor name
                        trailing_dialogue = ACTOR_NAME_REGEX.search(sentence)
                        if trailing_dialogue.group(2):
                            # Actor name is followed by dialogue
                            dialogue_text = trailing_dialogue.group(2)
                            next_sentence_type = classify_type(trailing_dialogue.group(2), doc)
                            type_ngram, context_ngram = count_ngrams(type_counts, context_counts, type_ngram,
                                                                     context_ngram, next_sentence_type,
                                                                     SentenceContext.DIALOGUE)
                            current_context = SentenceContext.DIALOGUE
                            doc = nlp(dialogue_text)
                            if not contains_name(doc, us_names):
                                Sentence.objects.get_or_create(
                                    text=dialogue_text,
                                    genre=genre,
                                    sentence_context=SentenceContext.DIALOGUE,
                                    sentence_type=next_sentence_type
                                )
                    elif is_direction(sentence):
                        # Sentence is a direction
                        type_ngram, context_ngram = count_ngrams(type_counts, context_counts, type_ngram,
                                                                 context_ngram, sentence_type,
                                                                 SentenceContext.DIRECTION)
                        current_context = SentenceContext.DIRECTION
                        if not (has_direct_address(sentence)
                                or contains_name(doc, us_names)
                                or '(' in sentence):
                            Sentence.objects.get_or_create(
                                text=sentence,
                                genre=genre,
                                sentence_context=SentenceContext.DIRECTION,
                                sentence_type=sentence_type
                            )
                    elif not contains_name(doc, us_names):
                        # Avoid social/geographic discontinuities
                        # Sentence is most likely a description
                        type_ngram, context_ngram = count_ngrams(type_counts, context_counts, type_ngram,
                                                                 context_ngram, sentence_type,
                                                                 SentenceContext.DESCRIPTION)
                        current_context = SentenceContext.DESCRIPTION
                        if not (has_direct_address(sentence) or contains_name(doc, us_names)):
                            Sentence.objects.get_or_create(
                                text=sentence,
                                genre=genre,
                                sentence_context=SentenceContext.DESCRIPTION,
                                sentence_type=sentence_type
                            )
                    if current_context:
                        # Track starting sentence types for each context
                        start, _ = StartSymbol.objects.get_or_create(
                            sentence_context=current_context,
                            sentence_type=sentence_type
                        )
                        start.count += 1
                        start.save()
        unpack_counts(type_counts, context_counts, total, genre)
    sys.stdout.write('\n')
