from enum import Enum


class Genre(Enum):
    ACTION = 'action'
    ROMANCE = 'romance'
    HORROR = 'horror'


class SentenceContext(Enum):
    DESCRIPTION = 'description'
    DIALOGUE = 'dialogue'
    DIRECTION = 'direction'


class SentenceType(Enum):
    DECLARATIVE = 'declarative'
    INTERROGATIVE = 'interrogative'
    EXCLAMATORY = 'exclamatory'
    IMPERATIVE = 'imperative'
