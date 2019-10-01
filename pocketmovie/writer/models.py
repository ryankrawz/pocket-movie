from django.db import models

from pocketmovie import enums


class ContextUnigramKeyValue(models.Model):
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    gram_1 = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    probability = models.FloatField()

    def joined_sequence(self):
        return self.gram_1


class ContextBigramKeyValue(models.Model):
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    gram_1 = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    gram_2 = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    probability = models.FloatField()

    def joined_sequence(self):
        return self.gram_1, self.gram_2


class ContextTrigramKeyValue(models.Model):
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    gram_1 = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    gram_2 = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    gram_3 = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    probability = models.FloatField()

    def joined_sequence(self):
        return self.gram_1, self.gram_2, self.gram_3


class TypeUnigramKeyValue(models.Model):
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    gram_1 = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )
    probability = models.FloatField()

    def joined_sequence(self):
        return self.gram_1


class TypeBigramKeyValue(models.Model):
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    gram_1 = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )
    gram_2 = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )
    probability = models.FloatField()

    def joined_sequence(self):
        return self.gram_1, self.gram_2


class TypeTrigramKeyValue(models.Model):
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    gram_1 = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )
    gram_2 = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )
    gram_3 = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )
    probability = models.FloatField()

    def joined_sequence(self):
        return self.gram_1, self.gram_2, self.gram_3
