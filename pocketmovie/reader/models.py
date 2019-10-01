from django.db import models

from pocketmovie import enums


class Sentence(models.Model):
    text = models.CharField(max_length=300)
    genre = models.CharField(
        max_length=20,
        choices=[(g, g.value) for g in enums.Genre]
    )
    sentence_context = models.CharField(
        max_length=20,
        choices=[(c, c.value) for c in enums.SentenceContext]
    )
    sentence_type = models.CharField(
        max_length=20,
        choices=[(t, t.value) for t in enums.SentenceType]
    )

    def __str__(self):
        return self.text
