from django.db import models
from django.contrib.auth.models import User

class CustomUser(User):
    pass

class Author(models.Model):
    name = models.CharField(max_length=200)
    affiliation = models.CharField(max_length=300, blank=True, null=True)

    def __str__(self):
        return self.name

class ArxivPaper(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=20, unique=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    title = models.CharField(max_length=400, blank=True, null=True)
    abstract = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    longer_summary = models.TextField(blank=True, null=True)
    blog = models.TextField(blank=True, null=True)
    authors = models.ManyToManyField(Author, blank=True)
    link_doi = models.URLField(blank=True, null=True)
    link_homepage = models.URLField(blank=True, null=True)
    published_arxiv = models.DateField(blank=True, null=True)
    journal_ref = models.CharField(max_length=200, blank=True, null=True)
    comments = models.TextField(blank=True, null=True)
    license = models.CharField(max_length=400, blank=True, null=True)
    category = models.CharField(max_length=50, blank=True, null=True)
    updated_arxiv = models.DateField(blank=True, null=True)
    total_votes = models.IntegerField(default=0)

    def __str__(self):
        return self.arxiv_id


class Vote(models.Model):
    UP = 1
    DOWN = -1
    VOTE_CHOICES = (
        (UP, 'Up'),
        (DOWN, 'Down'),
    )
    vote = models.SmallIntegerField(choices=VOTE_CHOICES)
    paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE)
    ip_address = models.GenericIPAddressField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.paper.arxiv_id
