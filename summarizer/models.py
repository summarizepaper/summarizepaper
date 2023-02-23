from django.db import models
from django.contrib.auth.models import User

class CustomUser(User):#add here if want to add CB info
    #address = models.CharField(max_length=100, blank=True)
    #phone_number = models.CharField(max_length=20, blank=True)
    pass

class Author(models.Model):
    name = models.CharField(max_length=200)
    affiliation = models.CharField(max_length=300, blank=True, null=True)

    def __str__(self):
        return self.name+' '+self.affiliation

class PaperHistory(models.Model):
    arxiv_id = models.CharField(max_length=20)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.arxiv_id#+' '+self.user.username


class ArxivPaper(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=20, unique=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    title = models.CharField(max_length=400, blank=True, null=True)
    abstract = models.TextField(blank=True, null=True)
    #summary = models.TextField(blank=True, null=True)
    #notes = models.TextField(blank=True, null=True)
    #longer_summary = models.TextField(blank=True, null=True)
    #blog = models.TextField(blank=True, null=True)
    authors = models.ManyToManyField(Author, blank=True, through='PaperAuthor')
    link_doi = models.URLField(blank=True, null=True)
    link_homepage = models.URLField(blank=True, null=True)
    published_arxiv = models.DateField(blank=True, null=True)
    journal_ref = models.CharField(max_length=200, blank=True, null=True)
    comments = models.TextField(blank=True, null=True)
    license = models.CharField(max_length=400, blank=True, null=True)
    category = models.CharField(max_length=50, blank=True, null=True)
    updated_arxiv = models.DateField(blank=True, null=True)
    #total_votes = models.IntegerField(default=0)

    def __str__(self):
        return self.arxiv_id+' '+self.title

class SummaryPaper(models.Model):
    paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE)
    summary = models.TextField(blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    lay_summary = models.TextField(blank=True, null=True)
    blog = models.TextField(blank=True, null=True)
    lang = models.CharField(max_length=10,default='en')

    def __str__(self):
        return self.paper.arxiv_id+' ('+self.lang+')'


class PaperAuthor(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE)
    author_order = models.PositiveSmallIntegerField()

    class Meta:
        unique_together = ('author', 'paper')
        ordering = ['author_order']

    def __str__(self):
        return self.paper.arxiv_id+' '+self.author.name+' ('+str(self.author_order)+')'

class Vote(models.Model):
    UP = 1
    DOWN = -1
    VOTE_CHOICES = (
        (UP, 'Up'),
        (DOWN, 'Down'),
    )
    vote = models.SmallIntegerField(choices=VOTE_CHOICES)
    paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)
    lang = models.CharField(max_length=10,default='en')

    def __str__(self):
        return self.paper.arxiv_id+' '+str(self.vote)+' '+str(self.created_at)


class PickledData(models.Model):
    arxiv_id = models.CharField(max_length=20)
    docstore_pickle = models.BinaryField(editable=True)
    buffer = models.BinaryField(editable=True)
    index_to_docstore_id_pickle = models.BinaryField(editable=True)

    def __str__(self):
        return self.arxiv_id
