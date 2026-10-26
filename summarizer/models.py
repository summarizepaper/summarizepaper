import json

from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.contrib.auth.models import AbstractUser

class PaymentPlan(models.Model):
    DURATION_CHOICES = [
        ('1M', '1 Month'),
        ('1Y', '1 Year'),
    ]
    name = models.CharField(max_length=50)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    duration_choices = models.CharField(max_length=2, choices=DURATION_CHOICES)
    auto_renew = models.BooleanField(default=False)

class CustomUser(User):
    pass
    '''
    id = models.AutoField(primary_key=True)
    #user = models.OneToOneField(User, blank=True, null=True, on_delete=models.CASCADE)
    payment_plan = models.ForeignKey(PaymentPlan, on_delete=models.CASCADE, blank=True, null=True)
    paid = models.BooleanField(default=False)

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_users',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        verbose_name='groups',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_users',
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )
    '''
class UserPlan(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User,blank=True, null=True, on_delete=models.CASCADE)
    payment_plan = models.ForeignKey(PaymentPlan, on_delete=models.CASCADE, blank=True, null=True)
    paid = models.BooleanField(default=False)
    #end of the paid period; None = no end (plans granted by hand in the admin)
    expires = models.DateTimeField(blank=True, null=True)
    #the subscriber cancelled: still paid until `expires`, not renewed after
    canceling = models.BooleanField(default=False)
    #which subscription: Premium+ also opens premium_tools.PLUS_TOOLS; set from the Stripe price (billing.sync_subscription)
    tier = models.CharField(max_length=10, default='premium', choices=(('premium', 'Premium'), ('plus', 'Premium+')))
    stripe_customer_id = models.CharField(max_length=100, blank=True, default='')
    stripe_subscription_id = models.CharField(max_length=100, blank=True, default='', db_index=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

class PremiumGrant(models.Model):
    """An explicit administrative entitlement, independent of Stripe subscriptions."""
    from .premium_tools import TOOL_CHOICES

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='premium_grants', verbose_name='utilisateur')
    tool = models.CharField('accès accordé', max_length=32, choices=TOOL_CHOICES,
                           help_text='Premium+ complet ouvre tous les outils ; Premium complet tous sauf ceux de Premium+. Sinon, seul l’outil choisi est débloqué.')
    active = models.BooleanField('actif', default=True, help_text='Décochez pour révoquer cet accès offert.')
    expires = models.DateTimeField('date de fin', blank=True, null=True,
                                  help_text='Facultatif : sans date, l’accès reste valable jusqu’à sa révocation.')
    note = models.TextField('note interne', blank=True, help_text='Motif, partenariat ou référence interne ; non visible par l’utilisateur.')
    granted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                                   related_name='premium_grants_issued', editable=False, verbose_name='accordé par')
    created = models.DateTimeField('créé le', auto_now_add=True)
    updated = models.DateTimeField('modifié le', auto_now=True)

    class Meta:
        verbose_name = 'accès Premium offert'
        verbose_name_plural = 'accès Premium offerts'
        constraints = [models.UniqueConstraint(fields=('user', 'tool'), name='unique_user_premium_grant')]
        ordering = ('user__username', 'tool')

    def __str__(self):
        return '%s — %s' % (self.user, self.get_tool_display())


#class CustomUser(User):#add here if want to add CB info
    #address = models.CharField(max_length=100, blank=True)
    #phone_number = models.CharField(max_length=20, blank=True)
#    payment_plan = models.ForeignKey(PaymentPlan, on_delete=models.CASCADE, blank=True, null=True)
#    paid = models.BooleanField(default=False)
    #pass

class Author(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200)
    affiliation = models.CharField(max_length=300, blank=True, null=True)

    def __str__(self):
        return self.name+' '+self.affiliation

class PaperHistory(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User,blank=True, null=True, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)
    lang = models.CharField(max_length=10,default='en')

    def __str__(self):
        return self.arxiv_id#+' '+self.user.username

class PDFHistory(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User,blank=True, null=True, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)
    lang = models.CharField(max_length=10,default='en')

    def __str__(self):
        return self.arxiv_id#+' '+self.user.username


class ArxivPaper(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=100, unique=True)
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
    category = models.CharField(max_length=100, blank=True, null=True)
    updated_arxiv = models.DateField(blank=True, null=True)
    closest_papers = models.ManyToManyField('self', through='PaperScore', symmetrical=False, related_name='closest_to')
    #total_votes = models.IntegerField(default=0)

    def __str__(self):
        return self.arxiv_id+' '+self.title

    def get_absolute_url(self):
        return reverse('arxividpage',args=[str(self.arxiv_id)])

    #def get_absolute_url2(self):
    #    return reverse('tree',args=[str(self.arxiv_id)])

class PaperScore(models.Model):
    id = models.AutoField(primary_key=True)
    from_paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE, related_name='from_paper')
    to_paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE, related_name='to_paper')
    score = models.FloatField()
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    active = models.BooleanField(default=True)

    def __str__(self):
        return 'From:'+self.from_paper.arxiv_id+' to '+self.to_paper.arxiv_id+' with score: '+str(self.score)

class Search(models.Model):
    id = models.AutoField(primary_key=True)
    query = models.TextField(blank=True, null=True)
    lang = models.CharField(max_length=10,default='en')
    created = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.query

class SummaryPaper(models.Model):
    id = models.AutoField(primary_key=True)
    paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE)
    summary = models.TextField(blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    lay_summary = models.TextField(blank=True, null=True)
    blog = models.TextField(blank=True, null=True)
    keywords = models.TextField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    lang = models.CharField(max_length=10,default='en')

    def __str__(self):
        return self.paper.arxiv_id+' ('+self.lang+') '+self.paper.title


class PaperAuthor(models.Model):
    id = models.AutoField(primary_key=True)
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
    id = models.AutoField(primary_key=True)
    vote = models.SmallIntegerField(choices=VOTE_CHOICES)
    paper = models.ForeignKey(ArxivPaper, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)
    lang = models.CharField(max_length=10,default='en')
    user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.paper.arxiv_id+' '+str(self.vote)+' '+str(self.created_at)+' '+str(self.user)


class PickledData(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=100)
    docstore_pickle = models.BinaryField(editable=True)
    buffer = models.BinaryField(editable=True)
    index_to_docstore_id_pickle = models.BinaryField(editable=True)

    def __str__(self):
        return self.arxiv_id

class AIassistant(models.Model):
    id = models.AutoField(primary_key=True)
    arxiv_id = models.CharField(max_length=100)
    query = models.TextField(blank=True, null=True)
    response = models.TextField(blank=True, null=True)
    user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)
    lang = models.CharField(max_length=10,default='en')
    evidence = models.JSONField(blank=True, null=True)  # the claims of the answer and their passages (evidence.py)
    # the papers discussed (assistant.paper_set): a chat page conversation can change them along the way
    papers = models.TextField(blank=True, null=True)
    kind = models.CharField(max_length=10, default='answer', choices=[('answer', 'Answer'), ('table', 'Comparison table')])
    cost = models.PositiveSmallIntegerField(default=1)  # questions of the daily quota it used (a table: one per paper)
    model = models.CharField(max_length=60, blank=True, default='')  # the AI model that answered (activity.answer_quality)

    def __str__(self):
        return self.arxiv_id+' '+self.query

    @property
    def claims(self):
        return (self.evidence or {}).get('claims') or []

    @property
    def evidence_json(self):
        return json.dumps(self.evidence, ensure_ascii=False)


class ChatShare(models.Model):
    """A public, read-only link to a chat page conversation as it was when shared (see chat_page.share).
    The exchanges asked after upto stay private until the owner shares again; the owner can stop sharing."""
    token = models.CharField(max_length=32, unique=True)
    room = models.CharField(max_length=100)
    user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)  # the owner without an account, as AIassistant keeps it
    upto = models.DateTimeField()
    expires = models.DateTimeField(blank=True, null=True)  # the link stops working then; sharing again extends it
    created = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)

    def __str__(self):
        return '%s (%s)' % (self.token, self.room)


class ChatConversation(models.Model):
    """The name its owner gave to a chat page conversation (its room), shown in their recent conversations."""
    room = models.CharField(max_length=100)
    user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
    ip_address = models.TextField(blank=True, null=True)  # the owner without an account, as AIassistant keeps it
    title = models.CharField(max_length=120)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class DigestSubscription(models.Model):
    """Email of new arXiv papers in the categories a user follows (Premium users may have several)."""
    FREQUENCIES = [('daily', 'Every weekday'), ('weekly', 'Once a week')]
    LEVELS = [('expert', 'Researcher'), ('general', 'Plain language')]
    # 'top': the best papers up to the plan's limit; the others (Premium, with a profile): every paper at least this close
    VOLUMES = [('top', 'Best papers'), ('high', 'Strict'), ('medium', 'Normal'), ('low', 'Broad')]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='digests')
    name = models.CharField(max_length=80, blank=True, default='')
    categories = models.CharField(max_length=1500)  # comma-separated, e.g. "astro-ph.EP,cs.LG,biorxiv.neuroscience"
    keywords = models.CharField(max_length=500, blank=True, default='')  # comma-separated, optional filter
    lang = models.CharField(max_length=10, default='en')
    frequency = models.CharField(max_length=10, choices=FREQUENCIES, default='daily')
    weekday = models.PositiveSmallIntegerField(default=0)  # weekly digests: 0 = Monday ... 4 = Friday
    send_hour = models.PositiveSmallIntegerField(default=7)  # local hour, 0-23
    timezone = models.CharField(max_length=64, default='Europe/Paris')  # IANA name
    # Premium: papers ranked by closeness to a research profile, and followed authors
    profile = models.TextField(blank=True, default='')  # a few sentences describing the research
    reference_papers = models.CharField(max_length=500, blank=True, default='')  # comma-separated arXiv ids
    authors = models.CharField(max_length=1000, blank=True, default='')  # comma-separated full names
    profile_vector = models.BinaryField(blank=True, null=True)  # embedding of profile + reference papers
    profile_key = models.CharField(max_length=64, blank=True, default='')  # hash of what profile_vector was made from
    profile_labels = models.TextField(blank=True, default='')  # JSON: what each row of profile_vector stands for
    level = models.CharField(max_length=10, choices=LEVELS, default='expert')
    volume = models.CharField(max_length=10, choices=VOLUMES, default='top')
    bundle_alerts = models.BooleanField(default=False)  # Premium: scoop, citation, version and publication alerts come in it
    active = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    last_sent = models.DateTimeField(blank=True, null=True)
    last_batch = models.DateField(blank=True, null=True)  # day of the latest arXiv announcement already sent

    CATCH_UP_HOURS = 3  # a missed hourly run is caught up, but never hours late at night

    class Meta:
        ordering = ('created', 'pk')

    def is_due(self, now):
        """True during the hours after the chosen local time, if this slot was not sent yet."""
        import datetime
        from zoneinfo import ZoneInfo
        try:
            tz = ZoneInfo(self.timezone)
        except Exception:
            tz = ZoneInfo('UTC')
        local_now = now.astimezone(tz)
        delay = (local_now.hour - self.send_hour) % 24
        if delay >= self.CATCH_UP_HOURS:
            return False
        slot_start = (local_now - datetime.timedelta(hours=delay)).replace(minute=0, second=0, microsecond=0)
        if slot_start.weekday() >= 5:  # arXiv announces nothing new for Saturday and Sunday
            return False
        if self.frequency == 'weekly' and slot_start.weekday() != self.weekday:
            return False
        return self.last_sent is None or self.last_sent < slot_start

    def hours_late(self, now):
        """Hours since this digest's local send time today (0 to 23)."""
        from zoneinfo import ZoneInfo
        try:
            tz = ZoneInfo(self.timezone)
        except Exception:
            tz = ZoneInfo('UTC')
        return (now.astimezone(tz).hour - self.send_hour) % 24

    def category_list(self):
        return [c for c in (x.strip() for x in self.categories.split(',')) if c]

    def keyword_list(self):
        return [k for k in (x.strip().lower() for x in self.keywords.split(',')) if k]

    def reference_list(self):
        return [r for r in (x.strip() for x in self.reference_papers.split(',')) if r]

    def author_list(self):
        return [a for a in (x.strip() for x in self.authors.split(',')) if a]

    def __str__(self):
        return '%s: %s' % (self.user.username, self.name or self.categories)


class DigestItem(models.Model):
    """A new paper with its short plain-language blurb, generated once per language."""
    arxiv_id = models.CharField(max_length=100)
    lang = models.CharField(max_length=10, default='en')
    title = models.TextField()
    authors = models.TextField(blank=True, default='')
    categories = models.CharField(max_length=300, blank=True, default='')
    published = models.DateTimeField(blank=True, null=True)
    blurb = models.TextField()
    points = models.TextField(blank=True, default='')  # JSON {"contribution", "method", "result"} for researchers
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('arxiv_id', 'lang')

    def __str__(self):
        return '%s (%s)' % (self.arxiv_id, self.lang)


class DigestDelivery(models.Model):
    """Which paper was already sent to which subscriber, so no paper is sent twice."""
    subscription = models.ForeignKey(DigestSubscription, on_delete=models.CASCADE, related_name='deliveries')
    arxiv_id = models.CharField(max_length=100)
    sent = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('subscription', 'arxiv_id')


class DigestVector(models.Model):
    """Embedding of a new paper's title and abstract, to rank it against Premium research profiles."""
    arxiv_id = models.CharField(max_length=100, unique=True)
    vector = models.BinaryField()
    created = models.DateTimeField(auto_now_add=True, db_index=True)


class ArxivBatch(models.Model):
    """One daily arXiv announcement, stored once every archive feed of that day was read (or, late, most of them)."""
    day = models.DateField(unique=True)  # the date arXiv gives the announcement (New York time)
    papers = models.PositiveIntegerField(default=0)
    # archives still unread when the batch was accepted without them (see digest.PARTIAL_AFTER)
    missing = models.CharField(max_length=300, blank=True, default='')
    fetched = models.DateTimeField(auto_now_add=True)

    def missing_list(self):
        return [a for a in self.missing.split(',') if a]

    def __str__(self):
        return '%s (%d papers)' % (self.day, self.papers)


class AnnouncedPaper(models.Model):
    """A new or cross-listed paper of a daily arXiv announcement, from its RSS feeds."""
    arxiv_id = models.CharField(max_length=100, unique=True)
    announced = models.DateField(db_index=True)
    title = models.TextField()
    abstract = models.TextField(blank=True, default='')
    authors = models.TextField(blank=True, default='')  # JSON list of names
    categories = models.CharField(max_length=300, blank=True, default='')  # comma-separated

    def __str__(self):
        return '%s (%s)' % (self.arxiv_id, self.announced)


class DigestFeedback(models.Model):
    """"More like this" (+1) or "less like this" (-1) on a paper of a digest, or a click on it (+1), steering its ranking."""
    SOURCES = [('vote', 'More or less like this'), ('click', 'Opened from the email')]
    subscription = models.ForeignKey(DigestSubscription, on_delete=models.CASCADE, related_name='feedback')
    arxiv_id = models.CharField(max_length=100)
    value = models.SmallIntegerField()
    source = models.CharField(max_length=10, choices=SOURCES, default='vote')
    vector = models.BinaryField(blank=True, null=True)  # the paper's embedding, kept after DigestVector is cleaned up
    created = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('subscription', 'arxiv_id')


class DigestEdition(models.Model):
    """One digest email: the announcements it covered and the papers it showed, in order, for its web page."""
    subscription = models.ForeignKey(DigestSubscription, on_delete=models.CASCADE, related_name='editions')
    sent = models.DateTimeField(db_index=True)
    days = models.CharField(max_length=200)  # comma-separated ISO dates of the announcements
    papers = models.TextField(default='[]')  # JSON list of the arxiv_ids in the email, in order
    scores = models.TextField(default='[]')  # JSON list of their closeness (None without a ranking), same order
    candidates = models.PositiveIntegerField(default=0)  # papers of its categories that matched (keywords included)

    class Meta:
        ordering = ('-sent',)

    def day_list(self):
        import datetime
        return [datetime.date.fromisoformat(d) for d in self.days.split(',') if d]


class DigestClick(models.Model):
    """A link of a digest email followed, with the paper's rank in it, to measure how good the ranking is."""
    subscription = models.ForeignKey(DigestSubscription, on_delete=models.CASCADE, related_name='clicks')
    edition = models.ForeignKey(DigestEdition, on_delete=models.SET_NULL, null=True, blank=True, related_name='clicks')
    arxiv_id = models.CharField(max_length=100)
    rank = models.PositiveSmallIntegerField(default=0)  # 1 = first paper of the email, 0 = outside the ranked list
    target = models.CharField(max_length=10, default='summary')  # summary, abs, pdf
    burst = models.BooleanField(default=False)  # one of many links opened within seconds: a mail scanner, not a reader
    created = models.DateTimeField(auto_now_add=True, db_index=True)


class CategorySynthesis(models.Model):
    """What happened in one arXiv category over the last 7 days, written once and shared by every digest."""
    category = models.CharField(max_length=100)
    lang = models.CharField(max_length=10)
    day = models.DateField()
    content = models.TextField()  # JSON: {"trends": [{"title", "text"}], "notable": [arxiv_id, ...]}
    papers = models.PositiveIntegerField(default=0)  # how many papers it was written from
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('category', 'lang', 'day')

    def __str__(self):
        return '%s %s (%s)' % (self.category, self.day, self.lang)


class PaperText(models.Model):
    """Text of a paper as given to the language model (see paper_source.py)."""
    SOURCES = [('html', 'arXiv HTML'), ('xml', 'Publisher XML (JATS or TEI)'), ('pdf', 'PDF'),
               ('abstract', 'Metadata and abstract only'), ('index', 'Rebuilt from FAISS index'),
               ('index_kept', 'Short FAISS index text, the paper could not be read again')]
    arxiv_id = models.CharField(max_length=100, unique=True)
    text = models.TextField()
    source = models.CharField(max_length=10, choices=SOURCES)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '%s (%s)' % (self.arxiv_id, self.source)


class ExternalPaper(models.Model):
    """Metadata of a paper from another source than arXiv, as sources.py found it (refreshed once a day)."""
    key = models.CharField(max_length=100, unique=True)  # 'doi_10.1093--mnras--stad1234', 'hal_hal-01234567'...
    source = models.CharField(max_length=20)  # biorxiv, medrxiv, osf, hal, europepmc, openalex, crossref
    doi = models.CharField(max_length=300, blank=True, default='')
    metadata = models.TextField(blank=True, default='')  # JSON, see sources._paper
    fetched = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return '%s (%s)' % (self.key, self.source)


class PaperEmbedding(models.Model):
    """One vector per paper, from its title and abstract (see similar.py)."""
    arxiv_id = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=100, blank=True, db_index=True)
    model = models.CharField(max_length=60)
    vector = models.BinaryField()
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '%s (%s)' % (self.arxiv_id, self.model)


class ReactivationEmail(models.Model):
    """One-off email to accounts whose activation email never arrived (see reactivation.py)."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    email = models.EmailField()
    sent = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s (%s)' % (self.email, self.sent)


class SuppressedEmail(models.Model):
    """Address we never write to again: hard bounce or complaint (see mailcheck.py)."""
    email = models.EmailField(unique=True)
    reason = models.CharField(max_length=200, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s (%s)' % (self.email, self.reason)


class SiteSettings(models.Model):
    """Quotas and AI models, edited in the admin. A single row; read it with SiteSettings.current()."""
    free_runs_per_day = models.PositiveIntegerField('résumés par jour (gratuit)', default=10,
        help_text='Nouveaux résumés par jour et par IP sans Premium. Premium : illimité.')
    chat_anonymous_per_day = models.PositiveIntegerField('questions par jour (sans compte)', default=30)
    chat_user_per_day = models.PositiveIntegerField('questions par jour (compte gratuit)', default=100)
    chat_paid_per_day = models.PositiveIntegerField('questions par jour (Premium)', default=500)
    model_free = models.CharField('modèle IA gratuit', max_length=100, default='gpt-6-luna',
        help_text='Résumés, assistant et digest des comptes gratuits. Identifiant exact de l’API OpenAI.')
    model_paid = models.CharField('modèle IA Premium', max_length=100, default='gpt-6.1-sol',
        help_text='Résumés et assistant des comptes Premium. Identifiant exact de l’API OpenAI.')
    model_tts = models.CharField('modèle de synthèse vocale', max_length=100, default='gpt-4o-mini-tts',
        help_text='Digest audio Premium. Identifiant exact de l’API OpenAI (audio/speech).')
    tts_voice = models.CharField('voix', max_length=40, default='alloy')
    free_readings_per_day = models.PositiveIntegerField('niveaux de lecture par jour (gratuit)', default=3,
        help_text='Nouvelles versions « expliqué simplement / traduit » par jour sans Premium. Premium : illimité.')
    updated = models.DateTimeField(auto_now=True)

    _cached = None
    _cached_at = 0.0
    CACHE_SECONDS = 30

    class Meta:
        verbose_name = 'réglages du site'
        verbose_name_plural = 'réglages du site'

    def __str__(self):
        return 'Réglages du site'

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)
        SiteSettings.clear_cache()

    @classmethod
    def clear_cache(cls):
        cls._cached = None

    @classmethod
    def current(cls):
        """The settings row, cached per process for CACHE_SECONDS so other dynos follow quickly."""
        import time
        now = time.monotonic()
        if cls._cached is None or now - cls._cached_at > cls.CACHE_SECONDS:
            cls._cached = cls.objects.get_or_create(pk=1)[0]
            cls._cached_at = now
        return cls._cached


class Guide(models.Model):
    """A guide edited in the admin. It replaces content/guides/<lang>/<slug>.md when both exist (see guides.py)."""
    lang = models.CharField('langue', max_length=10, choices=[('en', 'English'), ('fr', 'Français')], default='en')
    slug = models.SlugField(max_length=200)
    title = models.CharField('titre', max_length=300)
    description = models.CharField(max_length=300, help_text='Une phrase, 160 caractères au plus (meta description).')
    body = models.TextField('contenu', help_text='Markdown.')
    date = models.DateField('publié le')
    updated = models.DateField('mis à jour le', blank=True, null=True)
    draft = models.BooleanField('brouillon', default=True, help_text='Les brouillons ne sont visibles que par le staff.')
    product = models.BooleanField('page comparatif', default=False,
        help_text='Ajoute SummarizePaper en SoftwareApplication dans le balisage (comparatifs).')
    extra = models.TextField('front matter avancé', blank=True,
        help_text='YAML optionnel pour faq, sources et dataset, comme dans les fichiers Markdown.')
    modified = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('lang', 'slug')
        ordering = ('-date',)
        verbose_name = 'article de blog'
        verbose_name_plural = 'articles de blog'

    def __str__(self):
        return '%s (%s)' % (self.title, self.lang)

    def extra_meta(self):
        import yaml
        meta = yaml.safe_load(self.extra) if self.extra.strip() else {}
        if not isinstance(meta, dict):
            raise ValueError('the advanced front matter must be a YAML mapping')
        return meta

    def clean(self):
        from django.core.exceptions import ValidationError
        try:
            self.extra_meta()
        except Exception as e:
            raise ValidationError({'extra': 'YAML invalide : %s' % e})


class Generated(models.Model):
    """A model-written result made in the background (see generated.py): press kit, reading level, comparison...

    One row per kind, key and language, shared by every reader unless `user` is set (private results).
    """
    STATUSES = [('pending', 'En cours'), ('done', 'Prêt'), ('failed', 'Échec')]
    kind = models.CharField(max_length=20)
    key = models.CharField(max_length=200)
    lang = models.CharField(max_length=10, default='en')
    user = models.ForeignKey(User, blank=True, null=True, on_delete=models.CASCADE)
    status = models.CharField(max_length=10, choices=STATUSES, default='pending')
    step = models.CharField(max_length=40, blank=True, default='')  # what the background work is doing now
    content = models.TextField(blank=True, default='')  # JSON
    error = models.TextField(blank=True, default='')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('kind', 'key', 'lang')
        indexes = [models.Index(fields=['kind', 'user', '-created'])]

    def data(self):
        import json
        return json.loads(self.content) if self.content else {}

    def __str__(self):
        return '%s %s (%s, %s)' % (self.kind, self.key, self.lang, self.status)


class Collection(models.Model):
    """A user's folder of papers, to compare them, chat with them and export them."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='paper_collections')
    name = models.CharField(max_length=120)
    template = models.CharField(max_length=20, default='general')  # the columns of the comparison table (library.TEMPLATES)
    custom_columns = models.TextField(blank=True, default='')  # JSON [{"label": ..., "hint": ...}], after the template's
    watch = models.BooleanField(default=False)  # its papers are followed: citation and version alerts (Premium)
    share_token = models.CharField(max_length=32, blank=True, default='', db_index=True)  # read-only public link
    share_expires = models.DateTimeField(blank=True, null=True)
    share_notes = models.BooleanField(default=False)  # the shared page shows the owner's notes
    invite_token = models.CharField(max_length=32, blank=True, default='', db_index=True)  # joins as an editor
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('-updated',)

    def __str__(self):
        return '%s: %s' % (self.user.username, self.name)


class CollectionPaper(models.Model):
    READING = [('', ''), ('to_read', 'To read'), ('read', 'Read'), ('key', 'Key paper')]
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name='papers')
    arxiv_id = models.CharField(max_length=100)
    title = models.TextField(blank=True, default='')
    authors = models.TextField(blank=True, default='')
    note = models.TextField(blank=True, default='')
    tags = models.CharField(max_length=300, blank=True, default='')  # comma-separated
    reading = models.CharField(max_length=10, blank=True, default='', choices=READING)
    selected = models.BooleanField(default=True)  # read by the comparison table, the open questions and the chat
    position = models.PositiveIntegerField(default=0)  # the order its owner gave; papers never moved keep 0
    added = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('collection', 'arxiv_id')
        ordering = ('position', 'added', 'pk')

    def __str__(self):
        return '%s in %s' % (self.arxiv_id, self.collection.name)


class CollectionMember(models.Model):
    """Someone who edits a collection with its owner: papers, notes, tags, order, analyses (library.editable)."""
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name='members')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='shared_collections')
    added = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('collection', 'user')
        ordering = ('added',)


class FollowedPaper(models.Model):
    """A paper whose new citations a Premium user is told about (see citations.py)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='followed_papers')
    arxiv_id = models.CharField(max_length=100)  # without version, arXiv form ('2101.00001', 'hep-th/9901001')
    title = models.TextField(blank=True, default='')
    known = models.TextField(blank=True, default='')  # JSON list of the citing papers already reported
    pending = models.TextField(blank=True, default='')  # JSON list of citing papers not emailed yet
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'arxiv_id')
        ordering = ('-created',)

    def __str__(self):
        return '%s follows %s' % (self.user.username, self.arxiv_id)


class CitationCache(models.Model):
    """The citing papers of an arXiv paper, from Semantic Scholar, refreshed once a day at most."""
    arxiv_id = models.CharField(max_length=100, unique=True)
    citing = models.TextField(blank=True, default='')  # JSON {paperId: {title, arxiv_id, date, authors, contexts, intents}}
    total = models.PositiveIntegerField(default=0)
    # 'ads' or 's2': where `citing` came from, kept even when there are no citations to tell it by their ids
    source = models.CharField(max_length=10, blank=True, default='')
    checked = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return '%s (%d citations)' % (self.arxiv_id, self.total)


class CitationAlertLog(models.Model):
    """When a user was last emailed about new citations."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    sent = models.DateTimeField()


class CitationAlertHistory(models.Model):
    """Private snapshot of a citation alert, independent of live caches and follows."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='citation_alert_history')
    sent = models.DateTimeField()
    lang = models.CharField(max_length=10)
    count = models.PositiveIntegerField(default=0)
    papers = models.JSONField(default=list)

    class Meta:
        ordering = ('-sent', '-pk')
        constraints = [models.UniqueConstraint(fields=('user', 'sent'), name='unique_citation_alert_delivery')]


class PaperAudio(models.Model):
    """A paper's title and blurb read aloud, shared by every Premium digest that lists it (see audio.py)."""
    arxiv_id = models.CharField(max_length=100)
    lang = models.CharField(max_length=10)
    mp3 = models.BinaryField()
    created = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        unique_together = ('arxiv_id', 'lang')


class ZoteroAccount(models.Model):
    """Where 'Send to Zotero' writes, for a Premium user (see exports.py)."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='zotero')
    library_id = models.CharField(max_length=40)  # the numeric userID shown on zotero.org/settings/keys
    api_key = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)


class TermDay(models.Model):
    """How many papers of one arXiv announcement used each term, for one archive or '*' (see trends.py)."""
    day = models.DateField()
    scope = models.CharField(max_length=20)
    papers = models.PositiveIntegerField(default=0)
    counts = models.TextField()  # JSON {term: papers}

    class Meta:
        unique_together = ('day', 'scope')


class RisingTopics(models.Model):
    """The topics rising in one archive (or '*') over the week before `day`, in plain language."""
    scope = models.CharField(max_length=20)
    lang = models.CharField(max_length=10)
    day = models.DateField()
    content = models.TextField()  # JSON {"topics": [{"title", "text", "terms", "growth", "papers": [{arxiv_id, title}]}]}
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('scope', 'lang', 'day')


class Letter(models.Model):
    """A weekly article on one theme ('AI this week'), written from the week's arXiv papers (see letters.py)."""
    topic = models.CharField(max_length=20)
    lang = models.CharField(max_length=10)
    day = models.DateField()  # the Monday announcement it was written after
    title = models.CharField(max_length=300)
    dek = models.TextField(blank=True, default='')
    content = models.TextField()  # JSON {"sections": [{"heading", "text", "papers": [arxiv_id]}], "closing", "papers": {id: title}}
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('topic', 'lang', 'day')
        ordering = ('-day',)

    def __str__(self):
        return '%s %s (%s)' % (self.topic, self.day, self.lang)


class LetterSubscription(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='letter_subscriptions')
    topic = models.CharField(max_length=20)
    lang = models.CharField(max_length=10, default='en')
    active = models.BooleanField(default=True)
    last_letter = models.DateField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'topic')


class WatchedProject(models.Model):
    """A Premium user's ongoing project, compared with every new announcement to catch close papers (see scoops.py)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watched_projects')
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default='')  # written by the user, or by the model from their draft
    author_name = models.CharField(max_length=200, blank=True, default='')  # papers signed by it are the user's own
    search_text = models.TextField(blank=True, default='')  # the description told as an English abstract, embedded
    lang = models.CharField(max_length=10, default='en')
    vector = models.BinaryField(blank=True, null=True)
    vector_key = models.CharField(max_length=64, blank=True, default='')
    last_day = models.DateField(blank=True, null=True)  # the last announcement compared with it; None until the first check
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return '%s: %s' % (self.user.username, self.title)


class ScoopMatch(models.Model):
    """A new paper close to a watched project, with what it shares with it and what stays the project's own."""
    project = models.ForeignKey(WatchedProject, on_delete=models.CASCADE, related_name='matches')
    arxiv_id = models.CharField(max_length=100)  # AnnouncedPaper.arxiv_id
    title = models.TextField()
    authors = models.TextField(blank=True, default='')  # JSON list of names
    announced = models.DateField()
    similarity = models.FloatField()
    overlap = models.CharField(max_length=10)  # 'high' or 'partial'
    shared = models.TextField(blank=True, default='')
    original = models.TextField(blank=True, default='')
    emailed = models.BooleanField(default=False)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('project', 'arxiv_id')
        ordering = ('-announced', '-similarity')

    def author_list(self):
        import json
        return json.loads(self.authors or '[]')


class PaperVersion(models.Model):
    """The latest arXiv version seen of a paper someone follows or keeps (see changes.py)."""
    arxiv_id = models.CharField(max_length=100, unique=True)  # without version, arXiv form ('hep-th/9901001')
    version = models.PositiveIntegerField()
    checked = models.DateTimeField()


class VersionAlert(models.Model):
    """A new version of a paper a user follows, to email with what changed."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='version_alerts')
    arxiv_id = models.CharField(max_length=100)
    old = models.PositiveIntegerField()
    new = models.PositiveIntegerField()
    emailed = models.BooleanField(default=False)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'arxiv_id', 'new')


class PublicationAlert(models.Model):
    """A paper a user follows was published in a journal, or got a retraction or correction (see publication_alerts.py)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='publication_alerts')
    arxiv_id = models.CharField(max_length=100)  # the paper as followed ('2101.00001', 'hep-th/9901001')
    paper = models.CharField(max_length=100, blank=True, default='')  # its page ('2101.00001v2')
    title = models.TextField(blank=True, default='')
    kind = models.CharField(max_length=30)  # 'published', or a notice: 'retraction', 'correction'...
    journal = models.CharField(max_length=300, blank=True, default='')
    doi = models.CharField(max_length=300, blank=True, default='')
    emailed = models.BooleanField(default=False)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'arxiv_id', 'kind', 'doi')


class AIPrice(models.Model):
    """Price of an OpenAI model, set in the admin, to turn the counted tokens into dollars (see usage.py)."""
    model = models.CharField('modèle', max_length=100, unique=True, help_text='Identifiant exact de l’API OpenAI.')
    input_per_million = models.DecimalField('$ par million de tokens en entrée', max_digits=8, decimal_places=4, default=0)
    output_per_million = models.DecimalField('$ par million de tokens en sortie', max_digits=8, decimal_places=4, default=0,
        help_text='Pour la synthèse vocale, le prix par million de caractères va en entrée.')

    class Meta:
        verbose_name = 'prix d’un modèle IA'
        verbose_name_plural = 'prix des modèles IA'

    def __str__(self):
        return self.model


class AIUsage(models.Model):
    """Tokens used by the AI in a day, for one task, model and user (see usage.py)."""
    day = models.DateField('jour', db_index=True)
    task = models.CharField('tâche', max_length=40)
    model = models.CharField('modèle', max_length=100)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='ai_usage')
    calls = models.PositiveIntegerField('appels', default=0)
    prompt_tokens = models.PositiveBigIntegerField('tokens en entrée', default=0)
    completion_tokens = models.PositiveBigIntegerField('tokens en sortie', default=0)

    class Meta:
        verbose_name = 'usage de l’IA'
        verbose_name_plural = 'usage de l’IA'
        ordering = ('-day', 'task')
        indexes = [models.Index(fields=['day', 'task', 'model', 'user'])]

    def cost(self, prices=None):
        """Dollars, from AIPrice; None when the model has no price."""
        if prices is None:
            prices = {p.model: p for p in AIPrice.objects.all()}
        price = prices.get(self.model)
        if price is None:  # OpenAI answers with the dated snapshot: gpt-6-luna-2026-08-05 is priced as gpt-6-luna
            price = next((prices[name] for name in sorted(prices, key=len, reverse=True)
                          if self.model.startswith(name + '-')), None)
        if price is None:
            return None
        return (self.prompt_tokens * float(price.input_per_million) +
                self.completion_tokens * float(price.output_per_million)) / 1e6


class AnnouncementSent(models.Model):
    """An announcement email a user got, so a command run again does not send it twice (see announce_tools)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='announcements')
    name = models.CharField(max_length=50)
    sent = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'name')


class SavedSearch(models.Model):
    """A user's validated search and optional opt-in watch of its latest matches."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_searches')
    name = models.CharField(max_length=120)
    params = models.JSONField(default=dict)
    fingerprint = models.CharField(max_length=64)
    lang = models.CharField(max_length=10, default='en')
    watch_enabled = models.BooleanField(default=False)
    frequency = models.CharField(max_length=10, choices=[('daily', 'Daily'), ('weekly', 'Weekly')], default='weekly')
    seen = models.JSONField(default=list)
    watch_claim_until = models.DateTimeField(null=True, blank=True)
    last_attempt = models.DateTimeField(null=True, blank=True)
    last_checked = models.DateTimeField(null=True, blank=True)
    last_sent = models.DateTimeField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created', '-pk')
        constraints = [models.UniqueConstraint(fields=('user', 'fingerprint'), name='unique_user_saved_search')]

    def get_absolute_url(self):
        from urllib.parse import urlencode
        from django.urls import reverse
        return reverse('search_results') + '?' + urlencode(self.params, doseq=True)


class StudentShortlist(models.Model):
    """Private, source-backed placement lead saved by its student, never an offer."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='student_shortlist')
    candidate_key = models.CharField(max_length=64)
    snapshot = models.JSONField(default=dict)
    status = models.CharField(max_length=16, default='explore', choices=[
        ('explore', 'To explore'), ('read', 'Paper read'),
        ('contacted', 'Contacted'), ('replied', 'Reply received')])
    notes = models.TextField(blank=True, default='')
    remind_on = models.DateField(null=True, blank=True)
    revision = models.PositiveIntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'candidate_key')
        ordering = ('-updated', '-pk')
