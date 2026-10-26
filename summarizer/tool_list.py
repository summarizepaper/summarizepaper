"""Every tool for researchers, by step of research: the /tools/ page and the
Tools menu of the header both read this list."""
from django.utils.translation import gettext_lazy as _

# (path under /<lang>/, Font Awesome icon, name, premium: False, True or 'plus', description)
_GROUPS = [
    (_('Read papers faster'), [
        ('', 'file-alt', _('Summaries'), False, _('Detailed summary, key points, lay summary and blog article of any paper, from arXiv, a journal or a preprint server, in 16 languages.')),
        ('chat/', 'comments', _('Chat with papers'), False, _('Ask questions to one paper or several at once; answers from the full text when the license allows it.')),
        ('collections/', 'layer-group', _('Collections'), False, _('Keep papers together, chat with a whole collection, and with Premium compare them in a table and find their open questions.')),
    ]),
    (_('Follow your field'), [
        ('digest/', 'envelope', _('Daily digest'), False, _('The new arXiv, bioRxiv and medRxiv papers of your fields, ranked for you, every morning.')),
        ('scoops/', 'user-secret', _('Scoop watch'), True, _('Describe your project in progress: we email you the day a new paper comes close to it.')),
        ('citations/', 'quote-right', _('Citation alerts'), True, _('Who cites your papers and what they say about them, and what changed when they get a new version.')),
        ('students/', 'graduation-cap', _('Find a research placement'), 'plus', _('Potential research contacts, supporting publications, reading suggestions and a private shortlist.')),
        ('teams/', 'users', _('Who works on it?'), True, _('Institutions and authors publishing on your topic, with supporting papers and methods.')),
        ('trends/', 'chart-line', _('Rising topics'), False, _('The terms that rise on arXiv week after week, field by field.')),
        ('letters/', 'feather-alt', _('Weekly letters'), False, _('Each week, an article per theme (AI, space, health...) on the papers that matter, for curious readers; by email with Premium.')),
        ('saved-searches/', 'bookmark', _('Saved searches'), False, _('Save a search and get an email when new papers match it: one with a free account, five with Premium.')),
    ]),
    (_('Write and submit'), [
        ('press-kit/', 'newspaper', _('Press kit'), 'plus', _('Prepare a factual press kit with verified quotes, limitations and questions for the authors.')),
        ('literature-review/', 'book-open', _('Literature review'), 'plus', _('From a research question: 15 to 30 founding and recent papers, grouped by approach, with a referenced synthesis.')),
        ('missed/', 'search', _('What did I miss?'), True, _('The papers close to your draft that it does not cite.')),
        ('refcheck/', 'tasks', _('Bibliography check'), True, _('Retracted papers, preprints published since, outdated versions, and citations the cited abstract does not support.')),
        ('integrations/', 'book', _('Zotero, Notion, Obsidian'), False, _('BibTeX, RIS and Markdown exports, and papers sent to Zotero with their summary.')),
    ]),
    (_('Present and meet'), [
        ('journal-club/', 'chalkboard-teacher', _('Journal club'), 'plus', _('Prepare slides with figures, strengths, limitations and discussion questions, then edit or export them.')),
        ('conference/', 'calendar-alt', _('Conference companion'), 'plus', _('The talks of a conference program that matter for your research, and the people to meet.')),
    ]),
]

TOOL_GROUPS = [
    {'title': title, 'tools': [{'path': path, 'icon': icon, 'name': name, 'premium': premium, 'text': text}
                               for path, icon, name, premium, text in tools]}
    for title, tools in _GROUPS]
