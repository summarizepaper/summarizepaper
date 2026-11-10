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
        ('digest/', 'envelope', _('Daily paper alert'), False, _('The new arXiv, bioRxiv, medRxiv and journal papers of your fields, ranked for you, every morning.')),
        ('scoops/', 'user-secret', _('Scoop watch'), True, _('Describe your project in progress: we email you the day a new paper comes close to it.')),
        ('citations/', 'quote-right', _('Citation alerts'), True, _('Who cites your papers and what they do with them, with the citing sentence and self-citations apart, and what changed when they get a new version.')),
        ('students/', 'graduation-cap', _('Find a research placement'), 'plus', _('Who signs recent papers on your topic and how, their team and acknowledged funding, readings, a first email, and your applications followed with reminders.')),
        ('teams/', 'users', _('Who works on it?'), True, _('The people and institutions who sign the recent papers on your topic, how they sign them and with which approaches; referees without your co-authors, and the new names every month.')),
        ('trends/', 'chart-line', _('Rising topics'), False, _('The topics rising on arXiv, bioRxiv and medRxiv each week, field by field, from several teams, with their history; and whether your own topic is rising.')),
        ('letters/', 'feather-alt', _('Weekly letters'), False, _('What happens next door: each week, an article per field on the papers that matter beyond it, each number checked against the abstracts; on the web, by RSS, one by email for free.')),
        ('saved-searches/', 'bookmark', _('Saved searches'), False, _('Save a search: every new arXiv, bioRxiv, medRxiv and journal paper that matches it, ranked, by email, in your paper alert or by RSS, kept on its page. One watch free, five with Premium.')),
    ]),
    (_('Write and submit'), [
        ('press-kit/', 'newspaper', _('Press kit'), 'plus', _('Talk about a paper in public without overstating it: a draft release and the questions journalists will ask for its authors, strengths and weaknesses for a researcher asked to comment, reporting angles and questions for the authors for a journalist, every quote checked against the paper.')),
        ('literature-review/', 'book-open', _('Literature review'), 'plus', _('From a research question or a collection: 15 to 30 founding and recent papers, grouped by approach, each statement with the sentences that support it; to adjust, update and export to BibTeX, LaTeX or Zotero.')),
        ('missed/', 'search', _('What did I miss?'), True, _('The papers close to your draft that it does not cite, even those written with other words: found through your references, each with where to cite it; to export, and to search again before resubmitting.')),
        ('refcheck/', 'tasks', _('Bibliography check'), True, _('Retracted and corrected papers, preprints published since, entries to correct and duplicates, and citations the cited papers do not support, with the passage; then your references watched until publication.')),
        ('integrations/', 'book', _('Zotero, Overleaf, Notion, Obsidian'), False, _('Exports to BibTeX, RIS, Obsidian and Notion; a collection’s .bib kept up to date in Overleaf, and your results opened there; with Premium, a collection kept in sync with Zotero, and your bibliography check and literature review written into it.')),
    ]),
    (_('Present and meet'), [
        ('journal-club/', 'chalkboard-teacher', _('Journal club'), 'plus', _('Slides for your audience and the length of your talk, each point with its passage of the paper, the earlier work, where the paper stands and the hard questions to prepare; edit, present, share or export. For one paper or the papers your team proposes.')),
        ('conference/', 'calendar-alt', _('Conference companion'), 'plus', _('The talks of a conference program that matter for your research, day by day with their clashes and their preprints, read whole from Indico or a calendar feed; the people to meet, a calendar, a team that sees who goes where, and the follow-up after.')),
    ]),
]

_GUIDE_SLUGS = {
    '': 'how-to-summarize-a-research-paper',
    'chat/': 'how-to-chat-with-research-papers',
    'collections/': 'organize-research-paper-collections',
    'digest/': 'daily-digest-research-papers',
    'scoops/': 'scoop-watch-research-project',
    'citations/': 'track-research-paper-citations',
    'students/': 'find-research-placement-contacts',
    'teams/': 'find-research-teams-by-topic',
    'trends/': 'rising-research-topics',
    'letters/': 'weekly-science-letters',
    'saved-searches/': 'saved-searches-research-alerts',
    'press-kit/': 'research-paper-press-kit',
    'refcheck/': 'overleaf-bibliography-check',
}

TOOL_GROUPS = [
    {'title': title, 'tools': [{'path': path, 'icon': icon, 'name': name, 'premium': premium, 'text': text,
                              'guide': _GUIDE_SLUGS.get(path)}
                               for path, icon, name, premium, text in tools]}
    for title, tools in _GROUPS]
