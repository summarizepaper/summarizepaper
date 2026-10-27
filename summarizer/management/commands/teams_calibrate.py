"""What the map of who works on a topic shows on real topics, and how its counts compare with the old ones.

python manage.py teams_calibrate                                  # sample topics of several fields
python manage.py teams_calibrate --topic "exomoon detection" --years 5 --mode referees --orcid 0000-0002-1825-0097
python manage.py teams_calibrate --describe                       # also name the approaches (a paid model call)

For each topic, the papers are searched and screened as for a map (titles and abstracts). It prints what was read
(OpenAlex hits, papers retrieved, close, preprints merged, kept, large collaborations, alphabetical lists), the
institutions by share with their old rank by raw count, the people with their roles, and the size of each approach,
to read against what one knows of the field. A run costs the OpenAlex calls of a search, an embedding and a few
cheap model calls per topic (one paid call more with --describe), and stores nothing.
"""
from django.core.management.base import BaseCommand

from summarizer import student_papers, teams

SAMPLES = [
    'Detecting exomoons with astrometry and transit timing variations',
    'Gravitational waves from binary neutron star mergers and their electromagnetic counterparts',
    'CRISPR base editing to correct pathogenic mutations in mouse models',
    'Graph neural networks for molecular property prediction',
    'Monetary policy transmission and household inflation expectations',
]


class Command(BaseCommand):
    help = 'Map sample topics as the tool does, and print what is counted and why, to check it.'

    def add_arguments(self, parser):
        parser.add_argument('--topic', action='append', help='A topic (repeat for several); the samples otherwise.')
        parser.add_argument('--years', type=int, default=teams.YEARS, choices=teams.YEAR_CHOICES)
        parser.add_argument('--mode', default='overview', choices=teams.MODES)
        parser.add_argument('--orcid', default='', help='Mark (or leave out) this person and their co-authors.')
        parser.add_argument('--lang', default='en')
        parser.add_argument('--describe', action='store_true', help='Also name the approaches (a paid model call).')

    def handle(self, *args, **options):
        for topic in options['topic'] or SAMPLES:
            self.stdout.write(self.style.MIGRATE_HEADING('\n== %s' % topic))
            selection = teams.search_papers(topic, options['lang'], abstracts=True, years=options['years'], parts=True)
            kept = selection.get('works', [])
            for query in selection.get('queries') or []:
                self.stdout.write('query: %-50s OpenAlex finds %s' % (query, (selection.get('hits') or {}).get(query, '?')))
            self.stdout.write('retrieved %d · close %d · preprints merged %d · kept %d%s · more than %d authors %d · alphabetical %d · partial %s' % (
                selection.get('retrieved', 0), selection.get('close', 0), selection.get('duplicates', 0), len(kept),
                ' (relevance checked for the %d closest)' % teams.MAX_CANDIDATES if selection.get('truncated') else '',
                student_papers.BIG_TEAM, sum(len(w.get('authorships') or []) > student_papers.BIG_TEAM for w in kept),
                sum(student_papers.alphabetical(w) for w in kept), bool(selection.get('limited'))))
            if not kept:
                continue
            labels, members = teams.approaches(kept)
            relations = teams.conflicts([teams.orcid_of(options['orcid'])]) if teams.orcid_of(options['orcid']) else None
            if options['orcid']:
                self.stdout.write('orcid: %s' % ('unavailable' if relations is None else '%d papers, %d people marked' % (
                    relations['papers'], len(relations['authors']))))
            institutions = teams.institutions_of(kept, labels)
            old = sorted(institutions, key=lambda g: (-g['papers'], g['institution'].casefold()))
            old_rank = {g['id']: rank for rank, g in enumerate(old, 1)}
            self.stdout.write('institutions, by papers led then share (old rank by raw count):')
            for rank, g in enumerate(institutions[:teams.GROUPS], 1):
                self.stdout.write('%2d. (%2d) %-45s %-2s share %5.2f · papers %2d · lead %2d' % (
                    rank, old_rank[g['id']], g['institution'][:45], g['country'], g['share'], g['papers'], g['lead']))
            people, excluded = teams.people_of(kept, labels, relations, options['mode'])
            self.stdout.write('people (%s):' % options['mode'])
            for rank, p in enumerate(people[:20], 1):
                self.stdout.write('%2d. %-28s score %5.2f · %d papers · first %d, last %d, corresponding %d · %s%s%s' % (
                    rank, p['name'][:28], p['score'], p['papers'], p['roles']['first'], p['roles']['last'], p['roles']['corresponding'],
                    p['institution'][:40], ' · ' + p['orcid'] if p['orcid'] else '', ' · ' + p['relation'] if p['relation'] else ''))
            if excluded:
                self.stdout.write('left out: %s' % ', '.join('%s (%s)' % (p['name'], p['relation']) for p in excluded))
            if options['describe']:
                named, landscape = teams.describe(topic, kept, members, options['lang'])
                for a in named:
                    self.stdout.write('approach %d · %d papers · %s — %s' % (a['id'], a['papers'], a['name'], a['description']))
                self.stdout.write('landscape: %s' % landscape)
            else:
                by_id = {w['id']: w for w in kept}
                for label, wids in members.items():
                    self.stdout.write('approach %d · %d papers · e.g. %s' % (label, len(wids), by_id[wids[0]]['display_name'][:100]))
