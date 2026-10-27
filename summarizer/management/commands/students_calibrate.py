"""Whom the research placement search puts first on real topics, without writing anything.

python manage.py students_calibrate                                 # sample topics of several fields, anywhere
python manage.py students_calibrate --topic "exomoon detection" --country FR --city Paris
python manage.py students_calibrate --full                          # also readings, notes, skills (model calls)

For each topic, the papers are searched and screened as for a student (titles and abstracts), then the people are
ranked. It prints how many papers were kept, how many were alphabetical lists or large collaborations, how many of
the people shown sign first, last or as corresponding author, the teams, and each person with their roles, score,
earliest year, ORCID and affiliation, to read the order against what one knows of the field. A run costs the
OpenAlex calls of a search and a few cheap model calls per topic (more with --full), and stores nothing.
"""
from django.core.management.base import BaseCommand

from summarizer import student_papers, students, teams

SAMPLES = [
    'Detecting exomoons with astrometry and transit timing variations',
    'CRISPR base editing to correct pathogenic mutations in mouse models',
    'Graph neural networks for molecular property prediction',
    'Optimal transport methods for generative modelling',
    'Perovskite solar cell stability under humidity',
    'Monetary policy transmission and household inflation expectations',
]


class Command(BaseCommand):
    help = 'Rank the people of sample topics as for a student, and print why, to check the order.'

    def add_arguments(self, parser):
        parser.add_argument('--topic', action='append', help='A topic (repeat for several); the samples otherwise.')
        parser.add_argument('--country', default='', help='ISO code, e.g. FR.')
        parser.add_argument('--city', default='')
        parser.add_argument('--level', default='M2')
        parser.add_argument('--kind', default='phd', choices=['phd', 'internship'])
        parser.add_argument('--skills', default='')
        parser.add_argument('--lang', default='en')
        parser.add_argument('--full', action='store_true', help='Run the whole search: readings, notes and skills too.')

    def handle(self, *args, **options):
        for topic in options['topic'] or SAMPLES:
            profile = {'level': options['level'], 'kind': options['kind'], 'topic': topic, 'country': options['country'],
                       'city': options['city'], 'mobility': 'open', 'start': '', 'duration': None, 'skills': options['skills']}
            self.stdout.write(self.style.MIGRATE_HEADING('\n== %s' % topic))
            if options['full']:
                result = students.find(profile, options['lang'])
                people, unchecked, limited, beyond = result['candidates'], result['unchecked'], result['limited'], result['beyond']
                self.stdout.write('queries: %s' % '; '.join(result['queries']))
            else:
                selection = teams.search_papers(topic, options['lang'], abstracts=True)
                works = selection.get('works', [])
                self.stdout.write('queries: %s' % '; '.join(selection.get('queries') or []))
                self.stdout.write('papers kept: %d · alphabetical: %d · more than %d authors: %d' % (
                    len(works), sum(student_papers.alphabetical(w) for w in works), student_papers.BIG_TEAM,
                    sum(len(w.get('authorships') or []) > student_papers.BIG_TEAM for w in works)))
                report = {}
                people, limited = students.candidates(works, profile, scores=selection.get('scores'), report=report)
                unchecked, beyond = report.get('unchecked', 0), report.get('beyond', 0)
            leads = sum(any((p.get('role_counts') or {}).values()) for p in people)
            self.stdout.write('people shown: %d · signing first, last or as corresponding author: %d · not located: %d · beyond the budget: %d · partial: %s' % (
                len(people), leads, unchecked, beyond, bool(limited)))
            for rank, person in enumerate(people, 1):
                roles = person.get('role_counts') or {}
                team = ', '.join('%s%s' % (t['name'], '' if t.get('shown') else ' (not shown)') for t in person.get('team') or [])
                self.stdout.write('%2d. %-28s score %.2f · %d papers %s · first %d, last %d, corresponding %d · since %s · %s%s' % (
                    rank, person['name'][:28], person.get('score') or 0, person.get('count') or 0, '-'.join(dict.fromkeys(person.get('years') or [])),
                    roles.get('first', 0), roles.get('last', 0), roles.get('corresponding', 0), person.get('since') or '?',
                    person.get('institution', ''), ' · ' + person['city'] if person.get('city') else ''))
                details = [d for d in (person.get('orcid'), team and 'team: ' + team,
                                       person.get('skills_found') and 'skills: ' + ', '.join(person['skills_found']),
                                       person.get('funding') and 'funding: ' + ', '.join(f['funder'] for f in person['funding'])) if d]
                if details:
                    self.stdout.write('      ' + ' · '.join(details))
                for paper in (person.get('readings') or [])[:2]:
                    self.stdout.write('      - [%s] %s' % ('/'.join(paper.get('roles') or []) or 'coauthor', paper.get('title', '')[:110]))
