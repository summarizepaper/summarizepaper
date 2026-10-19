"""Key points as plain text when stored, and as safe HTML when shown.

The model tends to answer in markdown ("- **Label:** text"). Stored points are
plain text; the page shows a short leading label in bold. Points stored before
this cleanup still contain markdown, so the display path cleans them too.
"""
import re

from django.utils.html import escape

BULLET = re.compile(r'^\s*(?:[-•*+–]|\d{1,2}[.)])\s+')
LABEL = re.compile(r'^([^:.!?]{2,60}?\s?:)\s+(\S.*)$', re.S)


def clean_point(line):
    line = BULLET.sub('', line.strip())
    line = line.replace('**', '').replace('__', '')
    line = re.sub(r'^#+\s*', '', line)
    return re.sub(r'\s+', ' ', line).strip()


def is_heading(point):
    """'Key points:' / 'Points clés :' and the like, with nothing after the colon."""
    return point.endswith(':') and len(point) < 60


def clean_points(text):
    points = [clean_point(line) for line in text.splitlines()]
    return [p for p in points if p and not is_heading(p)]


def point_html(point):
    point = clean_point(point or '')
    match = LABEL.match(point)
    if match:
        return '<strong>%s</strong> %s' % (escape(match.group(1)), escape(match.group(2)))
    return escape(point)


def stored_points(notes):
    """The key points of a SummaryPaper: stored as the repr of a Python list, or as plain lines."""
    import ast
    if not notes or notes.startswith('Error'):
        return []
    try:
        value = ast.literal_eval(notes)
    except (ValueError, SyntaxError):
        return clean_points(notes)
    if isinstance(value, str):
        return clean_points(value)
    points = [clean_point(str(p)) for p in value] if isinstance(value, (list, tuple)) else []
    return [p for p in points if p and not is_heading(p)]
