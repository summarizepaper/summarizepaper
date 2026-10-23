"""Safe presentation of legacy and newly generated plain-text summaries."""
import re

from django.utils.html import escape
from . import keypoints

PREFIX = re.compile(r'^\s*(?:résumé|summary)\s*(?::\s*|\n\s*|(?=\d+[.)]\s))', re.I)
DEFINITIONS = re.compile(r'^\s*(d[ée]finitions?|definitions?)\s*:\s*(.*)$', re.I)
ITEM = re.compile(r'^\s*(?:(\d{1,2})[.)]|([-•*+–]))\s+(.+)$')


def lay_html(value):
    """Real paragraphs/lists; never trust model HTML, preserve TeX and decimal values."""
    text = str(value or '').replace('**', '').replace('__', '')
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.M).strip()
    text = PREFIX.sub('', text, count=1).strip()
    # A former storage step glued the heading to item 1; some replies also put
    # the complete numbered list on a single line. Split only sequential items.
    first = re.match(r'^1[.)]\s', text)
    if first:
        expected = 2
        parts, start = [], 0
        for match in re.finditer(r'(?<!\S)(\d{1,2})[.)]\s+', text):
            if int(match[1]) == expected:
                parts.append(text[start:match.start()].rstrip())
                start = match.start()
                expected += 1
        if parts:
            text = '\n'.join(parts + [text[start:]])
    blocks, paragraphs, items, kind = [], [], [], None
    separated = False
    def flush_paragraph():
        if paragraphs:
            blocks.append('<p>%s</p>' % escape(' '.join(paragraphs)))
            paragraphs.clear()
    def flush_list():
        nonlocal kind
        if items:
            blocks.append('<%s>%s</%s>' % (kind, ''.join('<li>%s</li>' % keypoints.point_html(item) for item in items), kind))
            items.clear()
        kind = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            flush_paragraph()
            separated = True
            continue
        heading = DEFINITIONS.match(line)
        if heading:
            flush_paragraph(); flush_list()
            blocks.append('<h3>%s</h3>' % escape(heading[1].capitalize()))
            line = heading[2].strip()
            if not line:
                continue
        item = ITEM.match(line)
        if item:
            flush_paragraph()
            new_kind = 'ol' if item[1] else 'ul'
            if kind and new_kind != kind:
                flush_list()
            kind = new_kind
            items.append(item[3])
        elif items and not separated:
            items[-1] += ' ' + line
        else:
            flush_list()
            paragraphs.append(line)
        separated = False
    flush_paragraph(); flush_list()
    return ''.join(blocks)
