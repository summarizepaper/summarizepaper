"""Offline math typography for searchable PDF prose; never executes TeX or HTML."""
import html
import re

from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexWalker

# Only matched, explicit math delimiters are converted. Prose and stray '$' stay literal.
MATH = re.compile(r'(?<!\\)(\$\$[\s\S]+?(?<!\\)\$\$|\$(?!\$)[^$]+?(?<!\\)\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\])')


def math_html(source):
    converter = LatexNodes2Text()
    pieces, scripts = [], {}
    start = 0
    for marker in re.finditer(r'(?<!\\)[_^]', source):
        if marker.start() < start:
            continue
        node, pos, length = LatexWalker(source).get_latex_expression(marker.end())
        if node is None or not length:
            continue
        # Preserve the surrounding TeX expression, including fractions and nested groups.
        token = chr(0xE000 + len(scripts))
        while token in source or token in scripts:
            token = chr(ord(token) + 1)
        tag = 'sub' if marker.group() == '_' else 'sup'
        scripts[token] = '<%s>%s</%s>' % (tag, math_html(source[pos:pos + length]), tag)
        pieces.extend([source[start:marker.start()], token])
        start = pos + length
    pieces.append(source[start:])
    rendered = html.escape(converter.latex_to_text(''.join(pieces)))
    for token, markup in scripts.items():
        rendered = rendered.replace(token, markup)
    return rendered


def prose_html(value):
    """Return only escaped prose and renderer-owned <sub>/<sup> tags, or None."""
    matches = list(MATH.finditer(value))
    if not matches:
        return None
    result, start = [], 0
    for match in matches:
        token = match.group()
        inner = token[2:-2] if token.startswith(('$$', '\\(', '\\[')) else token[1:-1]
        try:
            rendered = math_html(inner)
        except (ValueError, IndexError, TypeError, RecursionError):
            # Malformed source must not prevent the rest of the brief being downloaded.
            rendered = html.escape(token)
        result.extend([html.escape(value[start:match.start()]), rendered])
        start = match.end()
    result.append(html.escape(value[start:]))
    return ''.join(result).replace('\n', '<br>')
