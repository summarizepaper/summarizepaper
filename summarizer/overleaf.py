"""Opening what we write in LaTeX as a new Overleaf project, in one click.

Overleaf has no API to write into a researcher's projects, but it creates a project from what a page posts to
https://www.overleaf.com/docs (https://www.overleaf.com/devs). We post the project as a zip in a data URL
(snip_uri), so that Overleaf never has to fetch our pages, which are behind a login: a page of ours posts the
form to Overleaf as soon as it loads (open_response), or the browser builds the form from a zip it was sent
(the Beamer export of the journal club, slides.js).

A collection's references can also stay in sync with a project: Overleaf links a file to an external URL and
fetches it again when asked (New file > From external URL > Refresh). library.bib_feed serves the .bib at an
address only the collection's owner knows.
"""
import base64
import io
import zipfile

from django.shortcuts import render
from django.utils.translation import gettext as _

DOCS_URL = 'https://www.overleaf.com/docs'
MAX_ZIP = 8 * 1024 * 1024  # a data URL is about a third larger; Overleaf reads it from a form


def project_zip(files):
    """files: {name: str or bytes}, as the project's files."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content.encode('utf-8') if isinstance(content, str) else content)
    return out.getvalue()


def data_url(data):
    return 'data:application/zip;base64,' + base64.b64encode(data).decode('ascii')


def open_response(request, files, name, main='main.tex', engine='pdflatex'):
    """A page that posts the project to Overleaf when it loads, with a button when scripts are off."""
    return render(request, 'summarizer/overleaf_open.html', {
        'docs_url': DOCS_URL, 'snip_uri': data_url(project_zip(files)), 'name': name, 'main': main, 'engine': engine})


def tex(text):
    """Plain text in LaTeX: its special characters escaped."""
    out = []
    for c in ' '.join(str(text or '').split()):
        if c in '&%$#_{}':
            out.append('\\' + c)
        elif c == '\\':
            out.append('\\textbackslash{}')
        elif c == '~':
            out.append('\\textasciitilde{}')
        elif c == '^':
            out.append('\\textasciicircum{}')
        else:
            out.append(c)
    return ''.join(out)


def bib_project(bib, title, intro, bib_name='references'):
    """A project that prints every entry of a .bib (\\nocite{*}): to see it compile, and to copy the .bib."""
    main = '\n'.join([
        '%% ' + _('Written by SummarizePaper: every entry of %(file)s.bib is printed below.') % {'file': bib_name},
        '\\documentclass{article}', '\\usepackage[T1]{fontenc}', '\\usepackage[utf8]{inputenc}',
        '\\usepackage[numbers]{natbib}', '\\usepackage[hidelinks]{hyperref}', '',
        '\\title{%s}' % tex(title), '\\date{}', '\\begin{document}', '\\maketitle', '', tex(intro), '',
        '\\nocite{*}', '\\bibliographystyle{plainnat}', '\\bibliography{%s}' % bib_name, '\\end{document}', ''])
    return {'main.tex': main, bib_name + '.bib': bib}
