from django import template
register = template.Library()

@register.filter
def dash_slash(value):
    return value.replace("--","/")


@register.filter
def paper_ref(value):
    """'2101.00001v1' -> 'arXiv: 2101.00001v1', 'doi_10.1101--x' -> 'DOI: 10.1101/x' (see sources.label)."""
    from summarizer import sources
    return sources.label(value)


@register.filter
def is_external(value):
    """Whether a paper key is from another source than arXiv."""
    from summarizer import sources
    return sources.is_external(value)


@register.filter
def paper_url(value):
    """Where a paper lives outside SummarizePaper: its arXiv abstract page or its DOI (see sources.ref_url)."""
    from summarizer import sources
    return sources.ref_url(value)


@register.filter
def category_label(value):
    """'biorxiv.cell-biology' -> 'bioRxiv: Cell Biology', 'osf.psyarxiv' -> 'PsyArXiv'; arXiv codes unchanged."""
    from summarizer import digest, sources
    archive, _, rest = (value or '').partition('.')
    if archive in digest.SERVER_NAMES:
        return '%s: %s' % (digest.SERVER_NAMES[archive], digest.category_name(value) if value in digest.BIO_CODES
                           else rest.replace('-', ' ').capitalize())
    if archive == 'osf':
        return sources.OSF_PROVIDERS.get(rest, rest or 'OSF')
    if archive == 'hal':
        return 'HAL'
    return value
