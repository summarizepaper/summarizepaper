"""Questions and summaries over the full text of papers.

Current models read a whole paper at once, so instead of retrieving a few
FAISS chunks we put the paper text in the prompt. The paper comes first and
stays identical across the questions of a conversation, which lets OpenAI's
automatic prompt caching bill the repeated part at the cached rate.
"""
import json
import logging

import aiohttp
from asgiref.sync import sync_to_async
from django.conf import settings
from django.utils.translation import get_language_info

from .models import AIassistant, ArxivPaper, SummaryPaper
from .paper_source import arxiv_ref, paper_text
from .utils import MODEL_FREE, MODEL_PAID, chat_params

logger = logging.getLogger(__name__)

ENDPOINT = 'https://api.openai.com/v1/chat/completions'
MAX_CONTEXT_CHARS = 400_000  # about 100k tokens of paper text
HISTORY_TURNS = 5
OPEN_LICENSES = ('/by/', '/by-sa/', '/by-nc-sa/', '/zero/')

INSTRUCTIONS = """You are the SummarizePaper assistant. You explain scientific papers to researchers, students and curious readers.
The paper(s) to discuss are given above, between <paper> tags.
- Ground your answers in the paper(s). When you add general knowledge, say so.
- Point to the relevant section, figure or equation when it helps.
- Write mathematics in LaTeX between $...$ (inline) or $$...$$ (display).
- Be concise unless the user asks for detail.
- If a question has nothing to do with the paper(s), say briefly that you are here to discuss them.
- Answer in {language} unless the user asks for another language."""

RESTRICTED_NOTE = ('Only the title, authors and abstract are available for this paper: its license does not allow us '
                   'to read the full text. Say so when a question needs more than that.')


class AssistantError(Exception):
    pass


def _paper_block(paper, budget):
    """<paper> block with as much text as the budget allows."""
    authors = ', '.join(a.name for a in paper.authors.all()[:30])
    header = 'arXiv: %s\nTitle: %s\nAuthors: %s\n' % (arxiv_ref(paper.arxiv_id), ' '.join((paper.title or '').split()), authors)
    open_license = any(marker in (paper.license or '') for marker in OPEN_LICENSES)
    if open_license:
        body = paper_text(paper.arxiv_id)
    else:
        body = 'Abstract: %s\n\n%s' % (paper.abstract or '', RESTRICTED_NOTE)
    if len(body) > budget:
        summary = SummaryPaper.objects.filter(paper=paper).exclude(summary='').values_list('summary', flat=True).first()
        if summary and len(summary) < budget:
            body = 'Abstract: %s\n\nDetailed summary: %s' % (paper.abstract or '', summary)
        else:
            body = body[:budget]
    return '<paper>\n%s\n%s\n</paper>' % (header, body)


def build_context(arxiv_ids):
    papers = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=arxiv_ids).prefetch_related('authors')}
    ordered = [papers[i] for i in arxiv_ids if i in papers]
    if not ordered:
        raise AssistantError('unknown paper')
    budget = MAX_CONTEXT_CHARS // len(ordered)
    return '\n\n'.join(_paper_block(p, budget) for p in ordered)


def conversation_history(arxiv_id, lang, user, hashed_ip):
    rows = AIassistant.objects.filter(arxiv_id=arxiv_id, lang=lang, active=True)
    if user is not None and user.is_authenticated:
        rows = rows.filter(user=user)
    else:
        rows = rows.filter(user=None, ip_address=hashed_ip)
    messages = []
    for row in reversed(list(rows.order_by('-created')[:HISTORY_TURNS])):
        messages += [{'role': 'user', 'content': row.query or ''}, {'role': 'assistant', 'content': row.response or ''}]
    return messages


async def complete(messages, paid, max_tokens, on_delta=None):
    """Run a chat completion, streaming text pieces to on_delta; return the full text."""
    model = MODEL_PAID if paid else MODEL_FREE
    payload = {'model': model, 'messages': messages, 'stream': True, **chat_params(model, max_tokens)}
    headers = {'Authorization': 'Bearer %s' % settings.OPENAI_KEY}
    parts = []
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(ENDPOINT, json=payload, headers=headers) as response:
            if response.status != 200:
                raise AssistantError('OpenAI %s: %s' % (response.status, (await response.text())[:500]))
            async for raw in response.content:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                choices = json.loads(data).get('choices') or []
                delta = (choices[0].get('delta') or {}).get('content') if choices else None
                if delta:
                    parts.append(delta)
                    if on_delta is not None:
                        await on_delta(delta)
    return ''.join(parts).strip()


async def answer(arxiv_ids, lang, question, paid, user=None, hashed_ip=None, on_delta=None, conversation_id=None):
    """Answer a question about one or several papers and store the exchange.

    conversation_id keys the stored history (the page's room), it defaults to the first paper.
    """
    conversation_id = conversation_id or arxiv_ids[0]
    context = await sync_to_async(build_context)(arxiv_ids)
    history = await sync_to_async(conversation_history)(conversation_id, lang, user, hashed_ip)
    language = get_language_info(lang)['name']
    messages = [{'role': 'system', 'content': context + '\n\n' + INSTRUCTIONS.format(language=language)}]
    messages += history
    messages.append({'role': 'user', 'content': question})
    text = await complete(messages, paid, 1500, on_delta)
    await sync_to_async(AIassistant.objects.create)(
        arxiv_id=conversation_id, query=question, response=text, lang=lang, ip_address=hashed_ip,
        user=user if user is not None and user.is_authenticated else None)
    return text


async def detailed_summary(arxiv_id, lang, paid):
    """Long summary of the whole paper, used as the base of the summary page."""
    context = await sync_to_async(build_context)([arxiv_id])
    language = get_language_info(lang)['name']
    messages = [
        {'role': 'system', 'content': context},
        {'role': 'user', 'content': (
            'Write a detailed summary of this paper in %s, about 400 to 600 words, in plain paragraphs without '
            'headings or bullet points. Cover the question addressed, the method, the main results with their key '
            'numbers, and the conclusions and limitations. Do not use LaTeX.' % language)},
    ]
    return await complete(messages, paid, 1500)
