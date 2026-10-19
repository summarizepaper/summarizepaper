# arXiv Paper Summarization Website :books::robot:

A web application that uses AI to summarize academic papers from arXiv and provides a virtual assistant :robot: to answer questions about them. Live at [SummarizePaper.com](https://www.summarizepaper.com).

## Features :sparkles:

- Summaries of arXiv papers (detailed summary, key points, lay summary, blog-style article) in English and French
- An AI assistant that answers questions from the full text of the paper, with streamed answers
- Chat with several papers at once
- A daily email digest of new papers in the arXiv categories you follow
- Premium subscriptions (Stripe) for the most capable model and higher limits
- BibTeX, RIS and Markdown exports, related papers, PDF export of summaries
- Summaries rewritten for another reader (a 12-year-old to an expert) or in 16 languages
- Collections of papers, with a comparison table (Premium) and chat with a whole collection
- Premium tools: press kit with a hype check for journalists, "What did I miss?" (arXiv papers close to a draft that it does not cite), citation alerts, the digest read aloud, papers sent to Zotero with their summaries
- Rising topics on arXiv (term counts week over week) and weekly letters written like magazine articles

## Stack

Django 5.2 served by Daphne (ASGI), Channels websockets over Redis, PostgreSQL on Heroku. Python 3.11 (see `.python-version`).

| Module | Role |
| --- | --- |
| `summarizer/views.py` | Pages, search, paper page |
| `summarizer/consumers.py` | Websocket: runs summaries and the assistant |
| `summarizer/assistant.py` | Questions and detailed summaries over the full paper text |
| `summarizer/paper_source.py` | Paper text: arXiv HTML first, PDF fallback |
| `summarizer/access.py` | Client IP, paid status, quotas (never trusts the browser) |
| `summarizer/billing.py` | Stripe Checkout, Customer Portal, webhook |
| `summarizer/digest.py` | Daily digest selection, blurbs and emails |
| `summarizer/accounts.py` | Email login links (also activate new accounts), signup bot checks |
| `summarizer/generated.py` | Model-written results made in a background thread and polled by the page |
| `summarizer/reading.py`, `presskit.py` | Rewrites for another reader or language; press kit |
| `summarizer/library.py`, `exports.py` | Collections and their comparison table; BibTeX, RIS, Markdown, Zotero |
| `summarizer/missed.py`, `citations.py` | "What did I miss?"; citation alerts (Semantic Scholar) |
| `summarizer/audio.py`, `trends.py`, `letters.py` | Digest read aloud; rising topics; weekly letters |

## Local setup :floppy_disk:

```bash
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
export SECRET_KEY=dev OPENAI_KEY=sk-...
venv/bin/python manage.py migrate
venv/bin/python manage.py runserver
```

Websockets need a Redis on `127.0.0.1:6379` locally (`brew services start redis`).

Run the tests with `venv/bin/python manage.py test summarizer`.

## Configuration (Heroku config vars)

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Django secret key |
| `ON_HEROKU` | Enables production settings |
| `DATABASE_URL`, `REDISCLOUD_URL` | Set by the Postgres and Redis Cloud add-ons |
| `OPENAI_KEY` | OpenAI API key |
| `SECRETCMTA` | SMTP password (outgoing email) |
| `S2_API_KEY` | Optional Semantic Scholar key, raises the rate limit of the citation alerts |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_MONTHLY`, `STRIPE_PRICE_YEARLY` | Premium subscriptions; Premium stays closed while unset |

## Deployment

`git push heroku master`. The `release` phase of the Procfile applies migrations.

- Hourly job (Heroku Scheduler): `python manage.py send_digest`. It sends the digests that are due, then counts the
  day's terms for the rising topics, writes the weekly topics and letters, and sends citation alerts and letters
  within what is left of the hourly email budget.
- Stripe: webhook endpoint `https://www.summarizepaper.com/stripe/webhook/` with the events `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated` and `customer.subscription.deleted`.

## Contributors :octocat:

Developed by :sparkles: Quentin Kral :sparkles:. Questions or suggestions are welcome.

## License :page_with_curl:

MIT License, see the [LICENSE](LICENSE) file. :clipboard:
