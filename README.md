# arXiv Paper Summarization Website :books::robot:

A web application that uses AI to summarize academic papers from arXiv and provides a virtual assistant :robot: to answer questions about them. Live at [SummarizePaper.com](https://www.summarizepaper.com).

## Features :sparkles:

- Summaries of arXiv papers (detailed summary, key points, lay summary, blog-style article) in English and French
- An AI assistant that answers questions from the full text of the paper, with streamed answers
- Chat with several papers at once
- A daily email digest of new papers in the arXiv categories you follow
- Premium subscriptions (Stripe) for the most capable model and higher limits
- BibTeX export, related papers, PDF export of summaries

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
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_MONTHLY`, `STRIPE_PRICE_YEARLY` | Premium subscriptions; Premium stays closed while unset |

## Deployment

`git push heroku master`. The `release` phase of the Procfile applies migrations.

- Daily digest: Heroku Scheduler job `python manage.py send_digest`, once a day.
- Stripe: webhook endpoint `https://www.summarizepaper.com/stripe/webhook/` with the events `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated` and `customer.subscription.deleted`.

## Contributors :octocat:

Developed by :sparkles: Quentin Kral :sparkles:. Questions or suggestions are welcome.

## License :page_with_curl:

MIT License, see the [LICENSE](LICENSE) file. :clipboard:
