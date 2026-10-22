# arXiv Paper Summarization Website :books::robot:

A web application that uses AI to summarize academic papers from arXiv, journals and other preprint servers, and provides a virtual assistant :robot: to answer questions about them. Live at [SummarizePaper.com](https://www.summarizepaper.com).

## Features :sparkles:

- Summaries of arXiv papers (detailed summary, key points, lay summary, blog-style article) in English and French
- Papers from journals and other preprint servers too: paste a DOI or a bioRxiv, medRxiv, HAL, OSF (PsyArXiv...), PubMed or Europe PMC link, or search the open-access literature (OpenAlex). As for arXiv, the full text is read only under a license that allows it (CC BY, BY-SA, BY-NC, BY-NC-SA, CC0), otherwise the abstract
- An AI assistant that answers questions from the full text of the paper, with streamed answers
- Chat with several papers at once
- A daily email digest of new papers in the arXiv, bioRxiv and medRxiv categories you follow
- Premium subscriptions (Stripe) for the most capable model and higher limits
- BibTeX, RIS and Markdown exports, related papers, PDF export of summaries
- Summaries rewritten for another reader (a 12-year-old to an expert) or in 16 languages
- Collections of papers, with a comparison table and their open questions (Premium) and chat with a whole collection
- Premium tools: press kit with a hype check for journalists, "What did I miss?" (arXiv papers close to a draft that it does not cite), scoop watch (an email the day a new paper comes close to a project in progress), bibliography check (retractions, preprints published since, citations their abstract does not support), what changed between two versions of a paper (and an email when a followed paper gets one), who works on a topic (the groups that publish the most on it, OpenAlex), journal club slides, a conference companion (the talks of a program that matter for one's research), citation alerts, the digest read aloud, papers sent to Zotero with their summaries
- Rising topics on arXiv (term counts week over week) and weekly letters written like magazine articles

## Stack

Django 5.2 served by Daphne (ASGI), Channels websockets over Redis, PostgreSQL on Heroku. Python 3.12 (see `.python-version`).

| Module | Role |
| --- | --- |
| `summarizer/views.py` | Pages, search, paper page |
| `summarizer/consumers.py` | Websocket: runs summaries and the assistant |
| `summarizer/assistant.py` | Questions and detailed summaries over the full paper text |
| `summarizer/paper_source.py` | Paper text: arXiv HTML first, PDF fallback |
| `summarizer/sources.py` | Papers from other sources (DOI, bioRxiv, medRxiv, OSF, HAL, Europe PMC, OpenAlex): keys, licenses, metadata, full text, search |
| `summarizer/access.py` | Client IP, paid status, quotas (never trusts the browser) |
| `summarizer/billing.py` | Stripe Checkout, Customer Portal, webhook |
| `summarizer/digest.py` | Daily digest selection, blurbs and emails |
| `summarizer/accounts.py` | Email login links (also activate new accounts), signup bot checks |
| `summarizer/generated.py` | Model-written results made in a background thread and polled by the page |
| `summarizer/reading.py`, `presskit.py` | Rewrites for another reader or language; press kit |
| `summarizer/questions.py` | Open questions of a collection |
| `summarizer/library.py`, `exports.py` | Collections and their comparison table; BibTeX, RIS, Markdown, Zotero |
| `summarizer/missed.py`, `citations.py` | "What did I miss?"; citation alerts (Semantic Scholar) |
| `summarizer/refcheck.py` | Bibliography check of a .bib (and its .tex) |
| `summarizer/changes.py` | What changed between two arXiv versions; new version alerts |
| `summarizer/teams.py` | Who works on a topic: groups by institution, from OpenAlex |
| `summarizer/slides.py` | Journal club slides of a paper, with its figures under an open license |
| `summarizer/conference.py` | Conference companion: the talks of a program ranked for a researcher |
| `summarizer/scoops.py` | Scoop watch: projects in progress compared with each new announcement |
| `summarizer/usage.py` | AI tokens and cost by task, model, day and user (admin: usage de l’IA) |
| `summarizer/audio.py`, `trends.py`, `letters.py` | Digest read aloud; rising topics; weekly letters |

## Local setup :floppy_disk:

```bash
python3.12 -m venv venv
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
| `ADS_TOKEN` | NASA ADS API token, source of the citation alerts' citing papers (Semantic Scholar alone while unset) |
| `S2_API_KEY` | Optional Semantic Scholar key, raises the rate limit of the citation alerts |
| `OPENALEX_API_KEY` | Optional, free (openalex.org/settings/api): raises the daily budget of the journal search from $0.10 per IP to $1, and lets us read the full text OpenAlex keeps when publishers refuse scripts (MNRAS, A&A...) |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_MONTHLY`, `STRIPE_PRICE_YEARLY` | Premium subscriptions; Premium stays closed while unset |

## Premium conversion tracking

The 2026-10-21 code audit prioritised lost Premium intent after signup/activation,
pricing and subscribe buttons hidden below the feature list, and unrecoverable Checkout errors.
The home hero now links to Premium, the Premium card comes first with its prices and CTA above
its features, and signup, Google sign-in and email activation preserve a validated `next` URL.
Stripe errors display a retry message; active subscribers cannot start another purchase.

With `UMAMI_WEBSITE_ID` enabled, `static/summarizer/js/funnel.js` sends:

| Event | Meaning |
| --- | --- |
| `premium_click`, `premium_view` | Homepage/navigation interest, then pricing page |
| `premium_signup`, `signup_view`, `signup_submit`, `signup_error` | Signup entry, submission attempt or validation error |
| `activation_email_pending` | Activation instructions displayed; **not** proof of email delivery or account activation |
| `checkout_start` | Checkout form submitted, with `period` monthly/yearly; **not** proof a Stripe session was created |
| `checkout_error`, `checkout_cancel`, `checkout_return` | Stripe API failure shown, cancellation return, or success return |
| `premium_activated` | First Premium page rendered with server-confirmed paid access after Checkout started in this browser session |

Build separate Umami funnels for anonymous visitors (Premium → signup → Checkout → activation)
and existing accounts (Premium → Checkout → activation), split by language and billing period.
Returns and submitted forms must never be counted as payments. `premium_activated` is a browser
signal: it can be missed if the visitor never returns, blocks analytics, or activates on another
device. Stripe/webhooks and UserPlan remain the subscription source of truth. Compare their totals
before drawing conclusions about conversion. No production Umami dashboard or Stripe account was
available for this audit; no uplift or abandonment rate has been measured.

Analytics failures do not block forms. Event properties contain only allowlisted language, billing period, placement and tool identifiers; URLs remove
login/reset tokens and arbitrary query strings while preserving UTM campaign attribution.
Future priorities: measure stage drop-off, then test simpler signup (currently username, email and
two password fields) and a shorter benefit-led feature list. Do not remove email verification or bot
checks based on an unmeasured assumption.

Verification: `python manage.py test summarizer` and `node --test tests/funnel.test.cjs`.
French source translations are in `locale/fr/LC_MESSAGES/django.po`; compile catalogs with
`python manage.py compilemessages` when preparing translated runtime assets.

## Research companion homepage

The bilingual homepage groups tools by research task, with free paper search and Daily Digest
as entry points. Worked examples are editorial illustrations with original-paper links, not
customer results or a live digest. Premium uses the existing cached Stripe monthly price; an
unavailable quote is omitted. Existing search routes, canonical URLs and structured data remain.

Additional Umami events: `home_search_submit` (nonempty search attempt), `home_example_click`,
`home_tool_click`, `paper_result_view` (a page containing a completed summary), and
`digest_configured` (shown once after a successful subscription save). Tool links allow only
`paper`, `digest`, `refcheck`, `missed`, `scoops`; placements allow `hero`, `needs`, `examples`,
`offer`, `recent`. Research queries, manuscript text and email addresses are never event properties.
Measure distinct visitors reaching a result or saving a digest before examining Premium conversion.
Result views can recur, and digest configuration is not proof that an email was delivered.

Homepage search now defaults to a combined arXiv and journal/preprint search. Both providers run
concurrently, results are interleaved and deduplicated by DOI/arXiv identity, and successful searches
are cached for six hours (partial failures for one minute). Partial results carry an explicit notice.
Existing source-specific result URLs remain available as optional filters. arXiv IDs, abs/pdf/html
links, versions and legacy archive IDs open directly; DOI and other supported identifiers retain
the existing resolver. A fresh search from either page searches all sources by default.

Search scope is optional (`field=all`, `author`, or `title`) and persists through result filters
and pagination. Identifiers still take precedence over scope. arXiv uses `au:` / `ti:`;
OpenAlex uses quoted byline/title filters (see [search documentation](https://help.openalex.org/api/searching/)).
Crossref fallback uses `query.author` / `query.title` and verifies the requested metadata.
Caches include scope and source so author and title searches cannot share unrelated results.
Clickable homepage examples fill the field and scope without submitting. Search forms show a
translated loading state, prevent repeat submissions and reset after browser back navigation.
Verify with `node --test tests/funnel.test.cjs tests/home-search.test.cjs` and the Django suite.

The results page includes optional year bounds, relevance/newest/oldest ordering, publication type,
abstract-only search, ORCID (format and checksum validated), and up to five included and five excluded
journals. Journal suggestions use OpenAlex source IDs, display ISSN/publisher for disambiguation,
are cached, and share the existing daily search budget. Includes are OR; exclusions are AND.
All state is kept in GET parameters and pagination; changing filters starts from page one.

Date and abstract filters are sent to both providers before retrieval. The combined date ordering
sorts the returned records across providers. arXiv years refer to submission dates; indexed records
use publication dates. Journal, ORCID, published-article and review filters use indexed records,
with a visible notice; incompatible explicit arXiv-only requests show validation errors.
Crossref retains supported date/order filters during fallback, but never substitutes a broader
search for an unsupported abstract/journal/ORCID/type filter. Partial failures remain visible.
The provider results are capped at 25 each; pagination covers the retrieved matches, not the full
literature. Classification, journal assignment and ORCID coverage depend on the indexed metadata.

API references: [OpenAlex filters](https://help.openalex.org/api/filtering/),
[arXiv API](https://info.arxiv.org/help/api/user-manual.html),
[Crossref filters](https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-filters/).

## Deployment

`git push heroku master`. The `release` phase of the Procfile applies migrations.

- Hourly job (Heroku Scheduler): `python manage.py send_digest`. It sends the digests that are due, then counts the
  day's terms for the rising topics, writes the weekly topics and letters, and sends scoop alerts, citation alerts, new version alerts and letters
  within what is left of the hourly email budget.
- Stripe: webhook endpoint `https://www.summarizepaper.com/stripe/webhook/` with the events `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated` and `customer.subscription.deleted`.

## Contributors :octocat:

Developed by :sparkles: Quentin Kral :sparkles:. Questions or suggestions are welcome.

## License :page_with_curl:

MIT License, see the [LICENSE](LICENSE) file. :clipboard:
