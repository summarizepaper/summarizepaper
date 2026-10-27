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
| `summarizer/digest_web.py`, `digest_bundle.py` | Web page of each digest email, counted links, saving to a collection, live preview, ORCID/arXiv import; Premium alerts carried by the digest |
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

Search scope is optional (`field=all`, `author`, `title`, or `abstract`) and persists through result filters
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
Results load in provider pages of 25, with a signed continuation backed by cached prior batches.
Each click fetches only the next batch, reuses cached batches and deduplicates identities. Up to 20
batches (500 matches per provider) can be loaded per result window; expired windows restart with
a notice. Partial provider failures never offer a continuation that would skip the failed batch. Classification, journal assignment and ORCID coverage depend on the indexed metadata.

API references: [OpenAlex filters](https://help.openalex.org/api/filtering/),
[arXiv API](https://info.arxiv.org/help/api/user-manual.html),
[Crossref filters](https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-filters/).

Source abstracts appear directly as short excerpts, with the full abstract expandable. No AI calls
are made for previews. Active-filter chips remove only the chosen filter and reset result paging.

### Journal club editor and PDF layout

The slide language is chosen explicitly (French/English), independently from the interface language.
`deck_lang` and `deck_version` pin preview, save and PowerPoint URLs to the same content. Generation
uses `slides_v2` and requires a translated presentation title; every generated text field is requested
in that language. Built-in titles, figure credits, PowerPoint metadata and proofing language also follow
the deck language. Original image labels and author names remain unchanged. Legacy `slides` /
`slides_edit` records remain available through the previous-presentation link; new private copies use
`slides_edit_v2`, so upgrading never overwrites an earlier edited presentation.

Journal club keeps the generated deck shared and saves each Premium reader's text, notes and layout
in a separate `Generated(kind='slides_edit')` row keyed by user, paper and language. Private results
are owner-only and page/save responses are never cached. A revision hash rejects stale-tab saves
with HTTP 409; failed requests retain the unsaved editor state. Source figures and credits cannot be
replaced by client payloads. No schema migration is needed.

`slides.js` previews the exact 1280 × 720 canvas, scales it to the screen, allows four figure positions,
a 25–70% figure area and three text sizes, and shrinks overflowing text down to 20px before warning.
Notes remain outside the slide and never enter the PDF. The PDF button waits for fonts/images and
uses the browser's print-to-PDF dialog; `slides.css` specifies a zero-margin 338.666667 × 190.5mm page
(16:9), one slide per page, including when invoked during presentation mode. A print-dialog paper
size override may override the CSS page size. Check the final PDF in the target browser.

The PowerPoint button posts the current editor snapshot to the Premium-only, CSRF-protected
`journal-club/<paper>/powerpoint/` endpoint. It includes unsaved changes without modifying the saved
copy. The server validates the snapshot against the generated deck and revision; image URLs and
credits always come from the source. `python-pptx` writes a 16:9 file with native text boxes, image
objects, theme fonts and speaker notes. Text is measured before export; content that cannot fit
returns an actionable error rather than being truncated.

Source images are embedded as PNG (also converted from self-contained SVG with `resvg_py`). Downloads
are restricted to HTTPS arXiv HTML paths, including every redirect, bounded by size/time, and cached
six hours. SVG external/local resources are excluded; embedded PNG/JPEG panels are supported. Image
failure returns a retryable error rather than silently dropping a figure. No external AI service is used.

Validation: `python manage.py test summarizer` and `node --test tests/*.test.cjs`.

### Saved searches and email watches

Signed, validated search parameters can be saved by a logged-in user; records are private, unique
per user and parameter set, and limited to 50 per account. Reopening restores every filter, without
old continuation tokens. Users can rename or delete searches and explicitly enable/pause email
watches. Free accounts may run one watch, Premium five; after a downgrade only the earliest watches
within the current limit run, with their status shown in the management page.

Activation/resumption persists the opt-in immediately, without source requests in the web process.
The hourly worker retries initialization until every required source responds, then snapshots the latest
50 matches as a baseline and sends no historical email. The UI shows initialization pending until then.
Daily/weekly checks compare that bounded latest-match window with seen DOI/arXiv aliases. A message
contains up to ten unseen matches; coverage is explicitly bounded, not an exhaustive literature alert.
No summaries are generated by the watcher. Suppressed addresses and inactive users are skipped.
A database lease prevents overlapping workers from delivering the same check concurrently; failures
leave delivery state unchanged for retry. Successful mail advances the seen set and schedule.
A signed unsubscribe URL opens a confirmation; only POST pauses the watch, so email prefetching is safe.

Umami receives `saved_search_created` and `search_watch_enabled` once from server-confirmed
save/enable actions, and `search_more_click` for continuation clicks. No query, filter, source ID,
ORCID, saved-search ID or manuscript text is included in these event properties.

The existing hourly `send_digest` command runs search watches last, using only its remaining email
budget (and at most ten checks per run). `--dry-run` and `--user` never send search watches. There is no
new Heroku Scheduler job. The release migration creates `SavedSearch`; it activates no existing user.
Validation: `python manage.py test summarizer` and `node --test tests/funnel.test.cjs tests/home-search.test.cjs`.

## Daily digest for researchers

Each paper of a digest gets, in the same model call as its plain-language blurb, a line for researchers
(contribution, method, main result; `DigestSubscription.level`, researcher by default). Ranking
(`digest.rank_papers`) works the same for the email, its web page and the live preview of the settings:

- Keywords match whole words with or without a plural `s`; `"quoted words"` match as written; `-word` leaves
  papers out. Without a research profile they filter; with one they raise the papers that use them.
- A Premium profile is one embedding per paragraph and per reference paper (`profile_vector` holds the rows,
  `profile_labels` what they stand for); a paper is scored by the closest row, then by the papers liked,
  disliked or opened from the emails (`DigestFeedback.source`: `vote` or `click`). Free digests are ranked by
  their votes and clicks too. Without any of these, the first paper is the most typical of the batch, then each
  pick is the one least like those already picked.
- Premium digests may send every paper above a closeness threshold (`volume`: `high`, `medium`, `low` =
  cosine 0.55, 0.48, 0.42 with `text-embedding-3-small` at 512 dimensions; 50 papers at most per email).
  These thresholds are a first guess: check them with `digest_stats` and the scores shown on the web pages.

Every email is stored as a `DigestEdition` (announcement days, papers sent and their closeness). Its web page
(`/digest/e/<token>/`) lists the papers of the email in its order, then every other paper of those
announcements, ranked, with its closeness and why it is there, for two weeks (`PAGE_DAYS`); only its owner,
logged in, sees the profile excerpts and liked titles behind a reason. Email links are short signed codes
(`/digest/l/<code>/`, see `digest.link_code`: an email of 50 papers stays well under Gmail's 102 KB cut).
An opening link shows a page that goes on to the paper at once and whose script reports the click
(`DigestClick`, and a light `click` feedback): mail scanners run no script, and four links of one digest
reported within ten seconds are still set apart. `python manage.py digest_stats` prints the click-through
rate by rank, votes, stopped digests and, once there are clicks enough, the closeness of the papers opened
against the ones left, with the thresholds the data would suggest.

`/digest/save/<token>/` saves a paper to a collection from the email. The settings page imports categories,
reference papers and frequent co-authors from an ORCID iD or an arXiv author identifier, and shows what the
form's settings would have sent from the latest announcement (`/digest/try/`, 30 per user and 10 minutes; it
embeds up to 1,000 papers of categories nobody follows yet, kept for everyone). A bundled digest checks its
reader's scoop projects against the day's announcement before it is sent. Category names are translated. With `bundle_alerts`, a Premium digest carries the scoop, citation, version and
publication alerts as sections; each module then leaves that user's alerts for it (`digest_bundle`).
The weekly synthesis cites the papers behind each trend, and picks its notable papers from a shortlist ranked
by how many research profiles they come close to, rising topics and cross-listing, with a reason for each.

## Deployment

`git push heroku master`. The `release` phase of the Procfile applies migrations.

- Hourly job (Heroku Scheduler): `python manage.py send_digest`. It sends the digests that are due, then counts the
  day's terms for the rising topics, writes the weekly topics and letters, and sends scoop alerts, citation alerts, new version alerts and letters
  within what is left of the hourly email budget.
- Scoop watch thresholds: `python manage.py scoop_calibrate` scores sample projects (or `--text file`) against the last
  announcements and has the model read each closeness band; `--feedback` shows the readers' verdicts. It writes nothing.
- Stripe: webhook endpoint `https://www.summarizepaper.com/stripe/webhook/` with the events `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated` and `customer.subscription.deleted`.

## Contributors :octocat:

Developed by :sparkles: Quentin Kral :sparkles:. Questions or suggestions are welcome.

## License :page_with_curl:

MIT License, see the [LICENSE](LICENSE) file. :clipboard:
