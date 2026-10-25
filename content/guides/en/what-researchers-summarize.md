---
title: "What researchers summarize: arXiv categories on SummarizePaper (2023–2026 data)"
description: "Which arXiv papers people summarize with AI, from 9,995 papers summarized on SummarizePaper since 2023, compared with arXiv's own category shares."
date: 2026-10-18
dataset:
  name: "SummarizePaper usage by arXiv category, 2023–2026"
  description: "Aggregate counts of arXiv papers summarized, summary requests, assistant questions, PDF exports and summary votes on SummarizePaper, by arXiv primary category, from February 2023 to 18 October 2026."
  temporalCoverage: "2023-02-24/2026-10-18"
faq:
  - q: "Which arXiv papers do people summarize the most?"
    a: "Computer science papers make up 64.8% of the 9,995 arXiv papers summarized on SummarizePaper between 2023 and October 2026, against 44.2% of arXiv submissions over the same period. cs.CL (computation and language) alone accounts for 15.9%."
  - q: "Which arXiv categories are summarized more than their share of arXiv?"
    a: "On SummarizePaper, econ.GN (3.5 times its arXiv share), cs.CY, computers and society (3.4 times), cs.CL (2.8 times) and astro-ph.EP, planetary astrophysics (2.7 times) are the most over-represented categories."
  - q: "Are math papers summarized with AI?"
    a: "Rarely. Mathematics is 16.1% of arXiv submissions from March 2023 to September 2026, but only 2.9% of the papers summarized on SummarizePaper."
  - q: "Do people summarize new arXiv papers or older ones?"
    a: "Mostly older ones. The median paper is 387 days old when it is first summarized on SummarizePaper, and 51.0% are more than a year old. Only 10.7% are summarized within a week of publication."
  - q: "Are AI summaries of papers rated well by readers?"
    a: "On SummarizePaper, 75.3% of the 344 active reader votes on summaries are positive (October 2026)."
sources:
  - label: "SummarizePaper database, aggregate counts (queried 2026-10-18)"
    url: "https://www.summarizepaper.com/en/guides/what-researchers-summarize/"
  - label: "arXiv, Submissions by Category since 2022 (downloaded 2026-10-18)"
    url: "https://info.arxiv.org/about/reports/submission_category_by_year.html"
  - label: "SummarizePaper, arXiv statistics 2026"
    url: "https://www.summarizepaper.com/en/guides/arxiv-statistics/"
---

<div class="takeaways" markdown="1">

Key numbers

- **9,995** arXiv papers summarized on SummarizePaper between February 2023 and October 2026.
- **64.8%** of them are computer science papers, against **44.2%** of arXiv submissions over the same period.
- **cs.CL** (computation and language) is the most summarized category: **15.9%** of papers, **2.8 times** its share of arXiv.
- **Mathematics** is **16.1%** of arXiv but only **2.9%** of summarized papers.
- The median paper is **387 days old** when first summarized; **51.0%** are more than a year old.
- **75.3%** of reader votes on summaries are positive.

</div>

<p class="note">Data from SummarizePaper's own database, queried on 18 October 2026 as aggregate counts only, compared with arXiv's category statistics. Methodology at the end of the page.</p>

## What is in this dataset?

SummarizePaper publishes AI summaries of research papers; this page counts the arXiv papers, which were the only source during most of the period (journals and other preprint servers were added in October 2026). Summaries are created on request, so the papers summarized show what people want to understand, not what arXiv publishes. Between February 2023 and 18 October 2026, the site recorded:

| Measure | Count |
|---|---|
| arXiv papers with a published summary | 9,995 |
| Summaries published (English and French) | 10,037 |
| Summary requests | 10,963 |
| Questions asked to the AI assistant about a paper | 7,618 |
| Summaries exported as PDF | 1,431 |
| Reader votes on summary quality | 344 |

## Which arXiv categories do people summarize the most?

The ten most summarized primary categories, compared with each category's share of arXiv submissions from March 2023 to September 2026:

| Rank | Category | Topic | Papers summarized | Share on SummarizePaper | Share of arXiv | Ratio |
|---|---|---|---|---|---|---|
| 1 | cs.CL | Computation and language | 1,590 | 15.9% | 5.7% | 2.8× |
| 2 | cs.CV | Computer vision | 1,183 | 11.8% | 9.5% | 1.2× |
| 3 | cs.LG | Machine learning | 1,166 | 11.7% | 8.6% | 1.4× |
| 4 | cs.AI | Artificial intelligence | 525 | 5.3% | 3.1% | 1.7× |
| 5 | astro-ph.GA | Astrophysics of galaxies | 336 | 3.4% | 1.7% | 2.0× |
| 6 | cs.CY | Computers and society | 284 | 2.8% | 0.8% | 3.4× |
| 7 | cs.CR | Cryptography and security | 272 | 2.7% | 1.9% | 1.4× |
| 8 | cs.SE | Software engineering | 231 | 2.3% | 1.3% | 1.7× |
| 9 | astro-ph.EP | Earth and planetary astrophysics | 223 | 2.2% | 0.8% | 2.7× |
| 10 | quant-ph | Quantum physics | 217 | 2.2% | 4.2% | 0.5× |

A ratio above 1 means the category is summarized more often than its weight on arXiv would predict.

<figure class="viz">
<figcaption><strong>Share of papers summarized on SummarizePaper vs share of arXiv submissions</strong><span>Selected primary categories. SummarizePaper: 9,995 papers, Feb 2023–Oct 2026. arXiv: Mar 2023–Sep 2026.</span></figcaption>
<div class="viz-legend"><span><i style="background:var(--viz-1)"></i>SummarizePaper</span><span><i style="background:var(--viz-2)"></i>arXiv</span></div>
<div class="viz-scroll"><svg viewBox="0 0 760 368" role="img" aria-label="Dot chart comparing each category's share of papers summarized on SummarizePaper with its share of arXiv submissions. cs.CL is 15.9% on SummarizePaper against 5.7% on arXiv; mathematics is 2.9% against 16.1%.">
<line class="grid" x1="270.0" x2="270.0" y1="24" y2="338"/>
<text class="tick" x="270.0" y="356" text-anchor="middle">0%</text>
<line class="grid" x1="343.3" x2="343.3" y1="24" y2="338"/>
<text class="tick" x="343.3" y="356" text-anchor="middle">3%</text>
<line class="grid" x1="416.7" x2="416.7" y1="24" y2="338"/>
<text class="tick" x="416.7" y="356" text-anchor="middle">6%</text>
<line class="grid" x1="490.0" x2="490.0" y1="24" y2="338"/>
<text class="tick" x="490.0" y="356" text-anchor="middle">9%</text>
<line class="grid" x1="563.3" x2="563.3" y1="24" y2="338"/>
<text class="tick" x="563.3" y="356" text-anchor="middle">12%</text>
<line class="grid" x1="636.7" x2="636.7" y1="24" y2="338"/>
<text class="tick" x="636.7" y="356" text-anchor="middle">15%</text>
<line class="grid" x1="710.0" x2="710.0" y1="24" y2="338"/>
<text class="tick" x="710.0" y="356" text-anchor="middle">18%</text>
<text class="row" x="256" y="49.0" dy="4" text-anchor="end"><tspan class="cat">cs.CL</tspan> <tspan class="topic">Computation and language</tspan></text>
<g class="hit" data-tip="cs.CL: 15.9% of papers summarized vs 5.7% of arXiv"><rect x="0" y="34.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="658.7" x2="409.3" y1="49.0" y2="49.0"/>
<circle class="dot" cx="409.3" cy="49.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="658.7" cy="49.0" r="6" fill="var(--viz-1)"/></g>
<text class="val" x="670.7" y="49.0" dy="4">15.9%</text>
<text class="val" x="397.3" y="49.0" dy="4" text-anchor="end">5.7%</text>
<text class="row" x="256" y="79.0" dy="4" text-anchor="end"><tspan class="cat">cs.CV</tspan> <tspan class="topic">Computer vision</tspan></text>
<g class="hit" data-tip="cs.CV: 11.8% of papers summarized vs 9.5% of arXiv"><rect x="0" y="64.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="558.4" x2="502.2" y1="79.0" y2="79.0"/>
<circle class="dot" cx="502.2" cy="79.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="558.4" cy="79.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="109.0" dy="4" text-anchor="end"><tspan class="cat">cs.LG</tspan> <tspan class="topic">Machine learning</tspan></text>
<g class="hit" data-tip="cs.LG: 11.7% of papers summarized vs 8.6% of arXiv"><rect x="0" y="94.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="556.0" x2="480.2" y1="109.0" y2="109.0"/>
<circle class="dot" cx="480.2" cy="109.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="556.0" cy="109.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="139.0" dy="4" text-anchor="end"><tspan class="cat">cs.AI</tspan> <tspan class="topic">Artificial intelligence</tspan></text>
<g class="hit" data-tip="cs.AI: 5.3% of papers summarized vs 3.1% of arXiv"><rect x="0" y="124.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="399.6" x2="345.8" y1="139.0" y2="139.0"/>
<circle class="dot" cx="345.8" cy="139.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="399.6" cy="139.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="169.0" dy="4" text-anchor="end"><tspan class="cat">astro-ph.GA</tspan> <tspan class="topic">Galaxies</tspan></text>
<g class="hit" data-tip="astro-ph.GA: 3.4% of papers summarized vs 1.7% of arXiv"><rect x="0" y="154.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="353.1" x2="311.6" y1="169.0" y2="169.0"/>
<circle class="dot" cx="311.6" cy="169.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="353.1" cy="169.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="199.0" dy="4" text-anchor="end"><tspan class="cat">cs.CY</tspan> <tspan class="topic">Computers and society</tspan></text>
<g class="hit" data-tip="cs.CY: 2.8% of papers summarized vs 0.8% of arXiv"><rect x="0" y="184.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="338.4" x2="289.6" y1="199.0" y2="199.0"/>
<circle class="dot" cx="289.6" cy="199.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="338.4" cy="199.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="229.0" dy="4" text-anchor="end"><tspan class="cat">astro-ph.EP</tspan> <tspan class="topic">Planetary astrophysics</tspan></text>
<g class="hit" data-tip="astro-ph.EP: 2.2% of papers summarized vs 0.8% of arXiv"><rect x="0" y="214.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="323.8" x2="289.6" y1="229.0" y2="229.0"/>
<circle class="dot" cx="289.6" cy="229.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="323.8" cy="229.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="259.0" dy="4" text-anchor="end"><tspan class="cat">econ.GN</tspan> <tspan class="topic">General economics</tspan></text>
<g class="hit" data-tip="econ.GN: 1.1% of papers summarized vs 0.3% of arXiv"><rect x="0" y="244.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="296.9" x2="277.3" y1="259.0" y2="259.0"/>
<circle class="dot" cx="277.3" cy="259.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="296.9" cy="259.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="289.0" dy="4" text-anchor="end"><tspan class="cat">quant-ph</tspan> <tspan class="topic">Quantum physics</tspan></text>
<g class="hit" data-tip="quant-ph: 2.2% of papers summarized vs 4.2% of arXiv"><rect x="0" y="274.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="323.8" x2="372.7" y1="289.0" y2="289.0"/>
<circle class="dot" cx="372.7" cy="289.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="323.8" cy="289.0" r="6" fill="var(--viz-1)"/></g>
<text class="row" x="256" y="319.0" dy="4" text-anchor="end"><tspan class="cat">math</tspan> <tspan class="topic">All mathematics</tspan></text>
<g class="hit" data-tip="math: 2.9% of papers summarized vs 16.1% of arXiv"><rect x="0" y="304.0" width="760" height="30" fill="transparent"/>
<line class="link" x1="340.9" x2="663.6" y1="319.0" y2="319.0"/>
<circle class="dot" cx="663.6" cy="319.0" r="6" fill="var(--viz-2)"/>
<circle class="dot" cx="340.9" cy="319.0" r="6" fill="var(--viz-1)"/></g>
<text class="val" x="675.6" y="319.0" dy="4">16.1%</text>
<text class="val" x="328.9" y="319.0" dy="4" text-anchor="end">2.9%</text>
</svg></div>
</figure>


## Which fields are over-represented?

- **Language models.** cs.CL, the category of most large language model papers, is summarized 2.8 times more than its share of arXiv. Eight of the ten most requested individual papers are in cs.CL or cs.LG (see below).
- **Papers with a public beyond specialists.** econ.GN (general economics, 3.5×), cs.CY (computers and society, 3.4×) and cs.IR (information retrieval, 2.0×) are summarized far more than their size on arXiv.
- **Astrophysics.** astro-ph is 10.5% of summarized papers against 6.5% of arXiv, led by astro-ph.EP (2.7×) and astro-ph.GA (2.0×). SummarizePaper was created by an astrophysicist, which likely brings readers from that community.

## Which fields are rarely summarized?

| Field | Share of arXiv (Mar 2023–Sep 2026) | Share of papers summarized |
|---|---|---|
| Mathematics (math) | 16.1% | 2.9% |
| Condensed matter (cond-mat) | 6.7% | 2.6% |
| High-energy physics (hep) | 4.2% | 1.5% |
| Quantum physics (quant-ph) | 4.2% | 2.2% |
| Robotics (cs.RO) | 2.9% | 2.0% |

Mathematics is the clearest case: one arXiv submission in six, but fewer than one summarized paper in thirty. One likely reason: a proof is hard to compress into a summary without losing what matters.

## Do people summarize new papers or old ones?

Mostly established papers. Measuring the time between a paper's arXiv publication date and its first summary on SummarizePaper, for 9,995 papers:

| Age of the paper when first summarized | Share of papers |
|---|---|
| One week or less | 10.7% |
| One month or less | 19.4% |
| More than one year | 51.0% |

The median paper is **387 days old** when someone first asks for its summary. This suggests readers come to understand papers that matter to them, often well after the first wave of attention, more than to triage the day's new listings.

## Which papers are summarized the most?

The ten papers with the most summary requests (a paper can be summarized again, for example in another language):

| Paper | arXiv ID | Category | Requests |
|---|---|---|---|
| ReAct: Synergizing Reasoning and Acting in Language Models | 2210.03629 | cs.CL | 19 |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | 2305.10601 | cs.CL | 19 |
| Textbooks Are All You Need | 2306.11644 | cs.CL | 12 |
| Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models | 2101.00288 | cs.CL | 8 |
| Challenges and Applications of Large Language Models | 2307.10169 | cs.CL | 7 |
| Science opportunities with solar sailing smallsats | 2303.14917 | astro-ph.EP | 7 |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2501.12948 | cs.CL | 6 |
| Learning Analytics in Massive Open Online Courses | 1802.09344 | cs.CY | 6 |
| Absolute Zero: Reinforced Self-play Reasoning with Zero Data | 2505.03335 | cs.LG | 6 |
| PRAGMA: Revolut Foundation Model | 2604.08649 | cs.LG | 5 |

## What do readers ask about, and what do they keep?

- **Questions to the AI assistant follow the same pattern.** cs.CL papers received 1,146 of the 7,618 questions (15.0%); cs.CL, cs.CV, cs.LG and cs.AI together received 39.2%.
- **Exports concentrate even more on language models.** cs.CL papers account for 320 of the 1,431 summaries exported as PDF (22.4%).
- **Readers mostly approve of the summaries.** 75.3% of the 344 active votes on summary quality are positive.

## Methodology and limits

- **Source.** SummarizePaper's production database, queried on 18 October 2026 with read-only queries returning aggregate counts only. No user, IP address or question text was read.
- **Period.** From the first summary request (24 February 2023) to 18 October 2026.
- **Categories.** Each paper is counted once, under its arXiv primary category as stored when it was summarized. "Share of arXiv" uses arXiv's Submissions by Category data from March 2023 to September 2026 (975,113 submissions), described in our [arXiv statistics 2026](https://www.summarizepaper.com/en/guides/arxiv-statistics/).
- **Who asks.** 98.8% of summary requests came from visitors; 1.2% came from the site's administrator accounts.
- **Paper age.** Days between the arXiv publication date (first version) and the first summary; 9,995 papers with both dates.
- **Limits.** This reflects SummarizePaper's audience, not all researchers: an English and French site created by an astrophysicist. Requests are counted, not unique people. Half the papers summarized are more than a year old, while the arXiv shares used for comparison cover submissions from March 2023 onwards; category shares on arXiv change slowly, but the comparison is indicative, not exact.

<div class="cta-box">
<p>Want a summary of a paper in your field?</p>
<a class="btn btn-lg" href="https://www.summarizepaper.com/en/">Look up an arXiv paper</a>
<a class="btn btn-lg btn-ghost" href="https://www.summarizepaper.com/en/digest/">Get the daily digest</a>
</div>
