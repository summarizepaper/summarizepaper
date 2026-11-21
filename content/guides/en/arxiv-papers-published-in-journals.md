---
title: How many arXiv preprints are published in journals? Data by field (2022–2025)
description: Share of arXiv papers with a journal reference or DOI by field and year, and the journals they appear in, from every paper submitted since 2022.
date: 2026-11-21
updated: 2026-11-21
faq:
- q: What percentage of arXiv papers get published?
  a: At least 40% of the papers submitted to arXiv in 2022 carry a journal reference or a DOI on arXiv. The true share is higher, since authors do not always add them.
- q: Which fields publish most of their arXiv preprints?
  a: Astrophysics (astro-ph), with 87% of 2022 papers showing a journal version; Economics (econ) has the lowest share (18%). A low share also reflects authors who never add the reference, and conference papers.
- q: Are arXiv papers peer reviewed?
  a: 'No: arXiv only moderates submissions. Peer review happens at the journal or conference; a journal reference on arXiv shows that a version was published there.'
dataset:
  name: Journal references of arXiv papers by field, 2022–2025
  description: Whether each arXiv paper submitted from 2022 to 2025 carries a journal reference or DOI, by primary category.
  temporalCoverage: 2022-01-01/2025-12-31
sources:
- label: arXiv OAI-PMH interface
  url: https://info.arxiv.org/help/oa/index.html
- label: arXiv metadata format (arXiv OAI schema)
  url: https://oaipmh.arxiv.org/OAI/arXiv.xsd
---

<div class="takeaways" markdown="1">

Key numbers

- **40%** of the arXiv papers submitted in 2022 now show a journal reference or a DOI on arXiv (75,197 of 185,692).
- Highest: **Astrophysics (astro-ph)**, 87%.
- Lowest: **Economics (econ)**, 18%.
- For papers submitted in 2025, the share is still 21%: publication takes time.

</div>

<p class="note">Counted by SummarizePaper from the metadata of every arXiv paper submitted from 2022 to 2025 (arXiv OAI-PMH, harvested on 21 November 2026). A journal reference is added by the authors, or by arXiv from some publishers, so these shares are a lower bound.</p>

## What share of arXiv preprints are published in journals?

At least 40% of the papers submitted to arXiv in 2022 point to a published version: their arXiv record carries a journal reference or a DOI. The share depends on how long papers have had to be published:

| Submitted in | Papers | With a journal reference or DOI | Share |
|---|---|---|---|
| 2022 | 185,692 | 75,197 | 40% |
| 2023 | 208,493 | 64,196 | 31% |
| 2024 | 244,031 | 62,923 | 26% |
| 2025 | 284,486 | 60,162 | 21% |

## Which fields publish their preprints in journals most?

| Field | Submitted 2022 | Submitted 2023 | Submitted 2024 | Submitted 2025 |
|---|---|---|---|---|
| Astrophysics (astro-ph) | 87% | 58% | 45% | 46% |
| General relativity and quantum cosmology (gr-qc) | 81% | 64% | 58% | 52% |
| Nuclear physics (nucl-ex, nucl-th) | 77% | 55% | 48% | 44% |
| Condensed matter (cond-mat) | 74% | 61% | 55% | 47% |
| High-energy physics, theory (hep-th) | 74% | 46% | 37% | 33% |
| High-energy physics, phenomenology (hep-ph) | 72% | 49% | 40% | 38% |
| Quantum physics (quant-ph) | 69% | 59% | 52% | 39% |
| Nonlinear sciences (nlin) | 66% | 44% | 35% | 29% |
| High-energy physics, experiment (hep-ex) | 60% | 57% | 58% | 51% |
| Other physics (physics.*) | 57% | 41% | 35% | 31% |
| Mathematical physics (math-ph) | 55% | 43% | 34% | 23% |
| Quantitative biology (q-bio) | 29% | 23% | 21% | 16% |
| Electrical engineering and systems science (eess) | 25% | 21% | 20% | 16% |
| Mathematics (math) | 24% | 20% | 17% | 10% |
| Statistics (stat) | 20% | 18% | 16% | 11% |
| Computer science (cs) | 20% | 17% | 16% | 13% |
| Quantitative finance (q-fin) | 20% | 18% | 15% | 12% |
| Economics (econ) | 18% | 14% | 13% | 10% |

Physics and astronomy papers are usually submitted to arXiv when they go to a journal, and authors update the record once published. In computer science most papers go to conferences, whose proceedings are often not entered as a journal reference, so the low share there does not mean the papers stay unpublished.

## Where are arXiv preprints published?

The most frequent journals of papers submitted in 2022, read from their journal reference or DOI, by field:

| Field | Most frequent journals (papers) |
|---|---|
| Computer science (cs) | EPTCS (210), IEEE Robotics and Automation Letters (200), NeurIPS (134), IEEE Access (133), ICRA (117) |
| Mathematics (math) | J. Comput. Phys. (132), Algebr. Geom. Topol. (97), Advances in Mathematics (73), SIGMA (68), Alg. Number Th. (56) |
| Condensed matter (cond-mat) | Phys. Rev. B (3,099), Phys. Rev. Lett. (700), Phys. Rev. E (592), Nature Commun. (333), Phys. Rev. Research (321) |
| Astrophysics (astro-ph) | MNRAS (3,513), Astrophys. J. (2,477), Astron. Astrophys. (1,960), Astrophys. J. Lett. (581), Phys. Rev. D (521) |
| Other physics (physics.*) | Phys. Rev. A (317), JINST (238), J. Fluid Mech. (233), Opt. Express (206), Phys. Rev. E (205) |
| Electrical engineering and systems science (eess) | ICASSP (64), IEEE Trans. Signal Process. (55), Interspeech (51), IEEE Access (38), IEEE Trans. Geosci. Remote Sens. (21) |
| Quantum physics (quant-ph) | Phys. Rev. A (986), Phys. Rev. Lett. (366), Quantum (355), Phys. Rev. Research (231), Phys. Rev. B (171) |
| High-energy physics, phenomenology (hep-ph) | Phys. Rev. D (1,232), JHEP (693), Eur. Phys. J. C (311), Phys. Lett. B (231), JCAP (134) |
| Statistics (stat) | Journal of Machine Learning Research (26), NeurIPS (23), ICML (23), Statistics in Medicine (14), IEEE Trans. Signal Process. (13) |
| High-energy physics, theory (hep-th) | JHEP (1,207), Phys. Rev. D (676), Eur. Phys. J. C (148), Phys. Lett. B (132), SciPost Phys. (108) |
| General relativity and quantum cosmology (gr-qc) | Phys. Rev. D (898), Eur. Phys. J. C (248), Class. Quantum Grav. (199), JCAP (142), Universe (105) |
| Quantitative biology (q-bio) | Phys. Rev. E (35), Sci. Rep. (16), Chaos (12), PLOS ONE (12), Phys. Rev. Research (8) |
| Nuclear physics (nucl-ex, nucl-th) | Phys. Rev. C (492), Phys. Lett. B (104), Phys. Rev. D (91), Phys. Rev. Lett. (65), Eur. Phys. J. A (55) |
| Economics (econ) | Journal of Econometrics (5), Econom. Theory (5) |
| Mathematical physics (math-ph) | J. Phys. A (38), J. Math. Phys. (29), J. Phys. A: Math. Theor. (27), Commun. Math. Phys. (15), Ann. Henri Poincaré (12) |
| High-energy physics, experiment (hep-ex) | Phys. Rev. D (131), JHEP (99), Phys. Rev. Lett. (83), Eur. Phys. J. C (63), Phys. Lett. B (36) |
| Quantitative finance (q-fin) | Entropy (9) |
| Nonlinear sciences (nlin) | Phys. Rev. E (68), Physica D (29), Chaos (24), J. Phys. A (19), Phys. Rev. Lett. (11) |

## How to know if a preprint was published

- On arXiv, look for the "Journal reference" and "DOI" lines under the abstract; they are often missing even when the paper is published.
- On SummarizePaper, a paper page shows its journal version when one is found, and Premium readers can be [alerted when a followed preprint is published](https://www.summarizepaper.com/en/guides/track-research-paper-citations/).
- For a whole reading list, Premium watches a [collection](https://www.summarizepaper.com/en/guides/organize-research-paper-collections/) and emails you when one of its papers is published, retracted or corrected.

## Methodology and limits

- **Papers**: every arXiv paper whose identifier starts with the year (2201–2212 for 2022, and so on), from the arXiv metadata of arXiv's OAI-PMH interface, harvested on 21 November 2026.
- **Published**: the arXiv record has a journal reference or a DOI. Both are entered by the authors or added by arXiv for some publishers; many published papers have neither, so every share here is a lower bound, lowest in fields that publish in conference proceedings.
- **Field**: the archive of the primary category; physics.* is "physics (other)", nucl-ex and nucl-th are nuclear physics. Fields with fewer than 300 papers in 2022 are left out.
- **Journals**: names read from the journal reference, normalized (abbreviations and spacing), or else from the DOI prefix of the main publishers; a paper whose journal cannot be recognized is not counted in that table.

<div class="cta-box">
<p>Follow your field without reading every listing</p>
<a class="btn btn-lg" href="https://www.summarizepaper.com/en/digest/">Get the daily paper alert</a>
<a class="btn btn-lg btn-ghost" href="https://www.summarizepaper.com/en/">Summarize a paper</a>
</div>
