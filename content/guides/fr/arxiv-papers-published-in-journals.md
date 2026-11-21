---
title: Combien de preprints arXiv sont publiés en revue ? Les données par domaine (2022–2025)
description: Part des articles arXiv avec référence de revue ou DOI par domaine et par année, et revues où ils paraissent, sur tous les articles depuis 2022.
date: 2026-11-21
updated: 2026-11-21
faq:
- q: Quel pourcentage des articles arXiv est publié ?
  a: Au moins 40 % des articles soumis à arXiv en 2022 portent une référence de revue ou un DOI sur arXiv. La part réelle est plus élevée, car les auteurs ne les ajoutent pas toujours.
- q: Quels domaines publient le plus leurs preprints arXiv ?
  a: Astrophysique (astro-ph), avec 87 % des articles de 2022 affichant une version publiée ; Économie (econ) a la part la plus faible (18 %). Une part faible reflète aussi les auteurs qui n’ajoutent jamais la référence, et les articles de conférence.
- q: Combien de preprints d’informatique sur arXiv sont publiés ?
  a: Des articles d’informatique soumis à arXiv en 2022, 20 % affichent une référence de revue ou un DOI, mais 40 % si l’on compte aussi ceux dont les commentaires arXiv indiquent une acceptation, surtout en conférence (NeurIPS, CVPR, AAAI...).
- q: Les articles arXiv sont-ils évalués par les pairs ?
  a: 'Non : arXiv se contente de modérer les soumissions. L’évaluation par les pairs a lieu dans la revue ou la conférence ; une référence de revue sur arXiv montre qu’une version y a été publiée.'
dataset:
  name: Références de revue des articles arXiv par domaine, 2022–2025
  description: Présence d’une référence de revue ou d’un DOI pour chaque article arXiv soumis de 2022 à 2025, par catégorie principale.
  temporalCoverage: 2022-01-01/2025-12-31
sources:
- label: arXiv OAI-PMH interface
  url: https://info.arxiv.org/help/oa/index.html
- label: arXiv metadata format (arXiv OAI schema)
  url: https://oaipmh.arxiv.org/OAI/arXiv.xsd
---

<div class="takeaways" markdown="1">

Chiffres clés

- **40 %** des articles arXiv soumis en 2022 affichent aujourd’hui une référence de revue ou un DOI sur arXiv (75 197 sur 185 692).
- Le plus haut : **Astrophysique (astro-ph)**, 87 %.
- Le plus bas : **Économie (econ)**, 18 %.
- Pour les articles soumis en 2025, la part n’est encore que de 21 % : la publication prend du temps.
- En comptant les articles dont les commentaires arXiv indiquent une acceptation (surtout en conférence), l’informatique passe de 20 % à **40 %**.

</div>

<p class="note">Calculé par SummarizePaper à partir des métadonnées de chaque article arXiv soumis de 2022 à 2025 (OAI-PMH d’arXiv, récupérées le 21 novembre 2026). La référence de revue est ajoutée par les auteurs, ou par arXiv pour certains éditeurs : ces parts sont donc un minimum.</p>

## Quelle part des preprints arXiv est publiée en revue ?

Au moins 40 % des articles soumis à arXiv en 2022 renvoient à une version publiée : leur notice arXiv porte une référence de revue ou un DOI. La part dépend du temps que les articles ont eu pour être publiés :

| Soumis en | Articles | Avec référence de revue ou DOI | Part |
|---|---|---|---|
| 2022 | 185 692 | 75 197 | 40 % |
| 2023 | 208 493 | 64 196 | 31 % |
| 2024 | 244 031 | 62 923 | 26 % |
| 2025 | 284 486 | 60 162 | 21 % |

## Quels domaines publient le plus leurs preprints en revue ?

<figure class="viz">
<figcaption><strong>Part des articles arXiv de 2022 avec référence de revue ou DOI, par domaine</strong><span>Catégorie principale de chaque article. Un minimum : beaucoup d’articles publiés n’ont ni l’un ni l’autre. Source : métadonnées OAI-PMH d’arXiv.</span></figcaption>
<div class="viz-scroll"><svg viewBox="0 0 760 556" role="img" aria-label="Graphique en barres de la part des articles arXiv soumis en 2022 qui affichent une référence de revue ou un DOI, par domaine, de 87 % en Astrophysique à 18 % en Économie.">
<text class="row" x="140" y="17.0" dy="4" text-anchor="end"><tspan class="cat">astro-ph</tspan></text>
<g class="hit" data-tip="Astrophysique: 87 %"><rect x="0" y="2" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="8" width="468.7" height="18" rx="3"/></g>
<text class="val" x="624.7" y="17.0" dy="4">87 %</text>
<text class="row" x="140" y="47.0" dy="4" text-anchor="end"><tspan class="cat">gr-qc</tspan></text>
<g class="hit" data-tip="Relativité générale et cosmologie quantique: 81 %"><rect x="0" y="32" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="38" width="435.1" height="18" rx="3"/></g>
<text class="val" x="591.1" y="47.0" dy="4">81 %</text>
<text class="row" x="140" y="77.0" dy="4" text-anchor="end"><tspan class="cat">nucl</tspan></text>
<g class="hit" data-tip="Physique nucléaire: 77 %"><rect x="0" y="62" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="68" width="415.5" height="18" rx="3"/></g>
<text class="val" x="571.5" y="77.0" dy="4">77 %</text>
<text class="row" x="140" y="107.0" dy="4" text-anchor="end"><tspan class="cat">cond-mat</tspan></text>
<g class="hit" data-tip="Matière condensée: 74 %"><rect x="0" y="92" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="98" width="399.0" height="18" rx="3"/></g>
<text class="val" x="555.0" y="107.0" dy="4">74 %</text>
<text class="row" x="140" y="137.0" dy="4" text-anchor="end"><tspan class="cat">hep-th</tspan></text>
<g class="hit" data-tip="Physique des hautes énergies, théorie: 74 %"><rect x="0" y="122" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="128" width="398.2" height="18" rx="3"/></g>
<text class="val" x="554.2" y="137.0" dy="4">74 %</text>
<text class="row" x="140" y="167.0" dy="4" text-anchor="end"><tspan class="cat">hep-ph</tspan></text>
<g class="hit" data-tip="Physique des hautes énergies, phénoménologie: 72 %"><rect x="0" y="152" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="158" width="390.3" height="18" rx="3"/></g>
<text class="val" x="546.3" y="167.0" dy="4">72 %</text>
<text class="row" x="140" y="197.0" dy="4" text-anchor="end"><tspan class="cat">quant-ph</tspan></text>
<g class="hit" data-tip="Physique quantique: 69 %"><rect x="0" y="182" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="188" width="371.8" height="18" rx="3"/></g>
<text class="val" x="527.8" y="197.0" dy="4">69 %</text>
<text class="row" x="140" y="227.0" dy="4" text-anchor="end"><tspan class="cat">nlin</tspan></text>
<g class="hit" data-tip="Sciences non linéaires: 66 %"><rect x="0" y="212" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="218" width="358.3" height="18" rx="3"/></g>
<text class="val" x="514.3" y="227.0" dy="4">66 %</text>
<text class="row" x="140" y="257.0" dy="4" text-anchor="end"><tspan class="cat">hep-ex</tspan></text>
<g class="hit" data-tip="Physique des hautes énergies, expérience: 60 %"><rect x="0" y="242" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="248" width="321.6" height="18" rx="3"/></g>
<text class="val" x="477.6" y="257.0" dy="4">60 %</text>
<text class="row" x="140" y="287.0" dy="4" text-anchor="end"><tspan class="cat">physics.*</tspan></text>
<g class="hit" data-tip="Autres domaines de la physique: 57 %"><rect x="0" y="272" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="278" width="310.2" height="18" rx="3"/></g>
<text class="val" x="466.2" y="287.0" dy="4">57 %</text>
<text class="row" x="140" y="317.0" dy="4" text-anchor="end"><tspan class="cat">math-ph</tspan></text>
<g class="hit" data-tip="Physique mathématique: 55 %"><rect x="0" y="302" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="308" width="296.0" height="18" rx="3"/></g>
<text class="val" x="452.0" y="317.0" dy="4">55 %</text>
<text class="row" x="140" y="347.0" dy="4" text-anchor="end"><tspan class="cat">q-bio</tspan></text>
<g class="hit" data-tip="Biologie quantitative: 29 %"><rect x="0" y="332" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="338" width="158.4" height="18" rx="3"/></g>
<text class="val" x="314.4" y="347.0" dy="4">29 %</text>
<text class="row" x="140" y="377.0" dy="4" text-anchor="end"><tspan class="cat">eess</tspan></text>
<g class="hit" data-tip="Génie électrique et science des systèmes: 25 %"><rect x="0" y="362" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="368" width="133.4" height="18" rx="3"/></g>
<text class="val" x="289.4" y="377.0" dy="4">25 %</text>
<text class="row" x="140" y="407.0" dy="4" text-anchor="end"><tspan class="cat">math</tspan></text>
<g class="hit" data-tip="Mathématiques: 24 %"><rect x="0" y="392" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="398" width="131.0" height="18" rx="3"/></g>
<text class="val" x="287.0" y="407.0" dy="4">24 %</text>
<text class="row" x="140" y="437.0" dy="4" text-anchor="end"><tspan class="cat">stat</tspan></text>
<g class="hit" data-tip="Statistiques: 20 %"><rect x="0" y="422" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="428" width="108.6" height="18" rx="3"/></g>
<text class="val" x="264.6" y="437.0" dy="4">20 %</text>
<text class="row" x="140" y="467.0" dy="4" text-anchor="end"><tspan class="cat">cs</tspan></text>
<g class="hit" data-tip="Informatique: 20 %"><rect x="0" y="452" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="458" width="107.2" height="18" rx="3"/></g>
<text class="val" x="263.2" y="467.0" dy="4">20 %</text>
<text class="row" x="140" y="497.0" dy="4" text-anchor="end"><tspan class="cat">q-fin</tspan></text>
<g class="hit" data-tip="Finance quantitative: 20 %"><rect x="0" y="482" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="488" width="106.0" height="18" rx="3"/></g>
<text class="val" x="262.0" y="497.0" dy="4">20 %</text>
<text class="row" x="140" y="527.0" dy="4" text-anchor="end"><tspan class="cat">econ</tspan></text>
<g class="hit" data-tip="Économie: 18 %"><rect x="0" y="512" width="760" height="30" fill="transparent"/><rect class="bar" x="150" y="518" width="96.6" height="18" rx="3"/></g>
<text class="val" x="252.6" y="527.0" dy="4">18 %</text>
</svg></div>
</figure>

| Domaine | Soumis en 2022 | Soumis en 2023 | Soumis en 2024 | Soumis en 2025 |
|---|---|---|---|---|
| Astrophysique (astro-ph) | 87 % | 58 % | 45 % | 46 % |
| Relativité générale et cosmologie quantique (gr-qc) | 81 % | 64 % | 58 % | 52 % |
| Physique nucléaire (nucl-ex, nucl-th) | 77 % | 55 % | 48 % | 44 % |
| Matière condensée (cond-mat) | 74 % | 61 % | 55 % | 47 % |
| Physique des hautes énergies, théorie (hep-th) | 74 % | 46 % | 37 % | 33 % |
| Physique des hautes énergies, phénoménologie (hep-ph) | 72 % | 49 % | 40 % | 38 % |
| Physique quantique (quant-ph) | 69 % | 59 % | 52 % | 39 % |
| Sciences non linéaires (nlin) | 66 % | 44 % | 35 % | 29 % |
| Physique des hautes énergies, expérience (hep-ex) | 60 % | 57 % | 58 % | 51 % |
| Autres domaines de la physique (physics.*) | 57 % | 41 % | 35 % | 31 % |
| Physique mathématique (math-ph) | 55 % | 43 % | 34 % | 23 % |
| Biologie quantitative (q-bio) | 29 % | 23 % | 21 % | 16 % |
| Génie électrique et science des systèmes (eess) | 25 % | 21 % | 20 % | 16 % |
| Mathématiques (math) | 24 % | 20 % | 17 % | 10 % |
| Statistiques (stat) | 20 % | 18 % | 16 % | 11 % |
| Informatique (cs) | 20 % | 17 % | 16 % | 13 % |
| Finance quantitative (q-fin) | 20 % | 18 % | 15 % | 12 % |
| Économie (econ) | 18 % | 14 % | 13 % | 10 % |

En physique et en astronomie, les articles sont en général déposés sur arXiv au moment de la soumission à une revue, et les auteurs complètent la notice une fois l’article publié. En informatique, la plupart des articles vont en conférence, dont les actes sont rarement indiqués comme référence de revue : la part faible n’y signifie pas que les articles restent inédits.

## Et les articles de conférence ?

Beaucoup d’auteurs indiquent dans les commentaires arXiv où l’article a été accepté (« Accepted at NeurIPS 2022 », « To appear in ... »), sans ajouter de référence de revue. En comptant aussi ces articles, pour ceux soumis en 2022 :

| Domaine | Référence de revue ou DOI | Y compris acceptés selon les commentaires |
|---|---|---|
| Informatique (cs) | 20 % | 40 % |
| Génie électrique et science des systèmes (eess) | 25 % | 39 % |
| Statistiques (stat) | 20 % | 26 % |
| Mathématiques (math) | 24 % | 30 % |
| Astrophysique (astro-ph) | 87 % | 91 % |
| Physique des hautes énergies, expérience (hep-ex) | 60 % | 63 % |
| Biologie quantitative (q-bio) | 29 % | 33 % |
| Physique des hautes énergies, phénoménologie (hep-ph) | 72 % | 76 % |
| Finance quantitative (q-fin) | 20 % | 23 % |
| Relativité générale et cosmologie quantique (gr-qc) | 81 % | 83 % |
| Physique nucléaire (nucl-ex, nucl-th) | 77 % | 80 % |
| Physique des hautes énergies, théorie (hep-th) | 74 % | 76 % |
| Sciences non linéaires (nlin) | 66 % | 68 % |
| Économie (econ) | 18 % | 20 % |
| Physique mathématique (math-ph) | 55 % | 57 % |
| Autres domaines de la physique (physics.*) | 57 % | 59 % |
| Physique quantique (quant-ph) | 69 % | 70 % |
| Matière condensée (cond-mat) | 74 % | 75 % |

Les commentaires sont un texte libre écrit par les auteurs ; l’acceptation est lue dans des mots comme accepted, to appear ou camera-ready, et les commentaires qui disent « submitted to » ou « under review » ne sont pas comptés.

## Où les preprints arXiv sont-ils publiés ?

Les revues et conférences les plus fréquentes des articles soumis en 2022, lues dans leur référence de revue, leur DOI ou leurs commentaires arXiv, par domaine :

| Domaine | Revues et conférences les plus fréquentes (articles) |
|---|---|
| Informatique (cs) | NeurIPS (1 060), CVPR (1 006), AAAI (702), ECCV (633), EMNLP (586) |
| Mathématiques (math) | J. Comput. Phys. (132), Algebr. Geom. Topol. (97), Advances in Mathematics (73), SIGMA (68), Alg. Number Th. (56) |
| Matière condensée (cond-mat) | Phys. Rev. B (3 099), Phys. Rev. Lett. (700), Phys. Rev. E (592), Nature Commun. (333), Phys. Rev. Research (321) |
| Astrophysique (astro-ph) | MNRAS (3 513), Astrophys. J. (2 477), Astron. Astrophys. (1 960), Astrophys. J. Lett. (581), Phys. Rev. D (521) |
| Autres domaines de la physique (physics.*) | Phys. Rev. A (317), JINST (238), J. Fluid Mech. (233), Opt. Express (206), Phys. Rev. E (205) |
| Génie électrique et science des systèmes (eess) | Interspeech (199), ICASSP (184), MICCAI (112), IEEE Trans. Signal Process. (55), IEEE Access (38) |
| Physique quantique (quant-ph) | Phys. Rev. A (986), Phys. Rev. Lett. (366), Quantum (355), Phys. Rev. Research (231), Phys. Rev. B (171) |
| Physique des hautes énergies, phénoménologie (hep-ph) | Phys. Rev. D (1 232), JHEP (693), Eur. Phys. J. C (311), Phys. Lett. B (231), JCAP (134) |
| Statistiques (stat) | NeurIPS (92), ICML (62), AISTATS (32), Journal of Machine Learning Research (26), ICLR (21) |
| Physique des hautes énergies, théorie (hep-th) | JHEP (1 207), Phys. Rev. D (676), Eur. Phys. J. C (148), Phys. Lett. B (132), SciPost Phys. (108) |
| Relativité générale et cosmologie quantique (gr-qc) | Phys. Rev. D (898), Eur. Phys. J. C (248), Class. Quantum Grav. (199), JCAP (142), Universe (105) |
| Biologie quantitative (q-bio) | Phys. Rev. E (35), NeurIPS (21), Sci. Rep. (16), Chaos (12), PLOS ONE (12) |
| Physique nucléaire (nucl-ex, nucl-th) | Phys. Rev. C (492), Phys. Lett. B (104), Phys. Rev. D (91), Phys. Rev. Lett. (65), Eur. Phys. J. A (55) |
| Économie (econ) | Journal of Econometrics (5), Econom. Theory (5) |
| Physique mathématique (math-ph) | J. Phys. A (38), J. Math. Phys. (29), J. Phys. A: Math. Theor. (27), Commun. Math. Phys. (15), Ann. Henri Poincaré (12) |
| Physique des hautes énergies, expérience (hep-ex) | Phys. Rev. D (131), JHEP (99), Phys. Rev. Lett. (83), Eur. Phys. J. C (63), Phys. Lett. B (36) |
| Finance quantitative (q-fin) | Entropy (9) |
| Sciences non linéaires (nlin) | Phys. Rev. E (68), Physica D (29), Chaos (24), J. Phys. A (19), Phys. Rev. Lett. (11) |

## Comment savoir si un preprint a été publié

- Sur arXiv, cherchez les lignes « Journal reference » et « DOI » sous le résumé ; elles manquent souvent même quand l’article est publié.
- Sur SummarizePaper, la page d’un article indique sa version publiée quand elle est trouvée, et les lecteurs Premium peuvent être [alertés quand un preprint suivi est publié](https://www.summarizepaper.com/fr/guides/track-research-paper-citations/).
- Pour toute une liste de lecture, Premium surveille une [collection](https://www.summarizepaper.com/fr/guides/organize-research-paper-collections/) et vous écrit quand l’un de ses articles est publié, rétracté ou corrigé.

## Plus de données sur la publication scientifique

- [Statistiques d’arXiv : soumissions par année et par domaine](https://www.summarizepaper.com/fr/guides/arxiv-statistics/)
- [Les sujets de recherche en plus forte croissance](https://www.summarizepaper.com/fr/guides/fastest-growing-research-topics/)
- [Combien d’auteurs par article, par domaine](https://www.summarizepaper.com/fr/guides/authors-per-paper/)

## Méthode et limites

- **Articles** : chaque article arXiv dont l’identifiant commence par l’année (2201 à 2212 pour 2022, etc.), d’après les métadonnées de l’interface OAI-PMH d’arXiv, récupérées le 21 novembre 2026.
- **Publié** : la notice arXiv porte une référence de revue ou un DOI. Les deux sont saisis par les auteurs ou ajoutés par arXiv pour certains éditeurs ; beaucoup d’articles publiés n’ont ni l’un ni l’autre : chaque part est donc un minimum, plus encore dans les domaines qui publient dans des actes de conférence.
- **Domaine** : l’archive de la catégorie principale ; physics.* forme « physique (autres) », nucl-ex et nucl-th la physique nucléaire. Les domaines de moins de 300 articles en 2022 sont exclus.
- **Revues** : noms lus dans la référence de revue, normalisés (abréviations et espaces), ou à défaut dans le préfixe DOI des principaux éditeurs, ou à défaut dans une conférence citée par des commentaires qui disent l’article accepté ; un article dont la revue n’est pas reconnue n’est pas compté dans ce tableau.

<div class="cta-box">
<p>Suivez votre domaine sans lire toutes les listes</p>
<a class="btn btn-lg" href="https://www.summarizepaper.com/fr/digest/">Recevoir l'alerte quotidienne</a>
<a class="btn btn-lg btn-ghost" href="https://www.summarizepaper.com/fr/">Résumer un article</a>
</div>
