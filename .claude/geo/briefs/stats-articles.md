# Brief de relecture : articles de statistiques (plan prio 5 et 6)

| Fichier | Requête visée | Données |
|---|---|---|
| `content/guides/en/arxiv-statistics.md` | « arxiv statistics », « how many papers on arxiv », « arxiv submissions per year » | Sources primaires arXiv, téléchargées le 2026-10-18 |
| `content/guides/en/what-researchers-summarize.md` | « most popular arxiv categories », « what papers do people summarize » | Base SummarizePaper (agrégats) + arXiv |

Publiés ensemble le 2026-10-18. Points 1 et 2 validés par le membre ; graphiques ajoutés (point 3).

## Données et reproductibilité

- **arXiv, dépôts mensuels** : `https://arxiv.org/stats/get_monthly_submissions` (CSV). Totaux annuels = somme des mois de l'année civile. Total au 2026-10-18 : 3 215 245 (affiché sur la page Monthly Submissions).
- **arXiv, catégories** : données des graphiques Plotly de `https://info.arxiv.org/about/reports/submission_category_by_year.html` (149 catégories, janvier 2022 à septembre 2026), décodées en local. Total 2025 : 283 120, contre 284 486 dans le CSV mensuel (écart de 0,5 %, signalé dans l'article).
- **SummarizePaper** : `.claude/geo/data/first-party-queries.sql`, exécuté en lecture seule le 2026-10-18 (transaction annulée), plus une requête sur la part des lancements par des comptes staff (131 sur 10 963).

## Choix éditoriaux

- **Non publiés** : volumes annuels de résumés (ils décrivent l'activité du site, pas les lecteurs) ; catégories du digest (2 abonnés actifs, trop peu pour être anonymes ou significatifs).
- **Biais signalé dans l'article** : la sur-représentation de l'astrophysique (10,5 % contre 6,5 %) est probablement liée à l'auteur ; seuls 46 lancements astro-ph viennent de comptes staff.
- **Interprétations** formulées comme hypothèses (« one likely reason », « this suggests »).

## [FACT-CHECK HUMAIN] avant publication

1. **Liste des 10 articles les plus résumés** : titres d'articles publics, mais vérifier que tu es d'accord pour afficher « PRAGMA: Revolut Foundation Model » (article d'entreprise) et le nombre de demandes par article.
2. **« SummarizePaper was created by an astrophysicist, which likely brings readers from that community »** : hypothèse à confirmer ou retirer selon ce que tu sais de ton audience.
3. **Visuels** : pas de graphique. Un histogramme des dépôts annuels (article 5) et un graphique « part sur SummarizePaper contre part d'arXiv » (article 6) rendraient les pages plus citables ; à produire si souhaité.
4. **Mise à jour** : l'article 5 promet une mise à jour quand arXiv publie de nouveaux chiffres. Prévoir une relance annuelle (ou trimestrielle).
