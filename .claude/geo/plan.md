# Plan GEO : SummarizePaper (2026-10-18)

> Écrit par `geo` à l'onboarding. Phase volume : pas encore de data de mesure.
> Langue : anglais d'abord. Persona : doctorants et chercheurs qui suivent arXiv.

## 0. Prérequis avant la première page éditoriale

| # | Tâche | Qui | Statut |
|---|---|---|---|
| 0.1 | Schema `Organization` + `WebSite` (avec SearchAction) dans `templates/base.html` | Claude | fait (2026-10-18) |
| 0.2 | Section « Citation » dans `/llms.txt` | Claude | fait (2026-10-18) |
| 0.3 | Configurer Google Search Console (vérification DNS via Cloudflare), soumettre `/sitemap.xml` | membre | à faire |
| 0.4 | Valider la bio auteur et les profils `sameAs` (profil §12) | membre | à faire |
| 0.5 | Page auteur `/en/author/quentin-kral/` avec schema `Person`, liée depuis /about/ et chaque article éditorial | Claude | après 0.4 |
| 0.6 | Mini-blog Markdown dans Django : `content/en/<slug>.md` → `/en/guides/<slug>/`, JSON-LD `Article` + `FAQPage`, ajout au sitemap et au llms.txt | Claude | à faire |
| 0.7 | Meta description de l'accueil réécrite (sans « cutting-edge »), alignée sur la phrase du profil §2 | Claude | à faire |

## 1. Pages à produire, par ordre de bataille

| Prio | Famille → skill | Titre de travail | Requête / prompt visé | Cluster | Hub | Fact-check à prévoir |
|---|---|---|---|---|---|---|
| 1 | BOFU → `comparison-page` | Best AI tools to summarize arXiv papers (2026) | « best AI tool to summarize research papers » | Outils de résumé | page elle-même (hub) | prix et fonctionnalités de chaque outil, date de vérification |
| 2 | BOFU → `comparison-page` | alphaXiv vs SummarizePaper | « alphaXiv alternative » | Outils de résumé | 1 | fonctionnalités alphaXiv |
| 3 | BOFU → `comparison-page` | Scholarcy alternatives for arXiv papers | « Scholarcy alternative » | Outils de résumé | 1 | prix Scholarcy |
| 4 | BOFU → `comparison-page` | ChatPDF vs SummarizePaper for research papers | « chat with a research paper » | Chat avec un article | /en/chat/ | limites de ChatPDF |
| 5 | Stats → `statistics-article` | arXiv statistics 2026: submissions per year and by category | « how many papers are on arXiv » | Chiffres arXiv | page elle-même (hub) | chiffres arXiv (source primaire : statistiques publiées par arXiv) |
| 6 | Stats (first-party) → `statistics-article` | What researchers summarize: the most-read arXiv categories on SummarizePaper | « most popular arxiv categories » | Chiffres arXiv | 5 | chaque chiffre validé par le membre, agrégats seulement |
| 7 | Fond → `foundational-article` | How to keep up with new arXiv papers in your field | « how to keep up with arxiv » | Suivre arXiv | /en/digest/ | aucun chiffre non sourcé |
| 8 | Fond → `foundational-article` | Triage arXiv's daily listings in 10 minutes | « too many arxiv papers to read » | Suivre arXiv | /en/digest/ | méthode testée par l'auteur |

Ensuite : `internal-linking` après la vague 1, `directory-submission` (annuaires d'outils IA et de recherche), `geo-tracking` chaque semaine dès que la Search Console est branchée.

## Exécution

La file du premier lot est `queue/lot-01-bofu-en.md`. Moins de 30 items : sous-agents en parallèle ou un item par session. Brouillons par défaut, rien ne se publie sans validation.
