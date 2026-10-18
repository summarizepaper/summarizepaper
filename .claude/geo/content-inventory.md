# Inventaire de contenu : SummarizePaper

> Mis à jour par chaque skill de production à la fin de son travail, et par `internal-linking`.
> C'est la carte que `geo` lit pour décider quoi produire ensuite.

## Pages existantes

Aucune page éditoriale (pas de blog). Le contenu indexable existant est :

| URL | Famille | Cluster / hub | Langue | Publié le | Mis à jour le | Liens entrants | Liens sortants | Fact-check |
|---|---|---|---|---|---|---|---|---|
| /<lang>/arxiv-id/<id>/ (10 037 pages au 2026-10-18) | pages article générées | — | en (9 696), fr (341) | depuis 2023 | continu | accueil, arbre, sitemap | arXiv, articles proches | résumés IA, avertissement sur /about/ |
| /en/faq/ | page produit | — | en, fr | — | 2026-10 | menu | pages du site | validé |
| /en/about/ | page produit / auteur | — | en, fr | — | — | menu | — | à enrichir (E-E-A-T) |
| /en/digest/ | page produit | Suivre arXiv | en, fr | 2026 | — | menu | — | — |
| /en/chat/ | page produit | Chat avec un article | en, fr | — | — | menu | — | — |
| /en/premium/ | page produit | — | en, fr | 2026 | — | menu | — | — |

## Hubs

| Hub | Cluster | Spokes rattachés | Spokes manquants |
|---|---|---|---|
| /en/digest/ | Suivre arXiv | — | guide « keep up with arXiv » |
| /en/chat/ | Chat avec un article | — | comparatif « chat with papers » |

## Pages orphelines (0 lien entrant)

- Non mesuré.

## Ressources gated (lead magnets)

| Ressource | Persona | Placement (pages) | Outil emailing / tag | Inscrits |
|---|---|---|---|---|
| Digest quotidien (déjà en place, équivaut à un lead magnet) | chercheurs | /en/digest/ | SMTP du site | à mesurer |

## Outils gratuits

| Outil | URL | Job résolu | Usage (période) | Conversion | Décision (garder / promouvoir / laisser vivre) |
|---|---|---|---|---|---|
| Résumé d'un article par ID | /en/ | résumer un article | à mesurer (Umami) | — | garder |

## Prérequis infra

- Rendu serveur vérifié le : 2026-10-18 — OK (texte du résumé dans le HTML brut, environ 13 000 caractères visibles sur une page testée)
- Déploiement effectif vérifié le : 2026-10-18 — OK (pages article, FAQ, about, chat, digest, premium et privacy en 200)
- Title, meta description, canonical : OK sur les pages article ; meta description de l'accueil générique (« cutting-edge technology »)
- Sitemap vérifié le : 2026-10-18 — OK, index déclaré dans robots.txt, 10 037 pages article
- Bots IA autorisés vérifié le : 2026-10-18 — OK (GPTBot, OAI-SearchBot, ClaudeBot, PerplexityBot, CCBot en 200, sans challenge)
- JSON-LD : `ScholarlyArticle` sur les pages article ; **manque `Organization` et `WebSite` au niveau du site** (aucun JSON-LD sur l'accueil)
- Page auteur : **manquante**
- hreflang : OK sur les pages article
- llms.txt : OK (2026-10-18), sans section « Citation » pour l'instant
- Blocages ouverts : aucun blocage bloquant. À corriger avant la production en masse : (1) pas de dossier ni de vue pour publier des pages éditoriales ; (2) schema `Organization` / `WebSite` ; (3) page auteur ; (4) Search Console à configurer.
