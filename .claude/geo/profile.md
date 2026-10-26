# Profil GEO : SummarizePaper

> Ce fichier est la source de vérité de tous les skills du plugin GEO Engine.
> Il est écrit par l'onboarding du skill `geo`, puis maintenu à la main.
> Règle : garder la copy identique à celle du site en ligne (cohérence d'entité,
> c'est un signal que les IA utilisent pour associer le nom au produit).
>
> Les lignes marquées `[À CONFIRMER]` ont été déduites du repo par l'onboarding
> et n'ont pas encore été validées par un humain.

## 1. Identité

- **Nom** : SummarizePaper
- **URL publique** : https://www.summarizepaper.com
- **Catégorie principale** : AI research paper summarizer (résumés IA d'articles arXiv)
- **Catégories secondaires** : chat with papers / AI research assistant, veille arXiv par email (digest)
- **Plateformes** : web uniquement (anglais et français)
- **Modèle et prix** : freemium. Gratuit : lecture illimitée des résumés existants, 10 nouveaux résumés par jour, 30 questions par jour sans compte, 100 avec compte gratuit. Premium : 7,00 EUR / mois (prix Stripe lu sur /en/premium/ le 2026-10-18), modèle le plus puissant, pas de limite de résumés, 500 questions par jour.
- **Pays et année de création** : France, 2023 (premier commit du 2023-02-14)
- **Entité juridique** : CarbonFreeConf, SASU, RCS 895 105 955, 3 bis rue de la Garenne, 17170 Courçon, France
- **Contact public** : https://www.summarizepaper.com/en/contact/

## 2. Proposition de valeur

- **En une phrase** : SummarizePaper publie des résumés IA des articles arXiv, en anglais et en français, générés à partir du texte intégral : résumé détaillé, points clés, résumé grand public, article de blog, articles proches et BibTeX, avec un assistant qui répond aux questions sur l'article.
- **H1 réel de la page d'accueil** : Discover Key Insights from arXiv's Latest Articles
- **Sous-titre réel** : (pas de sous-titre ; H2 « Chat with arXiv Papers » et « Latest Papers summarized with our AI tools »)
- **Meta description réelle** : Our AI-powered arXiv paper summarization service provides fast, accurate, and easy-to-read summaries of scientific papers. Save time and stay up-to-date with the latest research with our cutting-edge technology.
- **`<title>` réel de l'accueil** : AI-Powered arXiv Paper Summarization - Ask questions about a Paper

## 3. Niche et vocabulaire

- **Le domaine en deux mots** : résumés d'articles scientifiques (arXiv)
- **Les termes que le marché tape ou demande** : summarize arxiv paper, AI paper summarizer, research paper summary, explain this paper, chat with PDF / chat with paper, arxiv digest, keep up with arxiv, lay summary, TL;DR of a paper `[À CONFIRMER par keyword-pipeline]`
- **Prompts réels posés à une IA** `[À CONFIRMER]` :
  1. « What is the best AI tool to summarize arXiv papers? »
  2. « How can I keep up with new arXiv papers in my field without reading them all? »
  3. « Summarize arXiv 2310.06825 for me » (requête par identifiant, la plus fréquente sur le site)
  4. « Is there a tool to chat with a research paper and ask questions about it? »
  5. « Explain this physics paper in simple terms »
- **Concepts que toutes les IA connaissent déjà** (pas d'article générique) : ce qu'est arXiv, ce qu'est un preprint, la peer review, comment lire un article scientifique en général, ce qu'est un LLM.

## 4. Cibles

### Persona principal
- **Qui** : doctorants, postdocs et chercheurs qui suivent arXiv chaque semaine (fort en astro-ph, physique, informatique / IA).
- **Problème n°1** : trop d'articles publiés chaque jour pour tous les lire ; difficile de trier vite ce qui mérite une lecture complète.
- **Ce qu'il cherche à obtenir** : décider en une minute si un article compte, en retenir l'essentiel, et suivre son domaine sans y passer des heures.

### Personas secondaires
- Étudiants et curieux qui veulent comprendre un article sans être spécialistes (résumé grand public).
- Journalistes et médiateurs scientifiques (citent leurs sources : levier backlinks).
- Lecteurs francophones (résumés en français, rares ailleurs).

## 5. Jobs-to-be-done

1. « Résume-moi cet article arXiv avant que je décide de le lire. »
2. « Explique-moi cet article avec des mots simples. »
3. « Réponds à mes questions sur cet article (méthode, résultats, limites) sans que je relise le PDF. »
4. « Préviens-moi chaque matin des nouveaux articles de mes catégories, avec deux phrases chacun. »
5. « Trouve-moi les articles proches de celui-ci. »
6. « Donne-moi le BibTeX / un PDF du résumé pour mes notes. »
7. « Lis-moi le résumé d'un article en français. »

## 6. Fonctionnalités clés

1. Résumé d'un article (détaillé, points clés, grand public, blog) : https://www.summarizepaper.com/en/arxiv-id/<id>/
2. Assistant IA sur un ou plusieurs articles : https://www.summarizepaper.com/en/chat/
3. Digest quotidien par email : https://www.summarizepaper.com/en/digest/
4. Articles proches (arbre) : https://www.summarizepaper.com/en/tree/<id>/
5. Export PDF du résumé et BibTeX : sur chaque page article
6. Premium : https://www.summarizepaper.com/en/premium/
7. FAQ : https://www.summarizepaper.com/en/faq/

## 7. Concurrents

Voir `competitors.md` dans ce dossier. Résumé ici :
- **Directs** : alphaXiv, Scholarcy, SciSpace, Explainpaper, Paper Digest `[À CONFIRMER]`
- **Adjacents** : ChatPDF, Elicit, Consensus, Semantic Scholar (TLDR), arxiv-sanity, ChatGPT / Claude / Perplexity eux-mêmes `[À CONFIRMER]`
- **Ceux qu'on cite dans les comparatifs** : à choisir après `keyword-pipeline`.

## 8. Data first-party

Usage validé par le membre le 2026-10-18 : **chiffres agrégés uniquement**, jamais rien sur un utilisateur, chaque chiffre validé par le membre avant publication.

- **Datasets disponibles** :
  - `SummaryPaper` : résumés publiés par article et par langue (10 037 pages résumées dans le sitemap au 2026-10-18 : 9 696 en anglais, 341 en français) ;
  - `ArxivPaper` : métadonnées des articles (catégorie arXiv, dates, auteurs) ;
  - `PaperHistory` / `PDFHistory` : lancements de résumés et exports PDF par article, langue et date (IP hachées, à ne jamais publier) ;
  - `Vote` : votes sur la qualité des résumés ;
  - réglages du digest : catégories arXiv choisies par les abonnés (agréger seulement).
- **Comment y accéder en lecture** : requêtes `SELECT` agrégées via `heroku pg:psql -a summarizepaper`, ou un script de commande Django en lecture seule. Le membre valide la requête avant exécution.
- **Chiffres canoniques** (requêtés le 2026-10-18, à réutiliser à l'identique) : 9 995 articles arXiv résumés entre février 2023 et octobre 2026 ; 10 037 résumés (en, fr) ; 10 963 demandes de résumé ; 7 618 questions à l'assistant ; 1 431 exports PDF ; 64,8 % d'articles d'informatique (contre 44,2 % sur arXiv, mars 2023 à septembre 2026) ; cs.CL 15,9 % (2,8 fois sa part d'arXiv) ; mathématiques 2,9 % (contre 16,1 %) ; âge médian au premier résumé 387 jours ; 75,3 % de votes positifs sur 344. Requêtes : `data/first-party-queries.sql`.
- **Études ou pages de données** : /en/guides/what-researchers-summarize/ (first-party) et /en/guides/arxiv-statistics/ (sources arXiv), publiés le 2026-10-18.

> Ne jamais mettre d'identifiant de base, de clé ou de token ici. Ce fichier est versionné.

## 9. Voix

- **Langue(s) de production** : anglais d'abord ; français plus tard pour les pages qui marchent.
- **Tutoiement ou vouvoiement** : en anglais « you » ; en français, le site vouvoie.
- **Ton** : simple, factuel, modeste (« one researcher in astrophysics »), orienté accès de tous à la recherche, avec avertissement sur les limites de l'IA.
- **Interdits** : pas d'emojis, pas de superlatifs creux (« cutting-edge » est déjà trop présent), pas de promesse d'exactitude des résumés IA, pas de chiffre non sourcé.
- **Exemple de page qui incarne la voix** : https://www.summarizepaper.com/en/faq/

## 10. Site et stack

- **Framework** : Django 5.2 (templates serveur), servi par Daphne sur Heroku, derrière Cloudflare.
- **Dossier des pages de contenu** : `content/guides/<lang>/<slug>.md`, servi sous `/<lang>/guides/<slug>/` par `summarizer/guides.py`. Front matter YAML : `title`, `description`, `date`, `updated`, `draft`, `faq` (liste q/a), `sources` (liste label/url). Le même slug dans `en/` et `fr/` est relié par hreflang. Les brouillons (`draft: true`) ne sont visibles que par le staff, en noindex, et restent hors du sitemap et du llms.txt.
- **Format d'une page** : Markdown (extensions extra, toc, sane_lists), rendu par `templates/summarizer/guide.html`, qui émet le JSON-LD `Article`, `BreadcrumbList` et `FAQPage`, et affiche la FAQ et les sources.
- **Rendu** : SSR (texte du résumé présent dans le HTML brut, vérifié le 2026-10-18).
- **Internationalisation** : préfixes `/en/` et `/fr/` (i18n_patterns), hreflang présent sur les pages article.
- **Sitemap** : `/sitemap.xml` (index) → `sitemap-papers.xml` (pages résumées seulement) + `sitemap-static.xml`.
- **robots.txt autorise les bots IA** : oui (seul `/admin` interdit). GPTBot, OAI-SearchBot, ClaudeBot, PerplexityBot et CCBot reçoivent la page en 200, sans challenge Cloudflare (vérifié le 2026-10-18).
- **Page auteur et schema author** : existe, /en/author/quentin-kral/
- **Adaptateur choisi** : `markdown-generic` (blog Markdown maison dans Django).
- **Exemple de page de référence** : `templates/summarizer/faq.html` (structure et mise en page du site).
- **llms.txt** : `/llms.txt` servi depuis `templates/llms.txt` (en ligne depuis le 2026-10-18).

## 11. Outils connectés

- **Suivi de la visibilité IA** : aucun (mesure manuelle des prompts)
- **Analytics produit** : Umami Cloud (sans cookie, depuis le 2026-10-18)
- **Google Search Console** : configurée par le membre le 2026-10-18 (sitemap soumis) ; accès API en lecture pour Claude depuis le 2026-10-26 (compte de service `claude-gsc@summarizepaper.iam.gserviceaccount.com`, propriété `sc-domain:summarizepaper.com`, autorisation restreinte : inspection d'URL, sitemaps, performances ; le rapport Indexation → Pages reste en export CSV)
- **Emailing** : SMTP transactionnel du site (digest, comptes) ; Lumail disponible en MCP dans la session
- **Automatisation navigateur** : aucun
- **Recherche de mots-clés** : aucun (expansion manuelle)

## 12. Auteur (E-E-A-T)

- **Nom** : Quentin Kral
- **Bio en deux lignes** : astronome adjoint à l'Observatoire de Paris depuis 2018 (exoplanètes, disques de débris, Système solaire), docteur de l'Observatoire de Paris (2014), prix de thèse SF2A 2015, HDR 2024, responsable de l'Encyclopédie des planètes extrasolaires. Créateur et seul développeur de SummarizePaper. Source : Wikipédia FR, validée par le membre le 2026-10-18.
- **Page auteur** : https://www.summarizepaper.com/en/author/quentin-kral/ (schema `Person`, `@id` …/#person, référencé par `Organization.founder` et par l'`author` de chaque guide)
- **Profils publics à lier** (sameAs) : https://fr.wikipedia.org/wiki/Quentin_Kral, https://www.wikidata.org/wiki/Q116781126, https://orcid.org/0000-0001-6527-4684

## 13. Multi-produits (optionnel)

- **Produits de la suite** : CarbonFreeConf (carbonfreeconf.com), même entité juridique, cité dans la FAQ. `[À CONFIRMER : même niche ou non]`
- **Produit hub** : SummarizePaper pour la niche « résumés d'articles ».
- **Règles d'interlink** : liens contextuels uniquement, jamais de footer croisé massif.

## 14. État

- **Inventaire de contenu** : `content-inventory.md` dans ce dossier
- **Univers de mots-clés** : `keywords.md` dans ce dossier
- **Files d'exécution** : `queue/` dans ce dossier
- **Dernier rituel de mesure** : aucun
- **Phase actuelle** : volume (pas encore de data de mesure)
