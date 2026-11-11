---
title: 'Zotero : envoyer des articles avec leurs résumés, et synchroniser une collection'
description: 'Envoyez des articles dans Zotero avec leurs résumés IA, synchronisez une collection dans les deux sens, et écrivez vos vérifications et revues dans Zotero.'
date: 2026-11-10
draft: false
faq:
  - q: "Faut-il Premium pour utiliser Zotero ?"
    a: "Non pour enregistrer des articles : le Zotero Connector de votre navigateur reconnaît nos pages d’articles, et chaque article se télécharge en fichier RIS. La connexion directe (articles envoyés avec leurs résumés, collections synchronisées, résultats écrits dans Zotero) est Premium."
  - q: "Cela crée-t-il des doublons dans ma bibliothèque ?"
    a: "Non. Un article déjà dans votre bibliothèque (même DOI, identifiant arXiv ou titre) reçoit la collection, les étiquettes et les notes qui lui manquent au lieu d’un second élément."
  - q: "Que deviennent mes éléments Zotero quand je retire un article ?"
    a: "Retirer un article d’une collection synchronisée le sort de la collection Zotero à la synchronisation suivante, mais l’élément reste dans votre bibliothèque."
  - q: "Ma clé Zotero est-elle protégée ?"
    a: "Avec la connexion en un clic, c’est Zotero qui donne une clé à SummarizePaper, après que vous vous êtes connecté à Zotero ; avec une clé collée, c’est vous qui la créez. Dans les deux cas, elle est stockée chiffrée et ne sert qu’à lire et écrire votre bibliothèque quand vous le demandez, ou pour la synchronisation quotidienne de vos collections. Déconnectez-la à tout moment depuis la page des intégrations."
sources:
  - label: Zotero — Ajouter des éléments à Zotero (Zotero Connector)
    url: https://www.zotero.org/support/adding_items_to_zotero
  - label: Zotero — API web
    url: https://www.zotero.org/support/dev/web_api/v3/start
  - label: Better BibTeX pour Zotero
    url: https://retorque.re/zotero-better-bibtex/
---

Vos références vivent dans Zotero, et vous lisez les articles ailleurs. **SummarizePaper écrit dans votre bibliothèque Zotero** : un article avec son résumé, une collection synchronisée dans les deux sens, les résultats de votre vérification de bibliographie et votre revue de littérature. Enregistrer un article ne demande aucun compte ; la connexion directe est Premium.

Les captures montrent le côté SummarizePaper, avec la collection *Gaz dans les disques de débris* de nos autres guides : trois articles sur le gaz des disques de débris, chacun avec une note, des étiquettes et un statut de lecture.

## 1. Enregistrer un article, gratuitement

Sur chaque page d’article, le **Zotero Connector** de votre navigateur reconnaît l’article et l’enregistre avec ses métadonnées. Dans **Outils pour cet article**, **Exporter** donne aussi le fichier **RIS**, que Zotero, Mendeley et EndNote importent ; il contient la référence et le résumé original. Un article publié depuis sa prépublication est exporté dans sa version publiée, avec sa revue et son DOI, et son identifiant arXiv conservé.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/exports/paper-export-fr.jpg"><img src="/static/summarizer/img/guides/exports/paper-export-fr.jpg" alt="Les outils d’une page d’article : exporter en BibTeX, RIS (Zotero, Mendeley, EndNote), Obsidian et Notion, et, parmi les outils Premium, Envoyer vers Zotero avec son résumé." width="936" height="266" loading="lazy" /></a>
<figcaption>Le RIS pour tout gestionnaire de références, gratuit ; « Envoyer vers Zotero avec son résumé » une fois la bibliothèque connectée.</figcaption>
</figure>

## 2. Connecter votre bibliothèque

Sur la [page des intégrations](/fr/integrations/), cliquez sur **Connecter ma bibliothèque Zotero** : Zotero vous demande de vous connecter et d’autoriser SummarizePaper à lire et écrire votre bibliothèque et vos groupes. Aucune clé à copier. Vous pouvez aussi coller une clé privée créée sur zotero.org, avec l’accès en écriture.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/exports/integrations-zotero-fr.jpg"><img src="/static/summarizer/img/guides/exports/integrations-zotero-fr.jpg" alt="Votre bibliothèque Zotero (Premium) : ce que fait la connexion, et le bouton Connecter ma bibliothèque Zotero, ou coller une clé." width="916" height="623" loading="lazy" /></a>
<figcaption>Un clic : Zotero vous demande d’autoriser la connexion.</figcaption>
</figure>

Une fois connecté, choisissez la bibliothèque utilisée (la vôtre ou un groupe où vous pouvez écrire) et la collection Zotero où vont les articles seuls. La page liste vos collections Zotero, pour en ouvrir une comme collection ici ou vérifier ses références.

## 3. Envoyer un article avec son résumé

**Envoyer vers Zotero avec son résumé** crée l’élément, dans sa version publiée s’il y en a une, avec **une note qui contient le résumé IA et les points clés**. La note dit qu’elle a été écrite par un modèle, à partir de quelle version de l’article et quand ; votre propre note est une note à part. Un article déjà dans votre bibliothèque n’est pas dupliqué : il reçoit la collection, les étiquettes et les notes qui lui manquent. L’élément porte notre clé de citation, fixée pour **Better BibTeX**, si bien que vos `\cite` correspondent au `.bib` de nos exports.

## 4. Synchroniser une collection, dans les deux sens

Dans une [collection](/fr/collections/), **Envoyer vers Zotero et synchroniser** crée une collection Zotero, ou en relie une, et les garde ensemble :

- les articles ajoutés d’un côté arrivent de l’autre ; un article retiré d’un côté quitte l’autre (l’élément Zotero reste dans votre bibliothèque) ;
- vos notes, étiquettes et statuts de lecture vont dans Zotero, en notes et en étiquettes ;
- les rétractations, corrections et versions publiées trouvées plus tard arrivent aussi dans votre bibliothèque : la collection est **resynchronisée chaque jour**, sans que vous ayez rien à faire.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/exports/collection-export-fr.jpg"><img src="/static/summarizer/img/guides/exports/collection-export-fr.jpg" alt="La collection Gaz dans les disques de débris avec ses exports : BibTeX, RIS, Ouvrir dans Overleaf, Obsidian (.zip), Markdown, Tableur ou Notion (CSV), et Envoyer vers Zotero et synchroniser (Premium)." width="1156" height="398" loading="lazy" /></a>
<figcaption>« Envoyer vers Zotero et synchroniser », à côté des autres exports de la collection.</figcaption>
</figure>

Dans l’autre sens, une collection Zotero s’ouvre ici avec **Depuis Zotero**, sous *Ajouter des articles* : vous pouvez alors discuter avec ses articles, les comparer dans un tableau, trouver les articles qui pourraient lui manquer ou écrire sa revue de littérature, et elle reste synchronisée.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/exports/collection-zotero-fr.jpg"><img src="/static/summarizer/img/guides/exports/collection-zotero-fr.jpg" alt="Depuis Zotero (Premium) : connectez votre bibliothèque Zotero pour importer une de ses collections." width="1156" height="71" loading="lazy" /></a>
<figcaption>Importer toute une collection Zotero.</figcaption>
</figure>

## 5. Vos résultats, écrits dans Zotero

- **Vérification de bibliographie** : sur chaque référence rétractée, corrigée ou publiée depuis, des étiquettes et une note dans votre bibliothèque, et la version publiée ajoutée à côté de sa prépublication. Une collection Zotero peut aussi être vérifiée avec votre manuscrit : les clés de citation de Better BibTeX sont reconnues, et les phrases signalées sont relues dans les PDF que vous gardez dans Zotero, pour votre vérification seulement, sans les stocker. Voir le [guide Overleaf et bibliographie](/fr/guides/overleaf-bibliography-check/).
- **Revue de littérature** : une collection Zotero avec **une sous-collection par approche**, chaque article avec une note qui contient les affirmations qu’il appuie et sa phrase qui les appuie. Voir le [guide de la revue de littérature](/fr/guides/literature-review-from-a-research-question/).
- **Qu’est-ce que j’ai raté ?** : les articles proches que votre brouillon ne cite pas, sans ceux que vous avez marqués déjà cités ou pas pertinents. Voir [son guide](/fr/guides/what-did-i-miss-uncited-papers/).

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/literature/searched-fr.jpg"><img src="/static/summarizer/img/guides/literature/searched-fr.jpg" alt="Les exports d’une revue de littérature, avec Envoyer vers Zotero, au-dessus de Comment cette revue a été construite." width="916" height="554" loading="lazy" /></a>
<figcaption>« Envoyer vers Zotero » sur une revue de littérature.</figcaption>
</figure>

Vos collections Zotero peuvent aussi servir d’**articles de référence** à votre [sélection du matin](/fr/digest/) et à votre [veille anti-scoop](/fr/scoops/) : les nouveaux articles sont classés selon leur proximité avec eux.

## Sans Premium

Les exports RIS et BibTeX sont gratuits, sur chaque article et chaque collection, et le Zotero Connector fonctionne sur chaque page d’article. Le `.bib` d’une collection peut aussi être relié à Overleaf, gratuitement ; voir le [guide Overleaf](/fr/guides/overleaf-bibliography-check/).
