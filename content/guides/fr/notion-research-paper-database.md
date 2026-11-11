---
title: 'Notion : une page par article, et une base de données de votre collection'
description: 'Importez un article comme page Notion avec ses points clés et son résumé, et une collection comme base Notion avec statuts de lecture, étiquettes et notes.'
date: 2026-11-10
draft: false
faq:
  - q: "Est-ce gratuit ?"
    a: "Oui. Le Markdown d’un article et le tableur d’une collection se téléchargent gratuitement."
  - q: "SummarizePaper écrit-il dans mon espace Notion ?"
    a: "Non. Vous téléchargez un fichier et l’importez vous-même dans Notion : SummarizePaper n’a jamais besoin d’accéder à votre espace."
  - q: "Comment ajouter plus tard de nouveaux articles à la base ?"
    a: "Exportez à nouveau la collection et, dans Notion, utilisez Fusionner avec un CSV (Merge with CSV) sur la base : les lignes sont ajoutées comme nouvelles lignes, les lignes existantes ne sont pas mises à jour. Ou importez le nouvel export comme une nouvelle base."
sources:
  - label: Aide de Notion — Importer des données dans Notion
    url: https://www.notion.com/help/import-data-into-notion
---

Vous organisez vos recherches dans Notion et voulez y retrouver vos articles : une page par article lu, et le tableau d’une liste de lecture avec ses statuts, étiquettes et notes. SummarizePaper exporte **un article en Markdown**, que Notion importe comme une page, et **une collection en tableur** (CSV), que Notion importe comme une base de données. C’est gratuit, et rien n’est écrit dans votre espace : vous importez les fichiers vous-même.

Les exemples viennent de la collection *Gaz dans les disques de débris* de nos autres guides : trois articles sur le gaz des disques de débris, chacun avec une note, des étiquettes et un statut de lecture.

## 1. Un article comme page Notion

Sur la page de résumé d’un article, ouvrez **Outils pour cet article**, puis **Exporter → Notion (Markdown)**.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/exports/paper-export-fr.jpg"><img src="/static/summarizer/img/guides/exports/paper-export-fr.jpg" alt="Les outils d’une page d’article : exporter en BibTeX, RIS, Obsidian (Markdown) et Notion (Markdown)." width="936" height="266" loading="lazy" /></a>
<figcaption>« Notion (Markdown) », dans les outils de chaque page d’article.</figcaption>
</figure>

Le fichier contient le titre de l’article en titre, sa référence, puis ses points clés et son résumé, dans la langue du site :

```markdown
# Observing planetary gaps in the gas of debris disks

**Auteurs:** C. Bergez-Casalou, Q. Kral  
**Publié dans:** Astronomy & Astrophysics  
**DOI:** [10.1051/0004-6361/202452097](https://doi.org/10.1051/0004-6361/202452097)  
**arXiv:** [2411.14241](https://arxiv.org/abs/2411.14241)  
**Date:** 2024-11-21  
**Page du résumé:** https://www.summarizepaper.com/fr/paper/2411.14241v1/

## Points clés

- Objectif : détecter indirectement des planètes géantes froides grâce aux sillons et aux perturbations locales de vitesse (« kinks ») qu’elles provoquent dans le gaz des disques de débris, observables avec ALMA.
- …

## Résumé
…
```

Dans Notion, ouvrez **Paramètres** (*Settings*) dans la barre latérale, puis **Importer → Texte et Markdown**, et choisissez le fichier : il devient une page avec ses titres, ses libellés en gras, ses liens et ses listes. Le résumé se termine par sa provenance (écrit par un modèle d’IA, à partir de quelle version de l’article et quand). Les formules sont écrites en LaTeX entre dollars : Notion peut les afficher comme du texte.

## 2. Une collection comme base de données Notion

Dans une [collection](/fr/collections/), cliquez sur **Tableur ou Notion (CSV)**. Réglez d’abord dans la collection le statut de lecture, les étiquettes et les notes des articles : ce sont des colonnes du tableur.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/exports/collection-export-fr.jpg"><img src="/static/summarizer/img/guides/exports/collection-export-fr.jpg" alt="La collection Gaz dans les disques de débris avec ses exports, dont Tableur ou Notion (CSV)." width="1156" height="398" loading="lazy" /></a>
<figcaption>« Tableur ou Notion (CSV) », parmi les exports d’une collection.</figcaption>
</figure>

Le fichier a une ligne par article, avec ces colonnes (leurs noms restent en anglais) : *Identifier, Title, Authors, Year, Venue, Editorial status, Notices, Citations, Newer version, Reading status, Tags, Note, Selected, Link*. Pour notre collection :

| Title | Year | Venue | Editorial status | Reading status | Tags |
| --- | --- | --- | --- | --- | --- |
| Thermochemical constraints on a primordial-origin of gas-rich debris disks | 2026 | | Soumission ou évaluation annoncée sur arXiv par les auteurs | Article clé | origine, thermochimie |
| Observing planetary gaps in the gas of debris disks | 2024 | Astronomy & Astrophysics | Publication retrouvée | À lire | planètes, ALMA |
| Predictions for the secondary CO, C and O gas content of debris discs… | 2017 | Monthly Notices of the Royal Astronomical Society | Publication retrouvée | Lu | origine, modèles |

**Editorial status**, **Notices** et **Newer version** viennent de la vérification de l’endroit où les articles ont été publiés, de leurs avis de rétractation ou de correction et de leurs nouvelles versions : lancez cette vérification dans la collection avant d’exporter pour les remplir.

Dans Notion, ouvrez **Paramètres**, puis **Importer → CSV**, et choisissez le fichier : il devient une base de données avec une ligne par article. Adaptez ensuite la base à votre usage : passez *Reading status* en propriété Sélection pour grouper les articles par statut dans un tableau Kanban, *Tags* en Sélection multiple, *Link* en URL. Notion prend la première colonne comme nom de chaque ligne ; la deuxième, *Title*, est le titre de l’article.

Pour ajouter les articles mis plus tard dans la collection, exportez-la à nouveau et choisissez **Fusionner avec un CSV** dans le menu de la base : Notion ajoute des lignes, il ne met pas à jour celles qui y sont déjà.

## Autres fichiers pour Notion

- **Markdown** de toute une collection : un seul fichier avec tous ses articles, à importer comme une page.
- Une **revue de littérature** s’exporte en Markdown, sa synthèse avec ses références numérotées ([guide de la revue de littérature](/fr/guides/literature-review-from-a-research-question/)).
- Le **compagnon de conférence** exporte votre programme et vos notes en Markdown ([guide de la conférence](/fr/guides/conference-program-talks-to-attend/)).
- Pour Obsidian, avec des propriétés et des liens entre les notes, voir le [guide Obsidian](/fr/guides/obsidian-research-paper-notes/).
