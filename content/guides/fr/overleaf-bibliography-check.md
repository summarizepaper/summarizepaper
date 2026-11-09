---
title: 'Vérifier la bibliographie d’un projet Overleaf, et la garder à jour'
description: 'Articles rétractés, prépublications publiées depuis, entrées fausses : vérifiez le .bib de votre projet Overleaf, rouvrez-le corrigé et synchronisez-le.'
date: 2026-11-09
draft: false
faq:
  - q: "Faut-il un abonnement Overleaf payant ?"
    a: "Non. La vérification lit le .zip que tout projet Overleaf peut télécharger. Ouvrir un projet dans Overleaf et relier un fichier depuis une URL externe n’utilisent pas les intégrations payantes d’Overleaf (Zotero, Mendeley, Git)."
  - q: "SummarizePaper garde-t-il mon manuscrit ?"
    a: "Non. L’archive est lue en mémoire pour la vérification : seulement ses fichiers .tex, .bib et .bbl, jamais extraits sur disque. Le rapport garde ce qu’il montre : les problèmes trouvés, les entrées corrigées et les phrases des citations signalées."
  - q: "SummarizePaper peut-il écrire dans mon projet Overleaf existant ?"
    a: "Overleaf ne permet à aucun autre site d’écrire dans un projet. « Ouvrir dans Overleaf » crée un nouveau projet ; pour mettre à jour le vôtre, remplacez son .bib, ou reliez le .bib d’une de vos collections, qu’Overleaf relit quand vous cliquez sur Actualiser."
sources:
  - label: Overleaf — Ajouter un fichier depuis une URL (en anglais)
    url: https://docs.overleaf.com/managing-projects-and-files/adding-files-to-a-project/adding-a-file-from-a-url
  - label: Overleaf — Ouvrir un projet depuis un autre site (développeurs, en anglais)
    url: https://www.overleaf.com/devs
---

Vous êtes sur le point de soumettre, et votre article a été écrit sur Overleaf. Son `.bib` a grossi pendant deux ans : des prépublications ont été publiées depuis, un article cité a peut-être été rétracté, et quelques entrées ont été copiées avec la mauvaise année ou une liste d’auteurs tronquée. **La vérification de bibliographie lit tout votre projet Overleaf et vous dit quelles références corriger**, et à quoi sert chaque citation.

## 1. Téléchargez votre projet depuis Overleaf

Dans Overleaf, ouvrez le menu et téléchargez la **source** du projet en `.zip`. Rien à nettoyer : la vérification ne lit que les fichiers `.tex`, `.bib` et `.bbl` qu’il contient, trouve le fichier principal (celui qui contient `\begin{document}`) et suit ses `\input` et `\include`.

Vous pouvez aussi coller votre LaTeX, envoyer un seul `.tex` avec son `.bib`, donner l’identifiant arXiv d’un article déjà déposé, ou un PDF (seule sa liste de références imprimée est alors lue).

## 2. Lancez la vérification

Ouvrez la [vérification de bibliographie](/fr/refcheck/) (Premium) et déposez le `.zip` sous **Votre manuscrit** : inutile d’envoyer le `.bib` à part. Chaque entrée est identifiée avec Crossref, NASA ADS, OpenAlex, arXiv et Semantic Scholar, puis :

- les **articles rétractés, les expressions de préoccupation et les corrections** viennent de Retraction Watch et de Crossref ;
- les **prépublications publiées depuis** (arXiv, bioRxiv, medRxiv…) reçoivent leur entrée de revue ;
- les **entrées fausses** (année, auteurs, titre, DOI) et les **doublons** sont listés avec le BibTeX corrigé ;
- avec le LaTeX, les clés que votre texte cite mais que votre `.bib` n’a pas (elles s’impriment en `[?]`) et les entrées jamais citées sont listées ;
- une IA lit chaque phrase qui cite à côté des articles cités et signale celles que les articles ne semblent pas soutenir, avec le passage. Une citation signalée est à vérifier, ce n’est pas la preuve d’une erreur.

## 3. Ramenez le .bib corrigé dans Overleaf

Sous **.bib corrigé**, décochez les corrections que vous ne voulez pas, puis :

- **Téléchargez** le `.bib` et envoyez-le à la place de l’ancien dans votre projet Overleaf ; ou
- **Ouvrir dans Overleaf** : un nouveau projet imprime chaque entrée du `.bib` corrigé (`\nocite{*}`), pour voir qu’il compile et à quoi ressemble chaque référence avant de remplacer le fichier de votre manuscrit.

Marquez chaque problème comme corrigé ou voulu, et vérifiez à nouveau après correction : ce qui a été résolu est montré à part. Avec Premium, les références peuvent être **surveillées jusqu’à la publication** : vous recevez un email si une prépublication citée est publiée ou un article cité rétracté entre-temps.

## 4. Gardez un .bib synchronisé avec une collection

Pendant que vous écrivez, vous ajoutez des articles à une [collection](/fr/collections/) sur SummarizePaper. Son onglet **Réglages** donne une adresse pour son `.bib`. Dans votre projet Overleaf, choisissez **Depuis une URL externe** dans le menu d’envoi de fichier, collez l’adresse et nommez le fichier, par exemple `collection.bib`. Après avoir ajouté des articles à la collection, cliquez sur **Actualiser** sur ce fichier dans Overleaf, et citez-les avec `\bibliography{collection}`.

Les clés des articles déjà présents ne changent jamais quand vous en ajoutez d’autres : vos `\cite` continuent de fonctionner. L’adresse ne montre que les références, pas vos notes ni vos étiquettes, et vous pouvez l’arrêter à tout moment. La synchronisation Zotero et Mendeley d’Overleaf demande un abonnement Overleaf payant ; ce lien, non.

## 5. Ce qui s’ouvre aussi dans Overleaf

- **Revue de littérature** (Premium+) : la synthèse avec ses `\cite` et son `.bib`, en nouveau projet Overleaf prêt à compiler.
- **Qu’ai-je manqué ?** (Premium) : les articles proches que votre brouillon ne cite pas, en projet qui les imprime, pour copier les entrées voulues.
- **Journal club** (Premium+) : vos diapositives modifiées en projet **LaTeX Beamer**, avec les figures de l’article quand sa licence le permet, vos notes d’orateur et les passages de l’article dans `\note`. Il compile avec XeLaTeX, qu’Overleaf choisit pour vous.

Overleaf permet à un autre site de créer un projet, jamais d’écrire dans un projet existant : chacun de ces boutons ouvre un **nouveau** projet dans votre compte Overleaf, après connexion à Overleaf.
