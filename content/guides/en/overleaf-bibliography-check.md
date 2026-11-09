---
title: 'Check the bibliography of an Overleaf project, and keep it up to date'
description: 'Find retracted papers, preprints published since and wrong entries in your Overleaf .bib, reopen it corrected in Overleaf, and keep it in sync.'
date: 2026-11-09
draft: false
faq:
  - q: "Do I need a paid Overleaf plan?"
    a: "No. The check reads the .zip that any Overleaf project downloads. Opening a project in Overleaf and linking a file from an external URL do not use Overleaf's paid integrations (Zotero, Mendeley, Git)."
  - q: "Does SummarizePaper keep my manuscript?"
    a: "No. The archive is read in memory for the check: only its .tex, .bib and .bbl files, never extracted on disk. The report keeps what it shows: the problems found, the corrected entries and the sentences of flagged citations."
  - q: "Can SummarizePaper write into my existing Overleaf project?"
    a: "Overleaf offers no way for another site to write into a project. 'Open in Overleaf' creates a new project; to update your own, replace its .bib, or link a .bib from one of your collections, which Overleaf fetches again when you click Refresh."
sources:
  - label: Overleaf — Adding a file from a URL
    url: https://docs.overleaf.com/managing-projects-and-files/adding-files-to-a-project/adding-a-file-from-a-url
  - label: Overleaf — Open a project from another site (developers)
    url: https://www.overleaf.com/devs
---

You are about to submit, and your paper was written on Overleaf. Its `.bib` grew over two years: some preprints have since been published, one cited paper may have been retracted, and a few entries were copied with the wrong year or a truncated author list. **The bibliography check reads your whole Overleaf project and tells you which references to fix**, and what each citation is used for.

## 1. Download your project from Overleaf

In Overleaf, open the menu and download the project's **source** as a `.zip`. Nothing needs to be cleaned up: the check reads only the `.tex`, `.bib` and `.bbl` files it contains, finds the main file (the one with `\begin{document}`) and follows its `\input` and `\include`.

You can also paste your LaTeX, upload a single `.tex` with its `.bib`, give the arXiv identifier of a paper already posted, or a PDF (then only its printed reference list is read).

## 2. Run the check

Open the [bibliography check](/en/refcheck/) (Premium) and drop the `.zip` under **Your manuscript**: no need to send the `.bib` separately. Each entry is identified with Crossref, NASA ADS, OpenAlex, arXiv and Semantic Scholar, then:

- **retracted papers, expressions of concern and corrections** come from Retraction Watch and Crossref;
- **preprints published since** (arXiv, bioRxiv, medRxiv…) get their journal entry;
- **wrong entries** (year, authors, title, DOI) and **duplicates** are listed with the corrected BibTeX;
- with the LaTeX, the keys your text cites but your `.bib` lacks (they print as `[?]`) and the entries never cited are listed;
- an AI reads each sentence with citations next to the papers it cites and flags those the papers do not seem to support, with the passage. A flagged citation is something to check, not a proof of error.

## 3. Bring the corrected .bib back into Overleaf

Under **Corrected .bib**, untick the corrections you do not want, then:

- **Download** the `.bib` and upload it in place of the old one in your Overleaf project; or
- **Open in Overleaf**: a new project prints every entry of the corrected `.bib` (`\nocite{*}`), so you see that it compiles and how each reference looks before you replace the file in your manuscript.

Mark each problem as fixed or intended, and check again after correcting: what was solved is shown apart. With Premium, the references can be **watched until publication**: you get an email if a cited preprint is published or a cited paper retracted in the meantime.

## 4. Keep a .bib in sync with a collection

While you write, you add papers to a [collection](/en/collections/) on SummarizePaper. Its **Settings** tab gives an address for its `.bib`. In your Overleaf project, choose **From External URL** in the file upload menu, paste the address and name the file, for example `collection.bib`. After adding papers to the collection, click **Refresh** on that file in Overleaf, and cite them with `\bibliography{collection}`.

The keys of the papers already there never change when you add others, so your `\cite` commands keep working. The address shows only the references, not your notes or tags, and you can stop it at any time. Overleaf's own Zotero and Mendeley sync needs a paid Overleaf plan; this link does not.

## 5. Other things that open in Overleaf

- **Literature review** (Premium+): the synthesis with its `\cite` commands and its `.bib`, as a new Overleaf project ready to compile.
- **What did I miss?** (Premium): the close papers your draft does not cite, as a project that prints them, to copy the entries you want.
- **Journal club** (Premium+): your edited slides as a **LaTeX Beamer** project, with the figures of the paper when its license allows it, your speaker notes and the passages of the paper in `\note`. It compiles with XeLaTeX, which Overleaf selects for you.

Overleaf lets another site create a project, never write into an existing one: each of these opens a **new** project in your Overleaf account, after you sign in to Overleaf.
