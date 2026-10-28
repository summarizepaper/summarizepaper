---
title: 'Chat with papers: from the Sagittarius A* image to the evidence'
description: Analyze a Sagittarius A* figure, check its evidence and compare two full-text
  papers with real questions, follow-ups and a sourced table.
date: 2026-10-27
updated: 2026-10-28
draft: false
sources:
- label: EHT collaboration — Paper I (2022)
  url: https://arxiv.org/abs/2311.08680v2
- label: EHT collaboration — Paper VI (2022)
  url: https://arxiv.org/abs/2311.09484v1
- label: Creative Commons Attribution 4.0
  url: https://creativecommons.org/licenses/by/4.0/
---

Below the Sagittarius A&#42; image sit four smaller images that look similar but not identical. Are they successive moments in time, or different ways to reconstruct the same observations? That is a specific question to ask **Chat with papers**, then check against the source. This walkthrough starts with a figure, turns the answer into a journal-club explanation and brings in a second paper to examine what the image can establish about gravity.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/visual-en.jpg"><img src="/static/summarizer/img/guides/chat/visual-en.jpg" alt="The answer distinguishes robust features from uncertainties and displays Figure 3, the image actually supplied to the assistant. Both full text and the figure image are used here." width="1280" height="720" loading="lazy" /></a>
<figcaption>The answer distinguishes robust features from uncertainties and displays Figure 3, the image actually supplied to the assistant. Both full text and the figure image are used here.</figcaption>
</figure>


The examples use two EHT collaboration publications from 2022: **Paper I**, presenting the image and overall findings, and **Paper VI**, examining tests of the black hole metric. We use arXiv versions **2311.08680v2** and **2311.09484v1**. Both are licensed **CC BY 4.0**, and both were actually read in full during these demonstrations.

<span id="single-paper-chat"></span>

## Start with a paper whose full text is available

Already on a paper’s page? Ask your question in its chat box: the paper is already selected. The steps in this guide about figures, source passages and follow-up questions also apply there. The screenshots show the Chat page; comparing several papers and building tables require that page.

[Open the chat with Paper I](/en/chat/?ids=2311.08680v2). Alternatively, paste `2311.08680v2` into **“Add a paper directly”** and click **“Add”**. A DOI or a paper link also works.

Wait for **“Full text”** on the selected paper’s card before sending your question. Its version appears alongside it, so you can identify the same source later. When only an abstract is available, that limitation is indicated in the same place; this example uses the full text.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/full-text-publication-en.jpg"><img src="/static/summarizer/img/guides/chat/full-text-publication-en.jpg" alt="Paper I, published in 2022, is loaded in full. Its card distinguishes the publication year from the 2023 arXiv submission and gives the version, PDF and references." width="1280" height="720" loading="lazy" /></a>
<figcaption>Paper I, published in 2022, is loaded in full. Its card distinguishes the publication year from the 2023 arXiv submission and gives the version, PDF and references.</figcaption>
</figure>


## Ask about the figure, rather than requesting a general summary

Open **“Figures and plots”**, locate **Figure 3** and click **“Ask about this figure”**. The button prepares a reference to the figure and paper; complete it with what you want to understand. We ran this question for the guide:

> Visually analyze Figure 3 of paper 2311.08680v2. In at most 200 words, explain the large ring and the smaller images below it, then distinguish robust features from remaining uncertainties. State whether you actually received the image and support scientific conclusions with passages from the paper.

The answer identifies the smaller panels as **averages of reconstruction groups**, rather than successive frames in a movie. It separates the ring and dim center from less certain brightness details. The **“Figures supplied to the assistant for this answer”** block, visible in the opening screenshot, identifies the image actually used.

A visual observation, a measurement reported in the paper and a physical interpretation require different evidence. The answer’s source checks let you inspect those distinctions.

## Follow a claim to its source

In the English answer, click **[6]**, after the explanation that the smaller panels are not snapshots at different times. The chat takes you to the original caption, which states that distinction. The source is a passage you can read, rather than simply a bibliography entry.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/evidence-en.jpg"><img src="/static/summarizer/img/guides/chat/evidence-en.jpg" alt="Reference [6] opens the caption explaining why the smaller panels are not a movie. The source link lets you locate the passage on arXiv." width="1280" height="720" loading="lazy" /></a>
<figcaption>Reference [6] opens the caption explaining why the smaller panels are not a movie. The source link lets you locate the passage on arXiv.</figcaption>
</figure>


The English answer reports **5 of 8 claims found word for word** in the paper. The other three are marked as readings of the figure image. This count describes provenance, rather than serving as a scientific correctness score. Open the claims that matter to your argument and examine their context.

## Follow up with an explanation for your audience

Ask for a formulation suited to your task. To keep the exchange on the intended image, repeat the figure number and paper: a vague follow-up about “this figure” can lead the answer toward other illustrations.

> Still referring specifically to Figure 3 of paper 2311.08680v2, not Figure 2, give three sentences for a journal club: what the image reconstructs, why the smaller panels are not a movie, and one limitation. Use the Figure 3 caption.

The result is a three-sentence explanation that retains a concrete limitation: sparse baseline coverage leaves the brightness distribution around the ring poorly constrained. You have a short explanation to work with, alongside its sources.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/followup-en.jpg"><img src="/static/summarizer/img/guides/chat/followup-en.jpg" alt="The follow-up explicitly identifies Figure 3 and produces three sentences for a journal club, with the supplied image and source references." width="1280" height="720" loading="lazy" /></a>
<figcaption>The follow-up explicitly identifies Figure 3 and produces three sentences for a journal club, with the supplied image and source references.</figcaption>
</figure>


## Bring two papers into the conversation

[Open a comparison with Papers I and VI](/en/chat/?ids=2311.08680v2,2311.09484v1), or add `2311.09484v1` to your selection. Both cards show **“Full text”** here. The **[Collaboration 2022a]** and **[Collaboration 2022b]** tags distinguish the texts; 2022 is their journal publication year. The November 2023 arXiv submission dates are labeled separately on the cards. When no publication year is known, the chat falls back to the deposit year.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/two-papers-en.jpg"><img src="/static/summarizer/img/guides/chat/two-papers-en.jpg" alt="Both selected papers are available in full. Tags, versions and identifiers help keep their sources distinct." width="1280" height="720" loading="lazy" /></a>
<figcaption>Both selected papers are available in full. Tags, versions and identifiers help keep their sources distinct.</figcaption>
</figure>


Ask a comparison question that gives each paper a specific role:

> Compare papers I (2311.08680v2) and VI (2311.09484v1) in at most 180 words: what does the first measure, what does the second calibrate and test, and why is “the image looks like a ring” insufficient to conclude that all alternatives to Kerr are excluded? Attribute each conclusion to the right paper and cite supporting passages.

The answer distinguishes the **emission-ring size** studied in Paper I from the shadow inference and geometric tests developed in Paper VI. It attributes each part to its paper and points to supporting passages.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/comparison-en.jpg"><img src="/static/summarizer/img/guides/chat/comparison-en.jpg" alt="The multi-paper answer separates ring measurement, calibration to shadow size and the scope of the Kerr test, with references to the relevant texts." width="1280" height="720" loading="lazy" /></a>
<figcaption>The multi-paper answer separates ring measurement, calibration to shadow size and the scope of the Kerr test, with references to the relevant texts.</figcaption>
</figure>


These publications analyze the same observing campaign. Comparing them clarifies their contributions; it does not turn them into independent experiments. Likewise, different reported numbers need not disagree if one describes a bright emission ring and another a geometric shadow.

## Build a table around the quantities you need

With **Premium**, click **“Comparison table”** and replace the default columns with your criteria, one per line. For this case, we chose:

```text
Role of the paper in the analysis of Sagittarius A*
Ring or shadow size and definition of the quantity
Scope and limitations of the Kerr test
```

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/table-setup-en.jpg"><img src="/static/summarizer/img/guides/chat/table-setup-en.jpg" alt="Three custom columns target the concepts being compared. The builder shows the cost: one question per paper." width="1280" height="720" loading="lazy" /></a>
<figcaption>Three custom columns target the concepts being compared. The builder shows the cost: one question per paper.</figcaption>
</figure>


Click **“Build the table”**. The two resulting rows organize the papers’ roles, reported quantities and limitations. All six filled cells in this example have a supporting passage found in their source text.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/table-result-en.jpg"><img src="/static/summarizer/img/guides/chat/table-result-en.jpg" alt="The completed table starts with Paper I, organizing its findings by the chosen criteria and supplying evidence for each cell." width="1280" height="720" loading="lazy" /></a>
<figcaption>The completed table starts with Paper I, organizing its findings by the chosen criteria and supplying evidence for each cell.</figcaption>
</figure>


Expand a cell’s supporting passage. Below, the Paper VI row lets you inspect the source accompanying its inferred shadow sizes. The table becomes a way back into the papers, keeping values connected to their definitions.

<figure class="guide-screenshot">
<a href="/static/summarizer/img/guides/chat/table-source-en.jpg"><img src="/static/summarizer/img/guides/chat/table-source-en.jpg" alt="An expanded cell from Paper VI: its source passage accompanies the value and identifies the quantity actually being compared." width="1280" height="720" loading="lazy" /></a>
<figcaption>An expanded cell from Paper VI: its source passage accompanies the value and identifies the quantity actually being compared.</figcaption>
</figure>


## Keep a discussion you can check later

Under an answer, **“Copy”** includes the text and its sources. **“Export the conversation”** preserves questions, answers and associated passages. The table also provides **CSV** and **Markdown** exports. Treat the export as working notes and open the original sources before using a conclusion in scientific writing.

To repeat this workflow, start with one figure and a specific question. Add a complementary paper when the question calls for comparison. If you want to orient yourself first, the [summaries guide](../how-to-summarize-a-research-paper/) shows how to read the paper’s essentials and limitations.

<div class="cta-box">
<p>From the Sagittarius A&#42; image to tests of gravity: ask your own questions of both papers.</p>
<a class="btn btn-lg" data-funnel="guide_tool_how_to_chat_with_research_papers" href="/en/chat/?ids=2311.08680v2,2311.09484v1">Chat with both papers</a>
</div>

*Demonstrations executed in English on 28 October 2026 using a Premium+ account. Both papers were read in full; Figure 3 was actually supplied for visual analysis. Conversations remain private: the guide links only open the selected papers. Figure shown in the screenshots: EHT collaboration et al. (2022), [Paper I](https://arxiv.org/html/2311.08680v2), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), displayed at reduced size in the interface.*
