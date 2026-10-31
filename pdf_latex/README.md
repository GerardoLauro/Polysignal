# PolySignal — Technical Report (LaTeX)

This folder contains the full technical report for the *PolySignal* project
(Big Data Engineering 2025/2026, Social Network Analysis — Track 2).

## Files

| File | Contents |
| --- | --- |
| `main.tex` | Driver: preamble, title page, abstract, table of contents, and `\input` of the four parts. |
| `part1_foundations.tex` | Ch. 1–3 — Introduction, Background & Related Work, Data Model & Dataset Construction. |
| `part2_pipeline.tex` | Ch. 4–7 — Architecture & Storage, Data Ingestion, Contract–Post Linking, NLP Enrichment (Task 1). |
| `part3_analysis.tex` | Ch. 8–12 — Descriptive Analysis, Discourse Analysis, Signal–Market Correlation, Cross-Platform Comparison, Dashboard (Task 2). |
| `part4_prediction.tex` | Ch. 13–16 — Outcome Prediction, Discussion, Limitations & Future Work, Conclusions (Task 3). |
| `references.bib` | Bibliography (BibTeX). |
| `figures/` | All plots and diagrams (20 vector PDFs). |
| `main.pdf` | Pre-compiled output (55 pages). |

All figures were generated from the **real** project datasets
(`data/normalized/…` and `data/analytical/…`) with matplotlib; no synthetic
data is used anywhere in the report.

## Snapshot & regeneration (the corpus is still growing)

Every number and figure in the report refers to a **fixed data snapshot**. Because
social collection is longitudinal and may still be running, the report is fully
reproducible from the current artifacts:

```bash
# 1. (optional) re-run the pipeline so the artifacts reflect the latest corpus
#    e.g. polysignal nlp enrich · linking score-candidates · analytics run · evaluate task3
# 2. refresh every figure PDF in figures/ from the current data
.venv/Scripts/python.exe scripts/_paper_figs.py
# 3. recompile (see below)
```

`scripts/_paper_figs.py` is read-only over `data/` and writes only into `figures/`.
Re-running it after the multi-platform scrape integrates will repopulate the
cross-platform chapter (Ch. 11) and update all counts; the qualitative findings —
especially the predictive null result — are stable across snapshots.

> **Toolchain:** a self-contained [Tectonic](https://tectonic-typesetting.github.io/)
> binary is installed at `.tools/tectonic/tectonic.exe` (gitignored). Rebuild the PDF with:
>
> ```bash
> (cd pdf_latex && ../.tools/tectonic/tectonic.exe main.tex)
> ```
>
> Tectonic runs the full multi-pass + BibTeX cycle automatically. The current `main.pdf`
> (55 pages) was produced this way and is fully resolved (20 figures, 0 undefined
> references, 0 undefined citations). Overleaf also works — upload the whole `pdf_latex/`
> folder and press *Recompile*.

## How to compile

The report uses only standard packages (`pdflatex` + `bibtex`).

**Command line:**

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

or, more simply, `latexmk -pdf main.tex`.

**Overleaf / VS Code (LaTeX Workshop):** open `main.tex` and build with the
default `pdfLaTeX + bibtex` recipe.

It also compiles with [Tectonic](https://tectonic-typesetting.github.io/):
`tectonic main.tex`.
