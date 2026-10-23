# Huanxin Sheng — Personal Website

Source code of my academic homepage and research blog, built as a static site and served via GitHub Pages.

**Live site:** https://brucesheng1202.github.io/index.html

---

## About

I am a PhD student in Statistics & Data Science at **MBZUAI**, advised by [Dr. Jian Kang](https://jiank2.github.io/). Before that I was a CS PhD student at the **University of Rochester** (2024–2025), where I mastered out and received my MS in Computer Science, advised by Dr. Jian Kang and [Dr. Hangfeng He](https://hornhehhf.github.io/).

My research lies in **trustworthy AI systems and model evaluation**. I am currently exploring **agent evaluation** and **on-policy distillation**.

## Research Highlights

- **EMNLP 2025 SAC Highlight (top 0.6%)** — *Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction*, my first paper, also presented at the NeurIPS 2025 LLM Evaluation Workshop.
- **ECCV 2026** — *Video-Based Reward Modeling for Computer-Use Agents* (collaboration).
- **JASA** — high-dimensional portfolio analysis, from a remote research assistantship at CUHK.
- Research Intern at **Ant Group (AQ team)**, 2026.

## Blog

Two long-form technical notes, each available in both English and Chinese:

| Date | Post | Topic |
| --- | --- | --- |
| Oct 2026 | [Accounting for Compute in a Looped Model](https://brucesheng1202.github.io/blog/loop-compute-ledger.html?lang=en) | Recurrent language models — the Parcae recipe averages eight loops, but the first four already collect 94% of the gain. Trained 60+ small models on a single RTX 5090 to find out how much of that compute is actually necessary. |
| Jul 2026 | [On-Policy Distillation: Discussion of Loss Design](https://brucesheng1202.github.io/blog/opd-loss-design.html?lang=en) | A practical and empirical guide to choosing an OPD objective: reverse KL, better estimators, Top-K truncation, and why representation distillation still struggles on-policy. |

## Site Structure

```
.
├── index.html              # Homepage: about, news, publications, services, misc
├── blog/
│   ├── index.html          # Blog listing
│   ├── loop-compute-ledger.html
│   ├── opd-loss-design.html
│   ├── loop-i18n.js        # Bilingual (EN / 中文) switching for the loop post
│   ├── style.css           # Blog styles
│   └── posts/              # Post source content (Markdown, JS)
├── img/                    # Profile photo and site images
└── README.md
```

The homepage is plain HTML + Bootstrap 4; the blog uses a lightweight custom stylesheet with client-side i18n and MathJax for formulas. Everything is static — no build step.

## Links

[Google Scholar](https://scholar.google.com/citations?user=oTxPsY0AAAAJ) · [GitHub](https://github.com/BruceSheng1202) · [LinkedIn](https://www.linkedin.com/in/huanxin-sheng-632497326/) · [X](https://x.com/HuanxinShe5254) · huanxin [dot] sheng [at] mbzuai [dot] ac [dot] ae
