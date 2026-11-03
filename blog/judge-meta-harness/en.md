This note collects two sets of experiments we ran on LLM-as-a-judge, both before we moved to Jev-style decision models (see the [previous post](../jevany-robotics/)).

The first is a systematic evaluation: on three rubric benchmarks — HealthBench meta-evaluation ([Arora et al., 2025](https://arxiv.org/abs/2505.08775)), ProfBench ([Wang et al., 2025](https://arxiv.org/abs/2510.18941)) and RuVerBench ([Peng et al., 2026](https://arxiv.org/abs/2606.29920)) — we ran 21 generative judges through one pipeline, then set them against the decision model Jev 1.13 ([TypeSafe](https://docs.typesafe.ai/introduction)), which returns option probabilities and generates no text. The second is a meta-harness: without touching model weights, starting from seeds found by hand iteration and earlier searches, a strong model rewrites the code around a 4B judge — its harness. The idea is close to Meta-Harness ([Lee et al., 2026](https://arxiv.org/abs/2603.28052)). On HealthBench meta-evaluation, Gemma-4-E4B goes from 61.1 test MF1 (bare) through 68.4 (seed) to 72.6.

# 1 Conclusions

1. **No model wins everywhere.** Qwen3.8-27B is best on HealthBench (MF1 68.59), Gemma4-31B on ProfBench (79.08), and on RuVerBench the DR and AC domains are led by Qwen3.8-27B and Gemma4-31B respectively. None of them reaches the 70.9 that GPT-4.1 scores in the HealthBench paper.
2. **The most consistent bias is toward passing.** On HealthBench and RuVerBench almost every model has a higher false-pass rate than false-reject rate (over the 18 models common to the two, 61.1% against 17.3% on HealthBench). ProfBench shows no such pattern.
3. **Parameter count is not sufficient.** Qwen3.5-4B reaches 69.65 on RuVerBench-DR, above every 7B–14B model; on ProfBench, Qwen3.8-27B sits below four 4B models.
4. **Jev 1.13 is within a point of the best generative judge.** It scores 67.74 on the full HealthBench (0.85 lower) and 79.76 on ProfBench (0.68 higher), at 0.4–0.5 s per request. It cannot do RuVerBench, which needs its own retrieval and a written rationale.
5. **A 4B judge with a searched harness beats every bare judge on the same test rows.** On 14,728 test rows, Gemma-4-E4B goes from 61.11 to 72.58, and negative-class F1 from 37.47 to 56.83. The best bare judge on those rows is 69.14; Jev is 68.18.
6. **The gain comes from reading more accurately, then calibrating on labels.** Scoring points distilled from physician annotations, plus positive and negative examples from the same rubric, make the model read better; a small logistic regression fitted on labels calibrates the log-odds of the verdict. But the labels carry a lot on their own: a constant verdict per rubric already gives 63.1, and a lexical classifier that never calls an LLM gives 66.6. So "4B beats 27B" should be read as "4B + 7,379 physician-labelled rows + harness" beats "a 27B bare judge that never sees labels".
7. **The acceptance rule matters.** A fixed dev subset turns noise into a gain (v3: +5.1 on dev, +0.7 on test). v4 separates the rows the proposer saw line by line from the rows drawn by gated sampling, and requires the lower bound of the paired interval on the full validation to stay above zero (along with several other setting changes). After that, the gains on validation carried over to test.
8. **Proposers differ a lot.** Under the same protocol and the same seeds, Claude Fable reaches +4.2 in 16 rounds and gpt-6-astra +1.8 in 60. Each was run once.

# 2 A systematic evaluation of generative judges

## 2.1 What we measured

We measure bare judging: given the material and one rubric item, decide satisfied or not satisfied, then compare with human labels. The three benchmarks:

<div class="text-table">

| Benchmark | What it judges | Size | Share of true negatives | Main metric |
| --- | --- | --- | --- | --- |
| HealthBench meta-evaluation | Whether a model answer satisfies one physician-written rubric item | 29,511 rows, 60,896 physician labels | 23.0% | MF1: the mean of positive and negative F1, counted per physician label |
| ProfBench | Whether a professional report satisfies an expert rubric | 3,486 pairs (40 tasks) | 49.7% | Macro-F1; we also report bias-corrected MF1-BI |
| RuVerBench | Whether a deep-research report (DR) or a coding-agent trace (AC) satisfies a rubric or checklist | 1,615 DR and 843 AC items | DR 60.9%, AC 17.2% | Mean balanced accuracy across categories (AvgBAcc), reported per domain |

</div>

**Models.** Twenty-one: the Qwen3, Qwen3.5 and Qwen3.8 families (0.6B–27B, including two uncensored 27B variants), Gemma-4 (E2B, E4B, 12B, 31B), GPT-OSS-20B, OLMo-3-7B, Qwen2.5-0.5B and Claude Sonnet 5. Claude Sonnet 5 was only run on HealthBench and Qwen2.5-0.5B was not run on it, so each benchmark has 20 models.

**Protocol.** HealthBench uses the official grader prompt from simple-evals ([OpenAI](https://github.com/openai/simple-evals)), temperature 0, structured JSON output. ProfBench uses the official prompt ([NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)) and answers only Yes or No. RuVerBench uses our own retrieval harness: the model searches the documents attached to the case, pages through them, and finally submits a verdict, a rationale and citations. Every local model that can toggle thinking has it off; the one exception is Gemma4-31B on HealthBench, which we reran with thinking on. Claude is called through <code>claude -p</code>, where we control neither temperature nor thinking, and about 40% of calls triggered extended thinking.

**Hardware.** All local models run under vLLM on A100-40GB cards; the 27B and 31B models use two cards (so does the HealthBench scoring job for GPT-OSS-20B).

The prompts, output budgets and tool protocols differ across the three benchmarks, so scores are not comparable across them. Everything below compares models within one benchmark.


## 2.2 Quality against cost: the Pareto frontier

On every benchmark the best judge needs two GPUs. In GPU cost that is 8 to 60 times a 4B model on the frontier (about 8× on HealthBench, 16× on RuVerBench, 60× on ProfBench); in the latency plotted here the gap is smaller — about 3–4× on HealthBench and RuVerBench, and 30× on ProfBench. What the extra cost buys depends on the benchmark: ProfBench is close to saturated at 4B, while HealthBench and RuVerBench still separate large models from small ones.

![Quality against latency on HealthBench meta-evaluation](img/figure-001.png)

***Figure 1** HealthBench meta-evaluation, 20 judges. The horizontal axis is concurrency-adjusted latency (wall clock × 32 / rows, log scale); the vertical axis is MF1 counted per physician label. Filled dots sit on the Pareto frontier, hollow ones are dominated (a faster and more accurate model exists); diamonds are API models with estimated latency; the dashed line at the top is 70.9, GPT-4.1 in the HealthBench paper.*

**HealthBench.** The frontier runs from Qwen3-0.6B (52.20 at 1.37 s) to Qwen3.8-27B (68.59 at 13.07 s). Qwen3.5-4B reaches 62.46 in 3.14 s, above Qwen3-8B (58.35) and Qwen3.5-9B (61.33). Claude Sonnet 5's 67.27 is dominated by Gemma4-12B (67.44 at 8.72 s), but its negative-class F1 (53.28) is the highest in the field. The top few are within a point of each other; we ran no significance test.

![Quality against latency on ProfBench](img/figure-002.png)

***Figure 2** ProfBench, 3,486 answer–criterion pairs. The bars are 95% intervals from resampling by task; the cross is GPT-OSS-20B, whose protocol is incompatible (it emits only Harmony control tokens, so 33.19 is not a reading of capability).*

**ProfBench.** The curve is almost flat past 0.3 s. Gemma4-E4B (75.82 at 0.32 s) and Gemma4-31B (79.08 at 9.32 s) differ by 3.3 points but roughly 30× in latency. Qwen3.8-27B scores only 72.19, below four 4B models.

![Quality against latency in the two RuVerBench domains](img/figure-003.png)

***Figure 3** The DR (1,615 rubric points) and AC (843 checklist items) domains of RuVerBench. The horizontal axis is measured mean time for one complete judgment, including all tool turns; the vertical axis is AvgBAcc; the dashed line is chance. A triangle marks a domain where more than 5% of judgments fell back to random because the protocol failed.*

**RuVerBench.** The gap between large and small models is widest here. On DR, Qwen3.8-27B scores 87.84 and Gemma4-31B 85.95; among 4B and smaller models only Qwen3.5-4B (69.65) and Gemma4-E4B (58.33) sit clearly above 50, and the rest are near chance. On AC, Gemma4-31B leads with 76.36 while Qwen3.8-27B has 66.85.

The horizontal axes are not the same quantity: HealthBench and ProfBench show concurrency-adjusted latency, RuVerBench shows measured time for one complete judgment, and the two should not be compared across panels. Claude runs on a subscription quota, so its latency is an estimate with rate-limit pauses removed.

## 2.3 Against the decision model Jev

Jev 1.13 ([TypeSafe](https://docs.typesafe.ai/introduction)) is a decision model: it returns options and probabilities and generates no text. On the two benchmarks where it applies it is within a point of the best generative judge (0.85 lower on HealthBench, 0.68 higher on ProfBench), at about 0.4–0.5 s per request.

| Judge | HealthBench MF1 (full) | ProfBench Macro-F1 | RuVerBench DR / AC | HealthBench time | ProfBench time |
| --- | --- | --- | --- | --- | --- |
| Jev 1.13 (API) | 67.74 | **79.76** | not applicable | 0.42 s (one request) | 0.46 s (one request) |
| Qwen3.8-27B (2 GPUs) | **68.59** | 72.19 | 87.84 / 66.85 | 13.07 s | 4.78 s |
| Gemma4-31B (2 GPUs) | 68.12 | 79.08 | 85.95 / 76.36 | 27.80 s (lower bound) | 9.32 s |
| Gemma4-12B | 67.44 | 77.97 | 61.28 / 59.27 | 8.72 s | 5.62 s |
| Claude Sonnet 5 (API) | 67.27 | – | – | about 11.9 s (estimate) | – |
| Gemma4-E4B | 61.62 | 75.82 | 58.33 / 58.88 | 5.39 s | 0.32 s |

Times for local models are concurrency-adjusted latency (Gemma4-31B on HealthBench is the thinking rerun, and its time is the sum of two partial jobs, so it is a lower bound); Jev's numbers are client-side single-request times including network, and on ProfBench 1.78 s once the 18 requests per second limit is folded in. Running all 29,511 HealthBench rows through Jev cost $1.42, and the 3,486 ProfBench rows cost $0.31.

A few details that belong together:

- **The interfaces are not the same.** On HealthBench, Jev goes through its native option-and-probability interface, with p ≥ 0.5 read as satisfied, and we did not reproduce the official grader prompt. On ProfBench we kept the official prompt text and scorer but switched to Jev's native option interface, adding descriptions for Yes and No, and read back the option it returned. Same data and same scoring function, but this is a result after interface adaptation, not one protocol.
- **RuVerBench is out of reach.** Our RuVerBench protocol asks the model to write its own search terms, notes, rationale and citations. Jev has no free-text output, so we record it as protocol-incompatible rather than zero.
- **Errors cluster in particular rubric items.** On HealthBench, Jev's positive and negative F1 are 84.63 / 50.85, with AUROC 0.75. On "complex answer — detail, accuracy and expression of uncertainty" (644 rows) it judges almost everything unsatisfied, with positive recall of only 5.9%; on communication and emergency referral it errs the other way, passing 63.6% and 58.7% of labels that physicians marked unsatisfied. It is not simply lenient or strict.
- **Probabilities can be calibrated, but that does not raise the score.** Raw probabilities run low: the label-weighted mean p is 65.6% while 77.0% of labels are true. Under five-fold cross-validation by conversation, moving the threshold adds only 0.08 points; Platt scaling drops ECE from 11.4% to 0.8% without changing the ranking, so MF1 does not improve.
- **A human reference.** Scoring each physician's labels against the other physicians, with the same weights used for model evaluation, gives MF1 around 69.22; Jev is 1.48 lower (conversation-resampled interval [0.69, 2.32]). Under a different aggregation this reference moves between 67.2 and 69.6.
- **Not every decision model behaves this way.** Another decision model, Laya, scores only 48.89 on the full HealthBench, close to chance.


## 2.4 Patterns across the benchmarks

The most common bias in this set is toward passing: on HealthBench and RuVerBench almost every model's false-pass rate (FPR, the share of true negatives judged satisfied) is higher than its false-reject rate (FNR), but the pattern does not hold on ProfBench. This compares conditional error rates; in absolute counts, AC — where negatives are only 17% — has more false rejects. The table covers the 18 local models shared by all three benchmarks.

| Benchmark | Models with FPR above FNR | Mean FPR | Mean FNR |
| --- | --- | --- | --- |
| HealthBench | 17 / 18 | 61.14% | 17.30% |
| ProfBench | 6 / 18 | 27.77% | 27.11% |
| RuVerBench-DR | 18 / 18 | 73.12% | 4.47% |
| RuVerBench-AC | 16 / 18 | 61.53% | 18.61% |

DR is 60.9% true negatives and all 18 models still lean toward passing, so this is not a case of "few negatives, so pass". Failures seen in row-by-row review include: treating an answer as satisfied as soon as it mentions the right topic; writing the rubric or background knowledge into the evidence as if the answer had said it; and getting the rationale right while flipping the boolean. These come from a small targeted sample and say nothing about the mix of error types.

A few more patterns. MF1 here is the mean of positive and negative F1 throughout, so the RuVerBench numbers differ from the AvgBAcc in the section 6 table.

1. **No model wins everywhere.** Gemma4-31B is first on ProfBench and AC, second on DR, fourth on HealthBench; Qwen3.8-27B is first on HealthBench and DR but only 72.19 on ProfBench. A good score does not transfer: Gemma4-12B has MF1 67.44 on HealthBench and 77.97 on ProfBench, but 50.47 on DR.
2. **Parameter count is not sufficient.** Qwen3.5-4B scores 64.86 on DR, above every 7B–14B model. Within the Qwen3 family, going from 4B to 14B adds 10.87 and 16.89 MF1 on HealthBench and DR, only 0.87 on ProfBench (interval includes 0), and minus 2.33 on AC (also includes 0; by AvgBAcc it is plus 4.73).
3. **Where large models win depends on the benchmark.** On HealthBench, the +6.50 from Gemma4-E4B to Gemma4-31B comes almost entirely from negative recall (+17.40, while positive recall falls 2.18); on DR the gain is +35.34, with negative recall +54.78. AC runs the other way: 236 of Qwen3.5-4B's 289 errors are false rejects. In the cases we reviewed, those rejects often come from missing later turns or misreading when the rule applies.
4. **Positive F1 alone hides failure.** Qwen3-1.7B on AC has F1+ of 90.29 and F1− of 1.32, passing 144 of 145 true negatives; Qwen3-4B on DR finds 3 of 984 negatives. Both classes have to be reported together.
5. **Results are sensitive to the protocol.** GPT-OSS-20B under ProfBench's single-token protocol emits only Harmony control tokens, with a format-valid rate of 0%, so 33.19 is not a reading of capability. On RuVerBench, four small models had more than 5% of judgments in at least one domain fall back to random because the protocol failed.
6. **Quality against cost.** On the full RuVerBench, Qwen3.5-4B uses about 0.92 GPU-hours and Gemma4-31B or Qwen3.8-27B about 14.5–15.0, roughly 16 times more.
7. **The labels themselves disagree.** In HealthBench, 22.5% of rows have physicians who disagree and 20.4% split exactly in half; ProfBench has items where the question and the answer do not line up yet the label is True. A score measures agreement with the given labels.

# 3 A meta-harness on HealthBench meta-evaluation

## 3.1 The question and the setup

Section 2 shows that small judges are cheap but not accurate enough. The question here is whether a 4B Gemma-4-E4B can reach large-model level on HealthBench meta-evaluation without touching weights — changing only the code around the model, its harness — and with a strong model doing that rewriting automatically.

**Why Gemma-4-E4B.** Under the full official protocol it scores 61.62, with negative F1 of 39.43 and a false-pass rate of 63.9%, at 0.17 GPU-seconds per row on one card. The strongest model, Qwen3.8-27B, is 68.59 and needs two cards at 0.82 GPU-seconds per row.

**What the harness can do.** Each row gives only the conversation, the last answer and one rubric item. The harness may organize the prompt, retrieve examples, call the model several times (up to six logical calls per row), read token probabilities, post-process with plain Python or numpy, and finally return met or not_met. It sees no labels, row ids or categories.

**Data split.** Grouped by whole conversation and stratified by category; the split looks at neither labels nor model outputs.

<div class="text-table">

| Split | Rows | Use | What the proposer sees |
| --- | --- | --- | --- |
| Calibration pool (25%) | 7,379 | the only source of any learnable asset, whether example bank or classifier; 2,000 rows are the training set, the rest feed gated dev sampling | all rows and labels; the training set also comes with row-level evaluation traces |
| Validation (25%) | 7,404 | the validation gate | only aggregate scores and paired intervals |
| test (50%) | 14,728 | scored once at the end | not evaluated or exported during the search; the Astra run did not isolate reads |

</div>

**Earlier work.** Between September 8 and 12 we iterated 42 versions by hand. Only one line held: show the model two answers to the same rubric item in the same conversation, one that physicians agreed passes and one that does not (65.85 → 67.15 on dev). Thinking mode, generic review, majority voting, many-shot prompting, GEPA-style prompt search ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)) and external medical retrieval never beat that consistently. We then moved to automatic search: the first run, v1, produced the guarded observer that shows up in the seeds below, and the lessons from v2 and v3 produced the v4 procedure.


## 3.2 How the automatic search runs

Each round a strong model, the proposer, reads the history of code and row-level traces and writes new candidate harnesses; a deterministic Python orchestrator checks them, evaluates on GPUs and gates them. Only a candidate that clears both paired gates becomes the new reference.

![The meta-harness search loop](img/figure-004.svg)

***Figure 4** One round of search. The proposer writes four candidates per round; the orchestrator runs static checks and training-set evaluation, then the dev gate and the validation gate in order, and only a candidate that clears both becomes the new reference. test is scored once, before the deadline.*

1. **Proposal.** Four proposals run in parallel each round, alternating between a local edit that starts from the current reference and a different mechanism altogether. The proposer may first try an uncertain idea on at most 64 rows (a probe) and submit the full candidate next round after seeing the traces.
2. **Checks.** An import allowlist, a ban on file- and process-level calls, a leakage scan (matching conversation ids and whole passages to see whether a candidate file contains validation or test content, which does not catch paraphrased or encoded leakage), and mock contract tests. Failures go back to the proposer to fix, at most twice.
3. **Training-set evaluation.** The full candidate runs on the 2,000-row training set, and row-level traces go back to the proposer for error analysis. This score does not decide acceptance.
4. **Dev gate.** Each round draws 1,000 rows from the dev pool, rotating by category (order fixed by hash, rows repeat across rounds, each row used about 11 times over 60 rounds). Candidate and reference are compared in pairs on the same rows: MF1 at least +0.3, and negative F1 not lower.
5. **Validation gate.** Paired row by row against the reference over all 7,404 rows, with 3,000 bootstrap resamples grouped by conversation: the 95% lower bound of the MF1 difference must be above zero, and negative F1 must not drop. Passing makes it the new reference.
6. **Final test.** Before the deadline, the two accepted candidates with the best validation scores and the three seeds are each scored once on test, reporting paired intervals against the seed reference.

Two proposers were used: gpt-6-astra (through <code>codex exec</code>, reasoning effort xhigh) and Claude Fable 5.1 (through <code>claude -p</code>, effort xhigh, with extra read limits). The evaluated model, Gemma-4-E4B, runs under vLLM on a single A100-40GB at temperature 0. Per-row budget: six logical calls, at most 2,048 output tokens each, 180 seconds of wall clock.

## 3.3 How much it improves

The best harness lifts Gemma-4-E4B from 61.11 to 72.58 test MF1, and negative F1 from 37.47 to 56.83. On the same 14,728 rows it beats every bare judge and Jev.

![MF1 on the same test rows](img/figure-005.svg)

***Figure 5** MF1 on the v4 test split (14,728 rows, 30,360 physician labels). Bare judges are read from their own full-set caches on the same rows (Claude is missing one row; Gemma-4-E4B's cached protocol differs slightly from the seed raw_nothink, so it reads 61.66 rather than 61.11), and Jev from its full per-row predictions. The dashed line is the best bare judge.*

| Candidate | Source | Validation MF1 (7,404 rows) | test MF1 | test F1− | Δtest against the seed reference [95% CI] | s / row |
| --- | --- | --- | --- | --- | --- | --- |
| raw_nothink: official prompt, one call | seed | 60.15 | 61.11 | 37.47 | −7.24 [−8.30, −6.18] | 2.90 |
| contrast_bank_v2: same-rubric positive and negative examples | seed | 66.60 | 67.47 | 48.86 | −0.88 [−1.24, −0.52] | 11.49 |
| prior_best_v1: retrieval plus guarded observer | seed reference | 67.12 | 68.35 | 49.66 | – | 11.64 |
| r15d_crossfit_negative_tail | Astra round 15 | 68.71 | 69.70 | 52.35 | +1.35 [+0.98, +1.74] | 11.65 |
| r29a_corroborated_edit_veto | Astra round 29 | 69.32 | 70.17 | 53.36 | +1.82 [+1.35, +2.32] | 12.12 |
| r06a_grading_notes_stack | Fable round 6 | 70.90 | 72.36 | 56.33 | +4.01 [+3.26, +4.73] | 16.79 |
| r07b_cheap_stack_violation_views | Fable round 7 | 71.48 | **72.58** | **56.83** | **+4.23 [+3.47, +4.98]** | 11.14 |

s/row is the mean row wall clock at concurrency 32. Intervals are paired bootstrap grouped by conversation, with no correction for multiple comparisons. Seed validation scores come from the Astra run; the seeds rerun inside the Fable run land within 0.2 of them, and the seed test results are shared by the two runs. The +4.54 Fable validation gain quoted in the text is measured against the seed reference rerun in its own run (66.94).

- **What the gain is made of.** The +7.24 from bare judging to the seed reference comes from the seeds themselves: same-rubric contrastive retrieval was built by hand (+6.36), and the guarded observer came out of the earlier v1 automatic search (+0.88). The v4 search adds +1.82 (Astra) or +4.23 (Fable) on top.
- **Rows where physicians agree** (11,383 rows). Bare 66.53, seed reference 76.94, Astra 78.59, Fable 82.62; Jev is 74.53 on these rows.
- **The ceiling.** Each row gets one boolean, yet it is compared against every physician label for that row, so MF1 cannot reach 100: taking the physician majority per row as the verdict, unsatisfied on a tie, gives 86.42 on these rows. Another reference is agreement among physicians, about 69.2 on the full set.
- **Cost.** r07b makes five calls per row, three of which only log and do not affect the verdict; at 11.14 s/row it matches the seed reference and is about 3.8 times bare judging. The time for keeping only the two decision calls was not measured.
- **An asymmetry worth keeping in mind.** The harness uses 7,379 physician-labelled rows from the calibration pool; the bare judges and Jev do not. As the next section shows, the labels alone are worth a lot of points.


## 3.4 Where the gain comes from

Taking Fable's best harness apart, most of the gain comes from two LLM views that read more accurately, plus label calibration on top of them; the lexical classifier adds only 0.4 points. The labels themselves carry a lot: a constant verdict per rubric alone is 2 points above bare judging. First, what each of the three routes did.

**Seeds.** Same-rubric contrastive retrieval, built by hand: for each row, up to six pairs of pass and fail answers from the calibration pool that share the same conversation and the same rubric and on which physicians agreed, placed in the prompt. That takes test from 61.11 to 67.47. Then the guarded observer from the v1 automatic search, routed by rubric: it may only flip in one direction, sees neither the initial verdict nor the rubric and examples while observing, and keeps the initial verdict when unsure. That brings 68.35.

**Astra (60 rounds) took the veto route.** r15d trains a TF-IDF logistic regression on the calibration pool with five-fold cross-fitting by conversation; when the reference says satisfied and the classifier scores at most 0.20, it flips the verdict to unsatisfied (test +1.35). r29a adds a minimal-edit counterfactual: the model makes one local edit to the answer, and if the edited version is satisfied while the original is not (checked in both orders) and the classifier agrees, the verdict is vetoed (test +0.47 more). These vetoes are not precise: on rows where physicians did not split evenly, the two veto types agree with the physician majority only 57% (108/188) and 54% (37/68) of the time, and MF1 rises mainly because negative recall goes from 43.5% to 49.8%.

**Fable (16 rounds) began with the same veto, then changed route.** Four acceptances:

1. r01a: the same lexical-prior veto (validation +2.13).
2. r03a: stop asking the model for an explanation and then a boolean; instead read the log-odds of the pass and fail tokens at the verdict position, and fit a logistic regression of about 32 parameters (log-odds, lexical prior, per-rubric intercept) on physician labels from the 2,000-row training set, with threshold 0.64.
3. r06a: the proposer read the physician annotations in the calibration pool and wrote a grading note for each of 30 rubrics (drafted in round 5 as a log only, used for verdicts from round 6): when to mark unsatisfied, and when something still counts as satisfied. One emergency-referral rubric, for example, says that vague advice such as "see a doctor if it gets worse", with no specific warning signs, is unsatisfied, while an incomplete red-flag list or an "I am not a doctor" disclaimer does not lose points. The notes enter as a new view with no examples.
4. r07b: keep only two views (grading notes; compact contrastive examples plus grading notes) and drop the 7k-token call with full examples, which is cheaper and more accurate.

The r07b row log keeps every component's score, so the harness can be decomposed layer by layer without new inference (Figure 6). Variants that needed refitting used only the 2,000-row training set. Refitting the full model under the same settings puts every coefficient within 0.003 of the original, with the threshold at 0.61 (0.64 originally) and 72.61 against 72.58 on test, so the fitting procedure reproduces.

![Layer-by-layer decomposition of r07b](img/figure-006.svg)

***Figure 6** MF1 of r07b's components on the same test rows, all recomputed from the row log with no new inference. Hollow dots make no LLM call at all; the dashed line is the best bare judge (69.14).*

- **Reading accurately is the bulk.** The grading-notes view alone — one call, no calibration — goes from 61.11 to 69.04, level with Qwen3.8-27B bare judging, which is 69.03 on the same rows.
- **Calibration adds another 2–3.** One view plus log-odds calibration (rubric intercept and lexical prior) takes grading notes from 69.04 to 71.92 and the compact contrastive view from 70.27 to 72.53. Both views calibrated without the lexical prior give 72.19; adding the prior buys only 0.39 (paired interval [0.04, 0.71]), and the compact contrastive view alone at 72.53 is indistinguishable from the full version (interval [−0.44, 0.33]).
- **The labels carry a lot of the score on their own.** A constant verdict per rubric gives 63.14, and the lexical classifier alone 66.60, both with no LLM call. Much of the MF1 on HealthBench meta-evaluation is therefore predictable from "what this rubric usually gets" and from the surface words of the answer, which echoes [Bagaria et al. (2026)](https://arxiv.org/abs/2609.02942) finding that rubric text alone predicts a judge's verdict above chance. A harness compared against a bare judge that never sees labels should report these two no-LLM baselines alongside.
- **What did not work.** Thinking mode (about 5.6 times the per-row time on Gemma, with unstable gains), cascades of different mechanisms, generic review (which pushes verdicts toward satisfied), various retrieval weights and example-bank variants, stacking rubric-specific veto and rescue observers (+0.2 to +0.5 each on fixed dev, about zero on held-out data), and ensembling the LLM judges themselves (on the Fable run's training set, intersecting reference and bare gives 66.4 and a majority over the three gives 67.4, both below the reference's own 67.9).

## 3.5 The acceptance rule: a fixed subset turns noise into a gain

After v4 changed the acceptance rule — and with it concurrency, proposals per round, prompts, and which rows the proposer sees — the gains on validation carried over to test. Figure 7 puts four searches side by side.

![Gains of four searches on the visible set, validation, and test](img/figure-007.svg)

***Figure 7** MF1 gain of each search's final reference against prior_best_v1 within the same run. Visible set: 600 fixed dev rows for v2 and v3, the 2,000-row training set for v4. Validation: a fixed 1,600-row gate sample for v2, the full 7,404 rows for the others. For v3 we take r36a, the last accepted version that has a full validation score.*

- **v2.** The proposer saw a fixed 600 dev rows and the gate was a fixed 1,600 rows where a point estimate only had to match or beat the reference. Against prior_best_v1, dev and the gate both rose by more than 2 points while test rose 0.88. The search actually started from contrast_bank_v2 (test +1.81 against it), and that seed was already 0.93 below prior_best_v1 on test.
- **v3.** The gate moved to a fresh 1,600 paired rows each round, but dev stayed at 600 fixed rows. By r36a the acceptance chain had gained 5.07 on dev, 0.90 on full validation and 0.70 on test (the last accepted, r44b, is +5.63 on dev with no full validation score and the same test). In hindsight, all six acceptance steps that have full validation scores have paired intervals crossing zero, and 34 of the 55 candidates that failed the dev gate missed it by less than 0.3.
- **v4.** The 2,000 rows whose traces the proposer sees do not overlap the rows drawn by the dev gate, though the dev pool's rows and labels are still visible and the two share 916 conversations; the gate rotates 1,000 rows per round, and validation requires the lower bound of the full paired interval to be above zero. Of 31 candidates that reached full validation only 2 were accepted, yet the +2.20 on validation turned into +1.82 on test. Fable, same rules: +4.54 on validation, +4.23 on test.
- **How large the noise is.** Even at temperature 0, changing the concurrency flips 1%–3% of verdicts, and MF1 moves by about ±1 point on 256 rows — the same order as the 0.3–0.5 dev gate. Acceptance decisions therefore have to rest on large samples compared pairwise on the same rows.

## 3.6 Proposers: same rules, +1.8 against +4.2

![Validation MF1 of the reference over rounds in two searches](img/figure-008.svg)

***Figure 8** Full validation MF1 of the accepted reference across rounds in the two v4 searches. Dots are accepted versions, the diamond is the final reference's test score; seed validation scores are the rerun values from each run.*

- **Astra.** 60 rounds, 233 candidates registered (91 of them probes), 31 reached full validation, 2 accepted; the 31 rounds after round 29 produced no further acceptance. It tried many directions, but both accepted changes add veto rules on top of the seeds.
- **Fable.** 16 rounds, 52 candidates, 19 reached full validation, 4 accepted, none after round 7; from round 13 most proposal calls failed (quota rejections are logged). From r03a on it stopped patching the seed's verdicts and instead read probabilities, calibrated on labels, and distilled physician annotations into prompts.
- **A caveat.** Each ran once. The Fable run started after Astra's results were in and added read limits, so this is not a clean one-variable comparison.


## 3.7 Swapping the judged model and the data

Swapping the judged model for the API model gpt-5.6-luna, the same procedure still works, but its final test never finished. Moving Fable's best harness to HealthBench Hard shows no visible gain against GPT-4.1's verdicts.

**Luna: works, unfinished.** The proposer is still Astra under the v4 rules. The final test needs five candidates across twelve shards each, sixty shards in all; forty-six finished (contrast_bank_v2 finished one shard and is left out of the table). The numbers below are summed directly from the finished shards; this is not a formal report and has no confidence intervals.

| Candidate | Validation MF1 (7,404 rows) | test, all 12 shards (14,728 rows) | test, the 9 shards where bare judging also finished (10,828 rows) |
| --- | --- | --- | --- |
| raw_nothink | 55.91 | not finished | 57.10 |
| prior_best_v1 (seed reference) | 60.78 | 62.32 | 61.59 |
| r03d_blind_anchor_full | 64.28 | 65.94 | 65.86 |
| r08a_stable_positive_boundary | 65.02 | 65.82 | 65.65 |

- **The bias points the other way from Gemma.** Luna bare judging is strict: positive F1 on those nine shards is only 66.53, and r08a lifts it to 79.39 while negative F1 moves from 47.67 to 51.91. The harness corrects whichever direction the model is off in.
- **The accepted mechanism is calibration on labels again.** Score the target answer blind alongside same-rubric positive and negative examples, then estimate a local boundary from the physicians' labels on those examples. Same family of idea as Fable's log-odds calibration.
- **Why it was slow and unfinished.** Luna is called through <code>codex exec</code>, so every judgment is its own subprocess, about 12 s per sequential call; concurrency was pinned at 10×3 by a timeout reason that had stopped applying after round 3, which put the final candidate at about 54 s per row. The bigger problem was idling: at the September 20 check, 69% of 69 hours were idle, including 31.8 hours after the orchestrator crashed on a parsing bug with nobody watching, and another 8.6 hours stuck on a rescheduling bug (section 4). The project was archived on September 22 with fourteen shards left, and there is no quota to finish them.

**HealthBench Hard: no collapse, no visible gain.** In October we ran three candidates from the Fable run over the 1,123 consensus criteria of HealthBench Hard. The labels there are GPT-4.1 grader verdicts, not physician annotations.

| Candidate | MF1 against GPT-4.1 verdicts | Negative F1 | s / row |
| --- | --- | --- | --- |
| raw_nothink | 79.95 | 67.99 | 3.3 |
| prior_best_v1 | 72.18 | 54.85 | 15.8 |
| r07b | 80.59 | 70.59 | 16.9 |

The harness was tuned against physician labels, and GPT-4.1's own verdicts agree with physicians only 70.9% of the time, so these numbers say that r07b does not fall apart out of distribution; they do not show a gain. The seed reference is 7.8 points below bare judging here. That may mean its examples and observer are tied to the distribution, but the labels and the bare run here come from the same official grader prompt, which favors bare judging, and we did not run contrast_bank_v2 separately, so we cannot tell whether the examples or the observer are responsible. A real transfer test needs new data with human labels.

# 4 Engineering scars

- **Measure throughput before guessing.** The first setup (CUDA graph off, concurrency 4 per card) used under 1% of the KV cache and ran 57–68 rows per minute per card. With CUDA graph on, max_num_seqs 32 and concurrency 16, it ran 144–152 rows per minute, bottlenecked by the roughly 8k-token prefill per row.
- **Temperature 0 is still not deterministic.** Changing concurrency flips 1%–3% of verdicts. This has been analyzed systematically elsewhere ([He & Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)); for us it means a gate on a small sample is itself noisy.
- **Unattended runs need defenses everywhere.** On NFS the "done" marker can appear before the data files do; the cluster's sbatch filter rejects scripts containing sleep or wait; some nodes failed to initialize vLLM. The most expensive one: a U+2028 inside a model-written explanation also breaks Python's <code>str.splitlines()</code>, which cut a JSON line in half and crashed the orchestrator for 31.8 hours; another bug, where a missing heartbeat file meant a job was never rescheduled, cost 8.6 more.
- **Compute.** The Astra v4 search used about 100 GPU-hours of evaluation jobs and about 204 GPU-hours of worker time, with 263 proposer calls at a median of about 10 minutes each.

# 5 Limits and next steps

## 5.1 What we cannot conclude yet

- **One benchmark, one fully finished judged model.** The harness search ran only on HealthBench meta-evaluation, with one run per configuration, so we have no estimate of search variance.
- **The test split is reused.** v2, v3, v4 and Fable all use the same 14,728 rows; the proposer's sandbox mainly restricts writes, not reads (the Fable run added read limits). The orchestrator neither evaluated nor exported test during the search, but that does not prove the proposer could not read it at the system level.
- **Validation is reused for acceptance decisions.** Astra alone ran 31 full validations, with no correction for multiple comparisons or adaptive selection.
- **It is not training-free.** Every good harness contains components trained on physician labels (TF-IDF classifier, logistic regression, grading notes), and looks up its coefficients by a hash of 30 known rubrics, so it does not carry over to unseen rubrics.
- **Higher MF1 is not better clinical judgment.** It comes with a trade-off in class recall, and 20% of rows split evenly among physicians.
- **No comparison against training a judge directly.** What the same 7,379 labelled rows would do under SFT, or under a linear probe, is unknown.

## 5.2 Next steps

1. **A training baseline:** LoRA SFT on Gemma-4-E4B from the same calibration pool, or a linear probe on its hidden states, against r07b.
2. **Across benchmarks:** move grading notes plus log-odds calibration to ProfBench and RuVerBench, both of which already have baselines for 20 models.
3. **Repeat the Fable search**, or swap the judged model (Qwen3.5-4B, for instance), to estimate the variance of the search itself.
4. **A counterfactual sensitivity test:** edit answers to satisfy or violate a rubric and see whether the verdicts of bare judging and of the harness flip with them.

# 6 Experimental details

**Full baseline results.** Official main metric for each benchmark, on a 0–100 scale. Bold marks the best generative judge in each column.

| Model | HealthBench MF1 | ProfBench Macro-F1 | ProfBench MF1-BI | RuVerBench-DR AvgBAcc | RuVerBench-AC AvgBAcc |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | – | 52.29 | 35.5 | 50.45 | 49.91 |
| Qwen3-0.6B | 52.20 | 36.40 | 32.9 | 49.78 | 50.34 |
| Qwen3.5-0.8B | 48.53 | 69.62 | 65.5 | 51.18 | 52.98 |
| Qwen3-1.7B | 51.93 | 61.09 | 58.3 | 49.92 | 49.84 |
| Qwen3.5-2B | 58.84 | 73.31 | 68.6 | 49.59 | 49.31 |
| Qwen3-4B | 53.51 | 75.16 | 69.1 | 50.09 | 52.27 |
| Qwen3-4B-Instruct-2507 | 57.75 | 75.79 | 70.9 | 51.69 | 54.76 |
| Qwen3.5-4B | 62.46 | 75.25 | 70.5 | 69.65 | 61.52 |
| Qwen3-8B | 58.35 | 75.80 | 73.1 | 50.85 | 58.13 |
| Qwen3.5-9B | 61.33 | 77.52 | 74.4 | 60.01 | 64.31 |
| Qwen3-14B | 64.38 | 76.02 | 72.1 | 57.27 | 57.00 |
| Qwen3.8-27B-abliterated | 68.49 | 73.27 | 72.6 | 81.88 | 69.85 |
| Qwen3.8-27B-uncensored | 68.21 | 74.41 | 73.4 | 85.66 | 71.91 |
| Qwen3.8-27B | **68.59** | 72.19 | 72.1 | **87.84** | 66.85 |
| Gemma4-E2B | 55.65 | 64.98 | 61.1 | 50.12 | 53.00 |
| Gemma4-E4B | 61.62 | 75.82 | 72.2 | 58.33 | 58.88 |
| Gemma4-12B | 67.44 | 77.97 | **76.3** | 61.28 | 59.27 |
| Gemma4-31B | 68.12 | **79.08** | 76.0 | 85.95 | **76.36** |
| OLMo-3-7B-Instruct | 54.83 | 69.97 | 69.3 | 49.94 | 48.15 |
| GPT-OSS-20B | 63.52 | 33.19 (protocol incompatible) | 25.6 | 77.71 | 64.26 |
| Claude Sonnet 5 | 67.27 | – | – | – | – |
| Jev 1.13 (interface adapted) | 67.74 | 79.76 | 78.3 | protocol incompatible | protocol incompatible |

On HealthBench, GPT-OSS-20B is missing one row and Claude excludes 32 hard refusals; everything else is full. The Gemma4-31B result on HealthBench is the thinking rerun. RuVerBench keeps the deterministic random fallback when the protocol fails.

**Code and versions.** HealthBench meta-evaluation uses simple-evals ([OpenAI](https://github.com/openai/simple-evals)) pinned at commit 652c89d0; ProfBench uses the official repository ([NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)) pinned at commit b06a29cd.

**Statistics.** Every interval in this note is a 95% paired bootstrap interval grouped by conversation (HealthBench) or by task (ProfBench), with no correction for multiple comparisons.

# References

1\. Arora, R. K., Wei, J., Hicks, R. S., et al. *HealthBench: Evaluating Large Language Models Towards Improved Human Health*. arXiv:2505.08775, 2025. [https://arxiv.org/abs/2505.08775](https://arxiv.org/abs/2505.08775)

2\. Wang, Z., Jung, J., Lu, X., et al. *ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge*. arXiv:2510.18941, 2025. [https://arxiv.org/abs/2510.18941](https://arxiv.org/abs/2510.18941)

3\. Peng, Y., Qi, Y., Xia, H., et al. *Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?* arXiv:2606.29920, 2026. Introduces RuVerBench. [https://arxiv.org/abs/2606.29920](https://arxiv.org/abs/2606.29920)

4\. TypeSafe. *Jev documentation*. 2026. [https://docs.typesafe.ai/introduction](https://docs.typesafe.ai/introduction)

5\. Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., & Finn, C. *Meta-Harness: End-to-End Optimization of Model Harnesses*. arXiv:2603.28052, 2026. [https://arxiv.org/abs/2603.28052](https://arxiv.org/abs/2603.28052)

6\. OpenAI. *simple-evals*. The official implementation of the HealthBench meta-evaluation. [https://github.com/openai/simple-evals](https://github.com/openai/simple-evals)

7\. NVIDIA. *ProfBench*. Data and the official scorer. [https://github.com/NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)

8\. Agrawal, L. A., Tan, S., Soylu, D., et al. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*. arXiv:2507.19457, 2025. [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)

9\. Bagaria, A., Sundaram, S. S., Krishnan, G. S., & Ravindran, B. *Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation*. arXiv:2609.02942, 2026. [https://arxiv.org/abs/2609.02942](https://arxiv.org/abs/2609.02942)

10\. He, H., & Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference*. Thinking Machines Lab: Connectionism, 2025. [https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
