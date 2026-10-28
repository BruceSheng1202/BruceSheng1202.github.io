window.POST_CONTENT = {
  en: String.raw`This note collects two sets of experiments we ran on LLM-as-a-judge, both before we moved to Jev-style decision models (see the [previous post](../jevany-robotics/)).

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

10\. He, H., & Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference*. Thinking Machines Lab: Connectionism, 2025. [https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)`,
  zh: String.raw`这篇整理我们在 LLM-as-a-judge 上做的两组实验，都做在我们转向 Jev 式决策模型（见[上一篇](../jevany-robotics/)）之前。

第一组是系统测评：在 HealthBench 元评测（[Arora et al., 2025](https://arxiv.org/abs/2505.08775)）、ProfBench（[Wang et al., 2025](https://arxiv.org/abs/2510.18941)）和 RuVerBench（[Peng et al., 2026](https://arxiv.org/abs/2606.29920)）三个 rubric 评测上，用同一套流程测了 21 个生成式 judge，再和只输出选项概率、不生成文本的决策模型 Jev 1.13（[TypeSafe](https://docs.typesafe.ai/introduction)）对比。第二组是 meta-harness：不改模型权重，从手工迭代和早期搜索得到的种子出发，让强模型自动改写一个 4B 小 judge 的外围代码（harness），思路与 Meta-Harness（[Lee et al., 2026](https://arxiv.org/abs/2603.28052)）相近。在 HealthBench 元评测上，Gemma-4-E4B 的 test MF1 从 61.1（裸判）经 68.4（种子）提到 72.6。

# 1 结论

1. **没有全能冠军。** HealthBench 上最好的是 Qwen3.8-27B（MF1 68.59），ProfBench 上是 Gemma4-31B（79.08），RuVerBench 的 DR 和 AC 两个域分别由 Qwen3.8-27B 和 Gemma4-31B 领先。没有一个模型达到 HealthBench 论文里 GPT-4.1 的 70.9。
2. **最普遍的偏向是放行。** HealthBench 和 RuVerBench 上，几乎所有模型的错误放行率都高于错误拒绝率（HealthBench 上 18 个共有模型平均 61.1% 对 17.3%）。ProfBench 没有这个规律。
3. **参数量不是充分条件。** Qwen3.5-4B 在 RuVerBench-DR 上拿到 69.65，高于所有 7B–14B 模型；Qwen3.8-27B 在 ProfBench 上却低于四个 4B 模型。
4. **决策模型 Jev 1.13 和最好的生成式 judge 只差不到 1 分。** 它在 HealthBench 全量上是 67.74（低 0.85），ProfBench 上是 79.76（高 0.68），单次请求只要 0.4–0.5 秒；但它做不了需要自己检索、写理由的 RuVerBench。
5. **4B judge 加上自动搜出来的 harness，在同一批 test 行上超过所有裸判模型。** 14,728 行 test 上，Gemma-4-E4B 从 61.11 升到 72.58，负类 F1 从 37.47 升到 56.83；同一批行上最好的裸判模型是 69.14，Jev 是 68.18。
6. **提升主要来自让模型读得更准，再在标签上校准。** 由医生标注提炼的“评分要点”和同 rubric 正反例负责读得准，一个在标签上拟合的小逻辑回归负责校准判决的 log-odds。但标签本身信息量很大：只按 rubric 给常数判决就有 63.1，只用词汇分类器、完全不调用 LLM 有 66.6。所以“4B 超过 27B”应理解为“4B + 7,379 行医生标注 + harness”超过“不看标签的 27B 裸判”。
7. **验收规则很关键。** 固定 dev 子集会把噪声当成增益（v3 在 dev 上 +5.1，test 只 +0.7）。v4 把提案者拿到逐行痕迹的行和门控抽样的行分开，并要求全量验证的配对区间下界大于 0（同时也改了其他几项设置），之后验证集上的增益在 test 上兑现了。
8. **提案者差别很大。** 同一协议、同一组种子下，Claude Fable 用 16 轮拿到 +4.2，gpt-6-astra 用 60 轮拿到 +1.8。两者各只跑了一次。

# 2 生成式 judge 的系统测评

## 2.1 测了什么

我们测的是裸判能力：给定材料和一条评分标准，判断“满足 / 不满足”，再和人工标签比。三个评测如下。

<div class="text-table">

| 评测 | 判断什么 | 规模 | 真实负类占比 | 主指标 |
| --- | --- | --- | --- | --- |
| HealthBench 元评测 | 模型回答是否满足医生写的一条 rubric | 29,511 行，60,896 个医生标签 | 23.0% | MF1：正、负类 F1 的平均，逐个医生标签计 |
| ProfBench | 专业报告是否满足专家 rubric | 3,486 对（40 个任务） | 49.7% | Macro-F1；另报偏差校正后的 MF1-BI |
| RuVerBench | 深度研究报告（DR）或编码 agent 轨迹（AC）是否满足 rubric / checklist | DR 1,615 条，AC 843 条 | DR 60.9%，AC 17.2% | 各类别平衡准确率的平均（AvgBAcc），两个域分开报 |

</div>

**模型。** 21 个：Qwen3 / Qwen3.5 / Qwen3.8 系列（0.6B–27B，含两个去审查的 27B 变体）、Gemma-4（E2B、E4B、12B、31B）、GPT-OSS-20B、OLMo-3-7B、Qwen2.5-0.5B，以及 Claude Sonnet 5。Claude Sonnet 5 只测了 HealthBench，Qwen2.5-0.5B 没测 HealthBench，所以每个评测各 20 个模型。

**协议。** HealthBench 用 simple-evals（[OpenAI](https://github.com/openai/simple-evals)）的官方 grader prompt，temperature 0，结构化 JSON 输出；ProfBench 用官方 prompt（[NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)），只答 Yes / No；RuVerBench 用我们自己的检索式 harness，模型在这个 case 自己的文档里搜索、翻页，最后提交判决、理由和引文。支持开关的本地模型都关了思考模式，只有 Gemma4-31B 的 HealthBench 结果是开启思考重跑的；Claude 经 <code>claude -p</code> 调用，temperature 和思考都控制不了，约 40% 的调用触发了 extended thinking。

**硬件。** 本地模型全部用 vLLM 跑在 A100-40GB 上，27B 和 31B 用两张卡（GPT-OSS-20B 的 HealthBench 计分作业也用了两张）。

三个评测的提示、输出预算和工具协议都不同，分数不能跨评测比较难度。下文都在单个评测内部做比较。

## 2.2 质量与成本：Pareto 前沿

每个评测上最好的 judge 都要用两张卡。按 GPU 成本算，它是前沿上 4B 模型的 8 到 60 倍（HealthBench 约 8 倍，RuVerBench 约 16 倍，ProfBench 约 60 倍）；按图中的延迟算差距小一些，HealthBench 和 RuVerBench 上约 3–4 倍，ProfBench 上约 30 倍。多花的成本换来多少分因评测而异：ProfBench 上 4B 级别已经接近饱和，HealthBench 和 RuVerBench 上大模型仍明显领先。

![HealthBench 元评测的质量与延迟前沿](img/figure-001.png)

***图 1** HealthBench 元评测，20 个 judge。横轴是并发折算延迟（墙钟 × 32 / 条数，对数坐标），纵轴是逐医生标签计的 MF1。实心点在 Pareto 前沿上，空心点被前沿支配（存在更快且更准的模型）；菱形是 API 模型，延迟为估计值；顶部虚线是 HealthBench 论文里 GPT-4.1 的 70.9。*

**HealthBench。** 前沿从 Qwen3-0.6B（52.20，1.37 秒）延伸到 Qwen3.8-27B（68.59，13.07 秒）。Qwen3.5-4B 用 3.14 秒拿到 62.46，高于 Qwen3-8B（58.35）和 Qwen3.5-9B（61.33）。Claude Sonnet 5 的 67.27 被 Gemma4-12B（67.44，8.72 秒）支配，但它的负类 F1（53.28）是全场最高。前几名差距不到 1 分，没有做显著性检验。

![ProfBench 的质量与延迟前沿](img/figure-002.png)

***图 2** ProfBench，3,486 个回答–标准对。竖条是按任务重采样的 95% 区间；叉号是协议不兼容的 GPT-OSS-20B（只输出 Harmony 控制符，33.19 不代表能力）。*

**ProfBench。** 曲线过了 0.3 秒几乎变平。Gemma4-E4B（75.82，0.32 秒）到 Gemma4-31B（79.08，9.32 秒）只差 3.3 分，延迟却差约 30 倍。Qwen3.8-27B 只有 72.19，低于四个 4B 模型。

![RuVerBench 两个域的质量与延迟前沿](img/figure-003.png)

***图 3** RuVerBench 的 DR（1,615 个 rubric 点）和 AC（843 个 checklist 项）。横轴是一次完整判断（含全部工具轮次）的实测平均耗时，纵轴是 AvgBAcc，虚线是随机水平；三角表示该域有超过 5% 的判断因协议失败触发了随机兜底。*

**RuVerBench。** 大小模型的差距在这里最大。DR 上 Qwen3.8-27B 87.84、Gemma4-31B 85.95；4B 及以下的模型里只有 Qwen3.5-4B（69.65）和 Gemma4-E4B（58.33）明显高于 50，其余都在随机水平附近。AC 上 Gemma4-31B（76.36）领先，Qwen3.8-27B 只有 66.85。

横轴口径要分开看：HealthBench 和 ProfBench 是并发折算延迟，RuVerBench 是一次完整判断的实测平均耗时，两者不能混排。Claude 走订阅配额，延迟是去掉限流停顿后的估计。

## 2.3 与决策模型 Jev 的对比

Jev 1.13（[TypeSafe](https://docs.typesafe.ai/introduction)）是一个决策模型：只返回选项和概率，不生成文本。在能套用的两个评测上，它和最好的生成式 judge 相差不到 1 分（HealthBench 低 0.85，ProfBench 高 0.68），而单次请求只要约 0.4–0.5 秒。

| Judge | HealthBench MF1（全量） | ProfBench Macro-F1 | RuVerBench DR / AC | HealthBench 耗时 | ProfBench 耗时 |
| --- | --- | --- | --- | --- | --- |
| Jev 1.13（API） | 67.74 | **79.76** | 无法评测 | 0.42 秒（单次请求） | 0.46 秒（单次请求） |
| Qwen3.8-27B（2 卡） | **68.59** | 72.19 | 87.84 / 66.85 | 13.07 秒 | 4.78 秒 |
| Gemma4-31B（2 卡） | 68.12 | 79.08 | 85.95 / 76.36 | 27.80 秒（下界） | 9.32 秒 |
| Gemma4-12B | 67.44 | 77.97 | 61.28 / 59.27 | 8.72 秒 | 5.62 秒 |
| Claude Sonnet 5（API） | 67.27 | – | – | 约 11.9 秒（估计） | – |
| Gemma4-E4B | 61.62 | 75.82 | 58.33 / 58.88 | 5.39 秒 | 0.32 秒 |

本地模型的耗时是并发折算延迟（Gemma4-31B 的 HealthBench 是开思考的重跑，耗时由两段部分作业相加，是下界）；Jev 是客户端测到的单次请求耗时，含网络，ProfBench 上套上 18 次/秒的限速后折算为 1.78 秒。Jev 跑完 HealthBench 全量 29,511 条花了 1.42 美元，ProfBench 3,486 条花了 0.31 美元。

几个需要一起看的细节：

- **口径不同。** HealthBench 上 Jev 走原生的选项 / 概率接口，p ≥ 0.5 判满足，没有复现官方 grader prompt；ProfBench 上保留了官方提示文字和计分器，但改用 Jev 原生的选项接口（加了 Yes / No 两个选项描述），直接取它返回的选项。数据和计分函数相同，但这是接口适配后的结果，不是同一协议。
- **RuVerBench 做不了。** 我们的 RuVerBench 协议要求模型自己生成搜索词、笔记、理由和引文，Jev 的接口没有自由文本输出，所以记为协议不兼容，不是 0 分。
- **错误集中在特定 rubric。** HealthBench 上 Jev 的正、负类 F1 是 84.63 / 50.85，AUROC 0.75。“复杂回答—详细准确性与不确定性表达”这一类（644 条）它几乎全判不满足，正类召回只有 5.9%；沟通类和急诊转诊类则反过来放行偏多，医生判不满足的标签分别有 63.6% 和 58.7% 被它判成满足。所以不能简单说它偏宽或偏严。
- **概率可以校准，但不提分。** 原始概率偏低：按标签加权的平均 p 为 65.6%，实际有 77.0% 的标签为真。按对话五折交叉验证，调阈值只多 0.08 分；Platt 校准把 ECE 从 11.4% 降到 0.8%，但不改变排序，MF1 不会因此提高。
- **医生之间的参照。** 拿每位医生的标签去对其他医生（按与模型评测相同的权重），得到的 MF1 约为 69.22；Jev 低 1.48 分（对话重采样区间 [0.69, 2.32]）。换一种汇总口径，这个参照会在 67.2 到 69.6 之间变化。
- **不是所有决策模型都这样。** 另一个决策模型 Laya 在 HealthBench 全量上只有 48.89，接近随机。

## 2.4 跨评测看到的规律

这批 judge 最普遍的偏向是放行：HealthBench 和 RuVerBench 上，几乎所有模型的错误放行率（FPR，真实负类被判满足的比例）都高于错误拒绝率（FNR），但这个规律在 ProfBench 上不成立。这是条件错误率的比较；按绝对次数，负类只占 17% 的 AC 上错误拒绝反而更多。下表是三个评测共有的 18 个本地模型。

| 评测 | FPR 高于 FNR 的模型数 | 平均 FPR | 平均 FNR |
| --- | --- | --- | --- |
| HealthBench | 17 / 18 | 61.14% | 17.30% |
| ProfBench | 6 / 18 | 27.77% | 27.11% |
| RuVerBench-DR | 18 / 18 | 73.12% | 4.47% |
| RuVerBench-AC | 16 / 18 | 61.53% | 18.61% |

DR 的真实负类占 60.9%，仍然 18 个模型全部放行偏多，所以这不是“负类少所以放行”。逐条复核里见到的失败类型包括：只要回答提到了相关主题就当成满足；把 rubric 或背景知识补写成原文证据；理由说对了，布尔值却映射反了。这些来自定向抽取的少量案例，不代表各类错误的占比。

其他几条规律（这里跨评测统一用 MF1，即正、负类 F1 的平均，所以 RuVerBench 的数值和第 6 节表里的 AvgBAcc 不同）：

1. **没有全能冠军。** Gemma4-31B 在 ProfBench 和 AC 第一、DR 第二、HealthBench 第四；Qwen3.8-27B 在 HealthBench 和 DR 第一，ProfBench 却只有 72.19。单个评测上的好成绩不能直接迁移：Gemma4-12B 在 HealthBench 和 ProfBench 的 MF1 是 67.44 和 77.97，在 DR 只有 50.47。
2. **参数量不是充分条件。** Qwen3.5-4B 的 DR（64.86）高于所有 7B–14B 模型。同系列 Qwen3-4B 到 14B，HealthBench 和 DR 的 MF1 分别涨 10.87 和 16.89，ProfBench 只涨 0.87（区间含 0），AC 反而降 2.33（区间同样含 0；按 AvgBAcc 则是升 4.73）。
3. **大模型赢在哪里因评测而异。** HealthBench 上 Gemma4-E4B 到 31B 的 +6.50 几乎全来自负类召回（+17.40，正类召回反而 −2.18）；DR 上 +35.34，负类召回 +54.78。AC 则相反：Qwen3.5-4B 的 289 个错误里 236 个是错误拒绝。复核案例里，这些错误拒绝常来自漏看后续执行或误解规则的适用条件。
4. **只看正类 F1 会掩盖失败。** Qwen3-1.7B 在 AC 的 F1+ 是 90.29、F1− 只有 1.32，145 个真实负类放过了 144 个；Qwen3-4B 在 DR 的 984 个负类里只识别出 3 个。所以必须同时报正负两类。
5. **对协议很敏感。** GPT-OSS-20B 在 ProfBench 的单 token 协议下只输出 Harmony 控制符，格式有效率 0%，33.19 不代表能力；RuVerBench 上有 4 个小模型在至少一个域里超过 5% 的判断因协议失败触发了随机兜底。
6. **质量与成本。** RuVerBench 全量上 Qwen3.5-4B 约用 0.92 GPU 小时，Gemma4-31B 和 Qwen3.8-27B 约 14.5–15.0，贵 16 倍左右。
7. **标签本身有分歧。** HealthBench 有 22.5% 的行医生意见不一，20.4% 正好对半；ProfBench 有题目和回答对不上却标为 True 的样本。分数表达的是与给定标签的一致程度。

# 3 HealthBench 元评测上的 meta-harness

## 3.1 问题与设置

第 2 节说明小 judge 便宜但不够准。这一节的问题是：不改模型权重，只改模型外面的代码（harness），并把这件事交给一个强模型自动去做，能不能让 4B 的 Gemma-4-E4B 在 HealthBench 元评测上达到大模型的水平。

**为什么是 Gemma-4-E4B。** 全量官方协议下它是 61.62，负类 F1 39.43，错误放行率 63.9%，单卡 0.17 GPU 秒/条；最强的 Qwen3.8-27B 是 68.59，要两张卡、0.82 GPU 秒/条。

**harness 能做什么。** 每条样本只给（对话、最后一个回答、一条 rubric）。harness 可以组织提示、检索示例、多次调用模型（每条最多 6 次逻辑调用）、读 token 概率、用纯 Python / numpy 做后处理，最后返回 met 或 not_met。它看不到标签、样本编号和类别。

**数据划分。** 按整个对话分组、按类别分层，划分不看标签也不看模型输出。

<div class="text-table">

| 分区 | 行数 | 用途 | 提案者能看到什么 |
| --- | --- | --- | --- |
| 校准池（25%） | 7,379 | 示例库、分类器等一切可学习资产的唯一来源；其中 2,000 行是训练集，其余给 dev 门抽样 | 全部行和标签；训练集还有逐行评测痕迹 |
| 验证（25%） | 7,404 | 验证门 | 只有聚合分数和配对区间 |
| test（50%） | 14,728 | 最后评一次 | 搜索期不评测、不导出结果；Astra 那次没有做读取隔离 |

</div>

**前史。** 9 月 8 日到 12 日先手工迭代了 42 个版本。稳定有效的只有一条线：给模型看**同一条 rubric、同一个对话下医生一致判为通过和不通过的两个回答**（dev 上 65.85 → 67.15）。思考模式、通用复核、多数投票、many-shot、GEPA 式提示搜索（[Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)）、外部医学检索都没有稳定超过它。之后改成自动搜索：第一次自动搜索 v1 产出了下文种子里的守卫式观察器，v2、v3 的教训促成了下面的 v4 流程。

## 3.2 自动搜索怎么跑

每一轮由强模型（提案者）读历史代码和逐行痕迹，写出新的候选 harness；一个确定性的 Python 编排器负责检查、在 GPU 上评测和门控。只有连过两道配对门的候选才会成为新的参考。

![meta-harness 搜索循环](img/figure-004.svg)

***图 4** 一轮搜索。提案者每轮写 4 个候选；编排器做静态检查和训练集评测，再依次过 dev 门和验证门，两道门都过才成为新参考。test 只在截止前评一次。*

1. **提案**：每轮 4 个提案并行，在“从当前参考出发定点修改”和“换一种机制”两种视角之间交替。提案者可以先在最多 64 行上试一个不确定的想法（probe），下一轮看痕迹再交完整候选。
2. **检查**：导入白名单、禁用文件和进程相关调用、泄漏扫描（按对话编号和整段文本匹配，检查候选文件里有没有验证集或 test 的内容，查不出改写或编码过的泄漏）、mock 契约测试；失败回传提案者修复，最多 2 次。
3. **训练集评测**：在 2,000 行训练集上跑完整候选，逐行痕迹回传给提案者做错误分析；这个分数本身不决定接受。
4. **dev 门**：每轮从 dev 池按类别轮转取 1,000 行（顺序由哈希确定，跨轮会重复，60 轮里每行平均用到约 11 次），候选与参考在同一批行上配对比较：MF1 至少 +0.3 且负类 F1 不降。
5. **验证门**：在全量 7,404 行上与参考逐行配对，按对话分组 bootstrap 3,000 次：MF1 差的 95% 区间下界大于 0，且负类 F1 不降。通过就成为新参考。
6. **终测**：截止前，按验证分取最好的 2 个接受候选和 3 个种子，在 test 上各评一次，报告相对种子参考的配对区间。

提案者用过两个：gpt-6-astra（经 <code>codex exec</code>，reasoning effort xhigh）和 Claude Fable 5.1（经 <code>claude -p</code>，effort xhigh，另加了读取限制）。被评模型 Gemma-4-E4B 跑在 vLLM 单张 A100-40GB 上，temperature 0。每条样本硬预算：6 次逻辑调用、每次最多 2,048 个输出 token、行墙钟 180 秒。

## 3.3 提升了多少

最好的 harness 把 Gemma-4-E4B 的 test MF1 从 61.11 提到 72.58，负类 F1 从 37.47 提到 56.83。在同一批 14,728 行上，它高于所有裸判模型和 Jev。

![同一批 test 行上的 MF1 对比](img/figure-005.svg)

***图 5** v4 test 分区（14,728 行、30,360 个医生标签）上的 MF1。裸判模型取自各自全量缓存中的同一批行（Claude 缺 1 行；Gemma-4-E4B 的缓存协议与种子 raw_nothink 略有不同，所以这里是 61.66 而不是 61.11），Jev 取自其全量逐条预测。虚线是最好的裸判模型。*

| 候选 | 来源 | 验证 MF1（7,404 行） | test MF1 | test F1− | 相对种子参考 Δtest [95% CI] | 秒/行 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_nothink：官方 prompt，单次调用 | 种子 | 60.15 | 61.11 | 37.47 | −7.24 [−8.30, −6.18] | 2.90 |
| contrast_bank_v2：同 rubric 正反例检索 | 种子 | 66.60 | 67.47 | 48.86 | −0.88 [−1.24, −0.52] | 11.49 |
| prior_best_v1：检索 + 守卫式观察器 | 种子参考 | 67.12 | 68.35 | 49.66 | – | 11.64 |
| r15d_crossfit_negative_tail | Astra 第 15 轮 | 68.71 | 69.70 | 52.35 | +1.35 [+0.98, +1.74] | 11.65 |
| r29a_corroborated_edit_veto | Astra 第 29 轮 | 69.32 | 70.17 | 53.36 | +1.82 [+1.35, +2.32] | 12.12 |
| r06a_grading_notes_stack | Fable 第 6 轮 | 70.90 | 72.36 | 56.33 | +4.01 [+3.26, +4.73] | 16.79 |
| r07b_cheap_stack_violation_views | Fable 第 7 轮 | 71.48 | **72.58** | **56.83** | **+4.23 [+3.47, +4.98]** | 11.14 |

秒/行是并发 32 下的行墙钟均值。区间是按对话分组的配对 bootstrap，没有做多重比较校正。种子的验证分取自 Astra 那次运行；Fable 运行里重跑的种子验证分与之相差不到 0.2，种子的 test 结果两次运行共用。文中 Fable 的验证增益 +4.54 是相对它自己运行里重跑的种子参考（66.94）算的。

- **增益的构成。** 从裸判到种子参考的 +7.24 来自种子本身：正反例检索是手工阶段做的（+6.36），守卫式观察器来自更早的 v1 自动搜索（+0.88）。v4 的自动搜索在种子之上再加 +1.82（Astra）或 +4.23（Fable）。
- **医生一致的行**（11,383 行）。裸判 66.53，种子参考 76.94，Astra 78.59，Fable 82.62；Jev 在这些行上是 74.53。
- **上限。** 每行只能给一个布尔判决，却要和这一行的全部医生标签比，所以 MF1 到不了 100：按行取医生多数票（对半时判不满足）作判决，这批行上是 86.42。另一个参照是医生之间的一致度，全量口径下约 69.2。
- **成本。** r07b 每行调用 5 次，其中 3 次只做记录、不影响判决；11.14 秒/行和种子参考相当，约为裸判的 3.8 倍。只保留两次判决调用的耗时没有实测。
- **要注意的不对称。** harness 用了校准池里 7,379 行医生标注，裸判模型和 Jev 没有。下一节会看到，光是标签里的信息就值不少分。

## 3.4 提升从哪里来

拆开 Fable 的最佳 harness 看，大头来自两个“读得更准”的 LLM 视角和对它们的标签校准；词汇分类器只多给 0.4 分。但标签本身信息量很大，只按 rubric 给常数判决就已经比裸判高 2 分。先看三条路线各做了什么。

**种子。** 同 rubric 正反例检索（手工阶段做的）：每条样本从校准池取最多 6 对“同一对话、同一 rubric、医生一致”的通过 / 不通过回答放进提示，test 上从 61.11 到 67.47。再加上 v1 自动搜索找到的、按 rubric 路由的守卫式观察器：只允许单向改判，观察时看不到初判、rubric 和示例，不确定就保留初判，到 68.35。

**Astra（60 轮）走的是“否决”路线。** r15d 在校准池上训练一个按对话五折交叉拟合的 TF-IDF 逻辑回归，当参考判满足、而分类器分数 ≤ 0.20 时改判不满足（test +1.35）。r29a 再加一个“最小编辑反事实”：让模型对回答做一处局部修改，如果修改版满足、原版不满足（正反顺序各确认一次），而且分类器也支持，就否决（test 再 +0.47）。这些否决的精度并不高：在医生不对半的行上，两类否决与医生多数一致的比例只有 57%（108/188）和 54%（37/68），MF1 涨主要是因为负类召回从 43.5% 升到 49.8%。

**Fable（16 轮）第一步也是否决，之后换了一条不同的路线。** 四步接受：

1. r01a：同样的词汇先验否决（验证 +2.13）。
2. r03a：不再让模型先写解释、再给布尔值，而是直接读判决位置上 pass / fail 两个 token 的 log-odds，用一个约 32 个参数的逻辑回归（log-odds、词汇先验、每条 rubric 的截距）在 2,000 行训练集的医生标签上拟合，阈值 0.64。
3. r06a：提案者读校准池的医生标注，给 30 条 rubric 各写了一份“评分要点”（第 5 轮先写出、只做记录，第 6 轮起参与判决）：什么情况判不满足、什么情况仍算满足。例如一条急诊转诊类 rubric 写着“只给出‘如果加重就看医生’这类模糊建议、没有具体危险信号”判不满足，“红旗清单不完整或带‘我不是医生’免责声明”不扣分。这份要点作为一个不带示例的新视角加进来。
4. r07b：只保留两个视角（评分要点；精简正反例加评分要点），去掉 7k token 的完整示例调用，更便宜也更准。

r07b 的逐行日志里存了每个组件的分数，所以不用新推理就能逐层拆开看（图 6）。需要重新拟合的变体只用了 2,000 行训练集。按同样的设置重拟合完整模型，各系数与原版相差不到 0.003，阈值选在 0.61（原版 0.64），test 上 72.61 对 72.58，拟合流程可以复现。

![r07b 的逐层拆解](img/figure-006.svg)

***图 6** 同一批 test 行上 r07b 各组成部分的 MF1，全部由逐行日志重算，没有新推理。空心点完全不调用 LLM；虚线是最好的裸判模型（69.14）。*

- **读得准是主体。** 只用评分要点这一个提示、单次调用、不做任何校准，就从 61.11 到 69.04，和 Qwen3.8-27B 裸判（同一批行上 69.03）持平。
- **校准再给 2–3 分。** 单个视角加上 log-odds 校准（rubric 截距加词汇先验），评分要点视角从 69.04 到 71.92，精简正反例视角从 70.27 到 72.53。两个视角一起校准、不用词汇先验是 72.19，再加词汇先验只多 0.39（配对区间 [0.04, 0.71]）；只用精简正反例视角的 72.53 和完整版没有可分辨的差别（区间 [−0.44, 0.33]）。
- **标签里的信息本身就值很多分。** 只按 rubric 给常数判决有 63.14，只用词汇分类器有 66.60，都不调用 LLM。这说明 HealthBench 元评测的 MF1 里有相当一部分可以由“这条 rubric 通常判什么”和回答的表面词汇预测，和 [Bagaria et al. (2026)](https://arxiv.org/abs/2609.02942) 发现“只看 rubric 文本就能以高于随机的水平预测 judge 判决”相呼应。所以拿 harness 和不看标签的裸判模型比，应该把这两个无 LLM 基线一起报。
- **没奏效的方向。** 思考模式（Gemma 上单条耗时约为不开时的 5.6 倍，增益不稳）、不同机制的级联、通用复核（会把判决推向“满足”）、多种检索权重和示例库变体、按 rubric 堆叠专用的否决 / 挽救观察器（在固定 dev 上各 +0.2 到 +0.5，留出集上约为 0）、LLM judge 自身的集成（Fable 那次运行的训练集上，参考与裸判取交集 66.4、三者多数投票 67.4，都不如参考自己的 67.9）。

## 3.5 验收规则：固定子集会把噪声当成增益

v4 换了验收规则（同时也改了并发、每轮提案数、提示和提案者可见的行）之后，验证集上的增益在 test 上兑现了。图 7 把四次搜索放在一起。

![四次搜索在可见集、验证集和 test 上的增益](img/figure-007.svg)

***图 7** 四次搜索的最终参考相对 prior_best_v1（在同一次运行里）的 MF1 增益。可见集：v2 / v3 是固定的 600 行 dev，v4 是 2,000 行训练集；验证：v2 是固定的 1,600 行门样本，其余是全量 7,404 行。v3 取有全量验证分的最后一个接受版本 r36a。*

- **v2。** 提案者看固定的 600 行 dev，验证门是固定的 1,600 行、点估计不低于参考即可。相对 prior_best_v1，dev 和验证门上都涨了 2 分多，test 只有 +0.88。这次搜索其实从 contrast_bank_v2 起步（相对它 test +1.81），而它在 test 上本来就比 prior_best_v1 低 0.93。
- **v3。** 验证改成每轮新抽 1,600 行配对，但 dev 仍是固定的 600 行。到 r36a 为止，接受链在 dev 上涨了 5.07，全量验证只涨 0.90，test 涨 0.70（最后接受的 r44b 在 dev 上是 +5.63，没有全量验证分，test 相同）。事后看，有全量验证分的 6 个接受步骤，配对区间全部跨 0；55 个在 dev 门失败的候选里有 34 个离门槛不到 0.3 分。
- **v4。** 提案者拿到逐行痕迹的 2,000 行和 dev 门抽样的行不重叠（不过 dev 池的行和标签对提案者可见，两边还共享 916 个对话），dev 门每轮轮换抽 1,000 行，验证门要求全量配对区间下界大于 0。到达全量验证的 31 个候选只接受了 2 个，但验证上的 +2.20 在 test 上兑现了 +1.82。Fable 用同一套规则，验证 +4.54、test +4.23。
- **噪声有多大。** 即使 temperature 为 0，换并发数也会翻转 1%–3% 的判决，256 行上 MF1 会抖动约 ±1 分，和 0.3–0.5 分的 dev 门槛同一量级。所以接受决策必须依赖大样本、在同一批行上配对的比较。

## 3.6 提案者：同一套规则，增益从 +1.8 变成 +4.2

![两次搜索中参考的验证 MF1 随轮次的变化](img/figure-008.svg)

***图 8** 两次 v4 搜索里，被接受参考的全量验证 MF1 随轮次的变化。圆点是被接受的版本，菱形是最终参考在 test 上的分数；种子参考的验证分是各自运行中的重跑值。*

- **Astra。** 60 轮，注册 233 个候选（含 91 个 probe），31 个到达全量验证，接受 2 个；第 29 轮之后的 31 轮都没有新的接受。它试过的方向很多，但被接受的两个改动都是在种子之上加否决规则。
- **Fable。** 16 轮，52 个候选，19 个到达全量验证，接受 4 个，第 7 轮以后没有新的接受；第 13 轮起提案调用大多失败（有配额拒绝记录）。从 r03a 起它不再修补种子的判决，而是改成读概率、在标签上校准，并把医生标注提炼成提示。
- **注意。** 两次各只跑了一次。Fable 那次是在 Astra 结果出来之后启动的，还加了读取限制，所以不是严格的单变量对照。

## 3.7 换被评模型、换数据

把被评模型换成 API 模型 gpt-5.6-luna，同样的流程也有效，但终测没跑完；把 Fable 的最佳 harness 搬到 HealthBench Hard 上，对照 GPT-4.1 的判决看不出增益。

**Luna：有效，但没跑完。** 提案者仍是 Astra，规则同 v4。终测要跑 5 个候选、每个 12 片，共 60 个分片，完成了 46 个（contrast_bank_v2 只完成 1 片，未列入表中）。下表是我们用已完成分片的计数直接汇总的，不是正式报告，也没有置信区间。

| 候选 | 验证 MF1（7,404 行） | test，全部 12 片（14,728 行） | test，裸判也跑完的 9 片（10,828 行） |
| --- | --- | --- | --- |
| raw_nothink | 55.91 | 未跑完 | 57.10 |
| prior_best_v1（种子参考） | 60.78 | 62.32 | 61.59 |
| r03d_blind_anchor_full | 64.28 | 65.94 | 65.86 |
| r08a_stable_positive_boundary | 65.02 | 65.82 | 65.65 |

- **方向与 Gemma 相反。** Luna 裸判偏严，在那 9 片上正类 F1 只有 66.53；r08a 把它提到 79.39，负类 F1 只从 47.67 到 51.91。harness 纠正的是各自的偏差方向。
- **被接受的机制也是“用标签校准”。** 把同 rubric 的正反例和目标回答放在一起盲打分，再用正反例的医生标签估一个局部分界。与 Fable 的 log-odds 校准是同一类思路。
- **慢和没跑完的原因。** Luna 经 <code>codex exec</code> 调用，每次判题都是一个独立子进程，顺序单次调用约 12 秒；并发又因一个第 3 轮之后就已失效的超时理由被压在 10×3，最终候选每行约 54 秒。更大的问题是空转：截至 9 月 20 日排查时，69 小时里有 69% 在空转，其中编排器因一个解析 bug 崩溃后无人值守 31.8 小时，另一个重排逻辑的 bug 又卡了 8.6 小时（见第 4 节）。9 月 22 日项目归档时停了下来，还剩 14 个分片，目前没有额度补测。

**HealthBench Hard：没有崩，也看不出增益。** 10 月用 Fable 运行的三个候选在 HealthBench Hard 的 1,123 条共识标准判断上跑了一遍。这里的标签是 GPT-4.1 grader 的判决，不是医生标注。

| 候选 | 对 GPT-4.1 判决的 MF1 | 负类 F1 | 秒/行 |
| --- | --- | --- | --- |
| raw_nothink | 79.95 | 67.99 | 3.3 |
| prior_best_v1 | 72.18 | 54.85 | 15.8 |
| r07b | 80.59 | 70.59 | 16.9 |

harness 是对着医生标签调的，而 GPT-4.1 的判决本身与医生也只有 70.9 的一致度，所以这组数字只能说明 r07b 换到分布外没有崩，不能说明有增益。种子参考在这里反而比裸判低 7.8 分。这可能说明它的示例和观察器依赖分布，但这里的标签和裸判用的是同一个官方 grader prompt，本身就偏向裸判，而且没有单独跑 contrast_bank_v2，分不清是示例还是观察器的问题。要真正检验迁移，需要带人工标签的新数据。

# 4 工程上踩过的坑

- **吞吐要先量再猜。** 最初的设置（关闭 CUDA graph、每卡并发 4）只用了不到 1% 的 KV cache，每卡 57–68 行/分钟。打开 CUDA graph、max_num_seqs 32、并发 16 后是 144–152 行/分钟，瓶颈是每行约 8k token 的 prefill。
- **temperature 0 也不确定。** 换并发数会翻转 1%–3% 的判决。这个现象本身已有人系统分析过（[He & Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)），对我们的影响是：小样本上的门槛本身就带噪声。
- **长时间无人值守要处处设防。** NFS 上“完成”标记会先于数据文件可见；集群的 sbatch 过滤器拒绝含 sleep / wait 的脚本；个别节点 vLLM 初始化失败。最贵的一次是：模型生成的解释里出现 U+2028，Python 的 <code>str.splitlines()</code> 在这里也会断行，把 JSON 行切断，编排器崩溃后停了 31.8 小时；另一个心跳文件缺失时任务永不重排的 bug 又卡了 8.6 小时。
- **算力。** Astra 那次 v4 搜索的评测任务计时约 100 GPU 小时，worker 进程时段约 204 GPU 小时；提案者调用 263 次，单次中位约 10 分钟。

# 5 局限和下一步

## 5.1 现在还不能下的结论

- **只有一个基准、一个完整跑完的被评模型。** harness 搜索只在 HealthBench 元评测上做过，每个配置只跑了一次，没有搜索方差。
- **test 分区被重复使用。** v2、v3、v4 和 Fable 用的是同一批 14,728 行；提案者的沙箱主要限制写入，不是读取隔离（Fable 那次加了读取限制）。编排器在搜索期间没有评测或导出 test，但不能证明提案者在系统层面无法读取。
- **验证集被反复用于接受决策。** Astra 一次就有 31 次全量验证，没有做多重比较或适应性选择的校正。
- **不是“免训练”。** 最好的 harness 都含有在医生标签上训练的组件（TF-IDF 分类器、逻辑回归、评分要点），且按 30 条已知 rubric 的哈希取系数，对没见过的 rubric 不适用。
- **MF1 的提升不等于临床判断更正确。** 它伴随类别召回的权衡，而且有 20% 的行医生对半。
- **没有和“直接训练 judge”比过。** 同样 7,379 行标注拿去做 SFT 或训探针会怎样，还不知道。

## 5.2 下一步

1. **训练基线**：用同一个校准池对 Gemma-4-E4B 做 LoRA SFT，或在它的隐状态上训线性探针，和 r07b 比。
2. **跨基准**：把“评分要点 + log-odds 校准”搬到 ProfBench 和 RuVerBench，这两个评测已经有 20 个模型的基线。
3. **重复一次 Fable 搜索**，或换一个被评模型（比如 Qwen3.5-4B），估计搜索本身的方差。
4. **反事实敏感度测试**：把回答改成满足 / 违反 rubric，看裸判和 harness 的判决是否跟着翻转。

# 6 实验细节

**完整基线结果。** 各评测的官方主指标，百分制。加粗的是生成式 judge 里各列最高值。

| 模型 | HealthBench MF1 | ProfBench Macro-F1 | ProfBench MF1-BI | RuVerBench-DR AvgBAcc | RuVerBench-AC AvgBAcc |
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
| GPT-OSS-20B | 63.52 | 33.19（协议不兼容） | 25.6 | 77.71 | 64.26 |
| Claude Sonnet 5 | 67.27 | – | – | – | – |
| Jev 1.13（接口适配） | 67.74 | 79.76 | 78.3 | 协议不兼容 | 协议不兼容 |

HealthBench 上 GPT-OSS-20B 缺 1 行、Claude 排除了 32 条硬拒答，其余全量。Gemma4-31B 的 HealthBench 结果是开启思考重跑的。RuVerBench 保留了协议失败时的确定性随机兜底。

**代码与版本。** HealthBench 元评测用 simple-evals（[OpenAI](https://github.com/openai/simple-evals)）固定在 commit 652c89d0；ProfBench 用官方仓库（[NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)）固定在 commit b06a29cd。

**统计。** 文中的区间都是按对话（HealthBench）或按任务（ProfBench）分组的配对 bootstrap 95% 区间，没有做多重比较校正。

# 参考文献

1\. Arora, R. K., Wei, J., Hicks, R. S., et al. *HealthBench: Evaluating Large Language Models Towards Improved Human Health*. arXiv:2505.08775, 2025. [https://arxiv.org/abs/2505.08775](https://arxiv.org/abs/2505.08775)

2\. Wang, Z., Jung, J., Lu, X., et al. *ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge*. arXiv:2510.18941, 2025. [https://arxiv.org/abs/2510.18941](https://arxiv.org/abs/2510.18941)

3\. Peng, Y., Qi, Y., Xia, H., et al. *Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?* arXiv:2606.29920, 2026. 提出 RuVerBench. [https://arxiv.org/abs/2606.29920](https://arxiv.org/abs/2606.29920)

4\. TypeSafe. *Jev documentation*. 2026. [https://docs.typesafe.ai/introduction](https://docs.typesafe.ai/introduction)

5\. Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., & Finn, C. *Meta-Harness: End-to-End Optimization of Model Harnesses*. arXiv:2603.28052, 2026. [https://arxiv.org/abs/2603.28052](https://arxiv.org/abs/2603.28052)

6\. OpenAI. *simple-evals*. HealthBench 元评测的官方实现. [https://github.com/openai/simple-evals](https://github.com/openai/simple-evals)

7\. NVIDIA. *ProfBench*. 数据与官方评分器. [https://github.com/NVlabs/ProfBench](https://github.com/NVlabs/ProfBench)

8\. Agrawal, L. A., Tan, S., Soylu, D., et al. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*. arXiv:2507.19457, 2025. [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)

9\. Bagaria, A., Sundaram, S. S., Krishnan, G. S., & Ravindran, B. *Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation*. arXiv:2609.02942, 2026. [https://arxiv.org/abs/2609.02942](https://arxiv.org/abs/2609.02942)

10\. He, H., & Thinking Machines Lab. *Defeating Nondeterminism in LLM Inference*. Thinking Machines Lab: Connectionism, 2025. [https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)`
};
