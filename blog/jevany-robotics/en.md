Decision models (Jev-style, System One) answer questions differently from generative models: give one a state and a set of options, and it returns a probability for every option in a single forward pass, without generating text. We wanted to know how far a 4B decision model gets at judging robot manipulation when it is trained only on public data with our own recipe.

# Conclusions

1. **Public data alone is enough to train a strong 4B decision model.** On RoboPulse, RoboRewardBench, and human-rated real-robot rollouts (RoboArena) it beats every multimodal Jev-style model we tested, including 27B ones; on RoboPulse++ and policy ranking it ties with the 27B models. Specialist robot reward models still win on their home ground: Robo-Dopamine on fine-grained progress (RoboPulse 90% vs. 71%), RoboReward on 1–5 scoring (MAE 0.358 vs. 0.443).
2. **Its probabilities are trustworthy, and it is fast.** On RoboRewardBench its NLL is the lowest of every model we tested (1.02; RoboReward-8B 1.39); 185 ms per judgment on a single A100, versus 316 ms for Robo-Dopamine-8B, and 732–838 ms for a 27B judge spread over two GPUs.
3. **The weaknesses are clear, and they look fixable.** Fine-grained progress, the synthetic questions in RoboRewardBench, and multi-camera views are where it is weakest. After continued training on 22k new real-robot questions, two-camera progress AUROC rises from 0.65 to 0.77 (the confidence interval just touches 0), but single-view gets worse and the overall picture is flat, so the release stays at the first version (v0.2).

# One model, every question you would ask about a rollout

Give it a robot rollout and a question, and it returns a probability for every possible answer. Anything you would ask about a rollout, it can answer:

- **Progress**: between two moments, did the task move forward?
- **Whole-episode score**: how well was the task done, 1–5?
- **Step success**: did this subtask succeed?
- **Failure type**: wrong object, wrong placement, missed grasp, or no motion at all?
- **Plan check**: before acting, is this plan correct, and where does it go wrong?
- **Failure localization**: at which step did the rollout fail?
- **Preference**: which of two rollouts is better?

Ask "how far along is the task" at every frame and you get a progress curve (Figure 1). In the successful rollout the score climbs to 5 as the corn lands on the plate; in the failed one it stays at 1.

![Progress curves on two real-robot rollouts](img/figure-001.png)

***Figure 1** Frame-by-frame scoring of two RoboArena real-robot rollouts (the RoboArena subset of RoboRewardBench, DROID platform): for every prefix of the video we ask "if the episode ended here, how well is the task done (1–5)", and the curve is the expected score. Both examples were chosen because the model's final score agrees with the human rating.*

# A decision model returns every option's probability in one forward pass

Each question pairs a state (task instruction, image, or video) with a typed question: yes/no, multiple choice, or 1–5 scoring, with the options and the grading rubric written into the question. The model encodes the state together with all the options, a pointer head scores the options, and it emits a probability distribution directly instead of generating token by token.

That gives two properties that are useful for judging robots. First, any number of options costs one pass, so latency does not grow with the length of the answer. Second, the output already is a probability, so it can serve directly as a reward signal, as a filter for data, or as a way to decide when a case should go to a human for review.

JevAny-Robotics-4B warm-starts from our open-source JevAny-Qwen3.5-4B-LoRA, which was trained on general decision data on a Qwen3.5-4B base. We train only the LoRA (r = 8, 17.6M trainable parameters) and the pointer head: one epoch, 5,780 steps, global batch 32, learning rate 1e-5, about 13–14 hours on 4×A100-40GB. Images are used at native resolution; videos are resampled to roughly 1–2 fps and 8–32 frames, then fed in with all frames.

# Training data comes only from public train splits

The question we wanted to answer: with no new data collection and no human annotation, how far can we train a 4B model just by rewriting existing public data as decision questions?

The sources are the train splits of RoboReward, Robo-Dopamine GRM, Guardian (BDV2 / RLBench), DenseReward, RoboFAC, ViFailback, and RoboVQA, covering real and simulated robots, images and video. Each source labels things differently: whole-episode 1–5 scores, frame-level progress, success labels, failure types, whether a plan is correct. We rewrite all of it uniformly into 7 question types: pairwise progress, adjacent-frame direction, 1–5 scoring, execution success, failure type and plan checks, failure localization, and trajectory preference — 184,942 questions in total (Figure 2).

![Training data pipeline](img/figure-002.png)

***Figure 2** How the training data was built. Train splits only; after decontamination against the four test sets, everything is rewritten as 7 question types and used for one epoch of LoRA SFT.*

Decontamination takes two steps. The first is an episode-level blocklist: any episode that appears in the RoboPulse, RoboPulse++, RoboRewardBench, or JevAny-Robotics-Public test sets is dropped. The second is frame-level hashing: training frames are compared against 168,861 test frames with a 64-bit perceptual hash, and a single hit drops the whole episode. Together with rule filtering and deduplication, this removes about 20k suspected-contaminated questions, about 10k that break the rules, and about 1,500 duplicates.

One caveat: decontamination removes the same episode, not the same scene or the same robot. RoboPulse and RoboRewardBench are built from the same public datasets as our training data, so the test questions are not entirely unfamiliar scenes for us. The same is true of the specialist reward models we compare against: RoboReward trains on the RoboReward train split, Robo-Dopamine on GRM data. Two of the five success-detection sets in JevAny-Robotics-Public (BDV2, RoboVQA) have their train split in our training data, and we flag that when we report the number below.

The training set, JevAny-Robotics-Train-v0.2, is still clearing redistribution permissions for some sources and will be released once that is done.

# How far public data takes a 4B model: past other multimodal Jev models, with specialists still holding their home ground

We put 8 models into the same harness, with identical questions and media. On one side are multimodal Jev-style decision models: JevAny-27B, OneJev-27B, OmniJev-4B. On the other are reward models trained specifically for robotics: Robo-Dopamine-GRM-2.0-8B, RoboReward-8B / 4B, Robometer-4B (Figure 3).

![Model comparison across seven benchmarks](img/figure-003.png)

***Figure 3** Scores of 8 models on 7 evaluations. No model wins every row; JevAny-Robotics-4B stays near the top on every row.*

The main numbers:

| Evaluation | Metric | Ours 4B | Robo-Dopamine-8B | RoboReward-8B | Robometer-4B | JevAny-27B | OneJev-27B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RoboPulse (1,800 pairs) | acc / AUROC | 71.1 / 0.806 | **90.1 / 0.953** | 49.9 / 0.499ᵃ | 63.6 / 0.700 | 56.6 / 0.627 | 60.4 / 0.651 |
| RoboPulse++ (7,720 pairs) | AUROC | **0.728** | 0.717 (0.730 debiased) | 0.489ᵃ | 0.630 | 0.692 | 0.700 |
| RoboRewardBench (2,831 episodes) | MAE ↓ | 0.443 | 1.023ᵇ | **0.358** | 0.777 | 0.692 | 0.805 |
| RoboArena real robot (1,000 episodes) | MAE ↓ | **0.687** | 0.928ᵇ | 0.968 | 0.834 | 0.867 | 0.919 |
| Policy ranking (805 pairs) | agreement with humans, % | 88.4 | 78.3ᵇ | 85.3 | 75.5 | 88.3 | **88.8** |
| Success detection (5 sets) | mean AUROC | 0.872ᶜ | 0.723 | 0.688 | 0.756 | not run | **0.903** |
| Question types answerable (13) |  | 13 | 6 | 6 | 6 | 13 | 13 |

ᵃ RoboReward only scores whole episodes; the pairwise questions are our adaptation. ᵇ Robo-Dopamine uses first-to-last-frame progress as its score, with the cut point fitted on val. ᶜ Two of the five sets have their train split in our training data. The adaptation rules are in "Experimental details". Numbers for RoboReward-4B and OmniJev-4B are on the model card.

Against the multimodal Jev-style models, this 4B model beats JevAny-27B and OneJev-27B, which have about 7× its parameters, on RoboPulse, RoboRewardBench, and RoboArena real-robot rollouts. The largest gap is RoboPulse: 71.1% against 56.6% and 60.4%. On RoboPulse++ our number is higher, but the gap is inside the confidence interval; on policy ranking the three tie. OneJev-27B is still better at success detection and trajectory preference (AUROC 0.903 vs. 0.872, paired difference −0.031 \[−0.052, −0.012\]; preference 0.836 vs. 0.771).

Against the specialist reward models, the conclusion depends on whether the question is their home turf.

- **Robo-Dopamine leads by a wide margin on fine-grained progress.** RoboPulse 90.1% vs. 71.1%, paired difference −19.1 \[−21.3, −16.8\]. RoboPulse's input format (3 views plus start and end reference images) matches its training format exactly. On the consecutive-step RoboPulse++ the two tie (paired difference +0.011 \[−0.026, +0.045\]).
- **RoboReward is better on RoboRewardBench overall.** MAE 0.358 vs. 0.443, and the gap comes from the synthetic OXE questions, which is exactly how its training data was constructed. On the 1,000 human-rated real-robot videos alone (RoboArena), our error is the lowest of all models: 0.687 vs. 0.968 for RoboReward-8B, paired difference −0.281 \[−0.380, −0.190\].
- **The specialists answer only 6 of the 13 question types.** On success detection our AUROC is 0.116 above the best reward model; failure type, plan checking, and failure localization they simply do not answer.

Our reading: with public data alone, a small decision model can answer every kind of question and place near the top on each; but on one fixed format, a model trained specifically for that format is still stronger.

# Trustworthy probabilities, and fast

A reward signal has to be right, and it has to know when it is uncertain. Every RoboRewardBench video requires a distribution over 1–5, and we compared every model that can produce one (Figure 4).

![Reliability diagram on RoboRewardBench](img/figure-004.png)

***Figure 4** Reliability diagram on RoboRewardBench. The horizontal axis is the model's confidence in its top-scoring option, the vertical axis the actual hit rate. RoboReward is clearly overconfident on real-robot video.*

| Model | NLL ↓ | ECE ↓ | Brier ↓ | RoboArena NLL ↓ | RoboArena ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| **JevAny-Robotics-4B** | **1.021** | 0.103 | **0.498** | **1.143** | 0.065 |
| RoboReward-8B | 1.388 | 0.227 | 0.563 | 2.082 | 0.347 |
| RoboReward-4B | 1.212 | 0.175 | 0.532 | 1.649 | 0.240 |
| JevAny-27B | 1.158 | **0.061** | 0.598 | 1.220 | **0.050** |
| OneJev-27B | 1.252 | 0.090 | 0.649 | 1.287 | 0.131 |

Our NLL is the lowest of every model that outputs probabilities, 0.367 below RoboReward-8B \[0.220, 0.517\]. The difference is mostly on real-robot video: on the RoboArena subset RoboReward-8B is on average 77% confident and right 42% of the time; we are 58% confident and right 52% of the time. Exact hit rates are close (62.1% vs. 63.5%), so the gap is not in being right or wrong but in whether the probability can be trusted.

Two points need to be clear. First, our curve also sits below the diagonal, so it is not "80% confident means right 80% of the time". Second, by ECE, JevAny-27B is lower than we are (0.061 vs. 0.103). What we can claim is the lowest NLL and Brier, not the best calibration on every metric.

These probabilities also rank policies. In RoboArena, one session has several policies attempt the same task. Ranking two policies by expected score agrees with the human ranking on 88.4% of the 805 pairs, tying with both 27B models and 4.7 points above RoboReward-4B \[1.4, 8.0\].

On speed, the benefit of a single forward pass is real. The table below gives median end-to-end latency at batch 1 on one A100-40GB.

| Model | RoboPulse per pair | RoboRewardBench per episode |
| --- | --- | --- |
| JevAny-Robotics-4B, flash-linear-attention + causal-conv1d | **185 ms** | 279 ms |
| JevAny-Robotics-4B, PyTorch reference kernel | 294 ms | 392 ms |
| Robo-Dopamine-8B (vLLM) | 316 ms | – |
| RoboReward-4B / 8B, single forward pass | – | **142 / 160 ms** |
| RoboReward-8B, normal generation | – | 361 ms |
| 27B Jev-style model, reference kernel, 2 GPUs | 732–838 ms | – |

We are faster than Robo-Dopamine-8B and the 27B models, but slower than RoboReward on RoboRewardBench. The reason is that we feed every frame at native resolution: the median episode is 1,433 visual tokens, about 2.5× RoboReward's 564. Compressing the input to JevAny's default of 8 frames and 512×512 total pixels brings latency down to 187 ms, but RoboRewardBench MAE rises from 0.443 to 0.607 and RoboArena from 0.687 to 1.289. So this model really has to run on the full media.

# Where these conclusions stop

## Fine-grained progress is still far off

On RoboPulse, 71.1% against Robo-Dopamine's 90.1% is the largest gap in this post. We checked that test overlap is not the cause: on the 939 questions whose episodes are not in Robo-Dopamine's public training data, it still scores 87.5% and we score 68.8%. By source, it leads on all 9. Our reading is that small progress changes need a large amount of dense supervision in the same format, and public data only gave us a fraction of that.

RoboPulse++ has a threshold problem as well. In training, rising and falling are 1:1; in the test set 81% rise. At the default threshold of 0.5 accuracy is only 60.4%, below the 81.1% of "always predict rising". Ranking quality (AUROC) and balanced accuracy are the meaningful metrics here, and in practice the threshold should be set on your own data.

## The synthetic questions in RoboRewardBench

Part of RoboRewardBench OXE is a set of constructed questions: successful videos with the task instruction swapped, and videos truncated in the middle. RoboReward was trained on exactly this kind of construction and is clearly better on both (MAE 0.223 vs. 0.297 and 0.418 vs. 0.584). We often read a truncated video as "almost done". A few other small subsets (such as utokyo\_xarm\_bimanual and stanford\_hydra) have high error, and the training data holds only a few hundred scoring questions for them.

## Multi-view

Questions that require watching 2–3 cameras at once remain the weakest category. On RoboPulse++, single-view AUROC is 0.758, two-view 0.646, three-view 0.672.

## Behind the 27B on success detection and preference

On success detection and trajectory preference in JevAny-Robotics-Public, OneJev-27B is better. And two of the five success sets have their train split in our training data, so this number is biased in our favor.

## Slower than RoboReward

Full media means more tokens per episode, and Qwen3.5's linear-attention layers are slower than the attention in Qwen3-VL, so on RoboRewardBench we take 279 ms per episode against 142–160 ms for a single RoboReward forward pass.

## Calibration degrades outside robotics

On general decision questions accuracy roughly holds at the level of the JevAny-4B starting point, but the probabilities become overconfident. For general use, take JevAny-4B itself.

## Clues from a second training round

Can the weaknesses be fixed with data? We ran one round of continued training (internally v0.3.1): starting from the model in this post, we added 22,051 new real-robot questions (AIRoA, AgiBot, RoboCOIN, RoboMIND, RH20T) and replayed part of the old data. Some things went up and some went down.

| Evaluation | Metric | This model | After continued training | Difference \[95% CI\] |
| --- | --- | --- | --- | --- |
| RoboPulse | acc | 71.1 | 72.2 | +1.2 \[0.0, +2.4\] |
| RoboPulse++ two-view | AUROC | 0.646 | 0.770 | +0.124 \[−0.005, +0.230\] |
| RoboPulse++ single-view | AUROC | 0.758 | 0.726 | −0.031 \[−0.056, −0.009\] |
| RoboPulse++ three-view | AUROC | 0.672 | 0.586 | −0.086 \[−0.186, +0.021\] |
| RoboPulse++ all | AUROC | 0.728 | 0.720 | −0.008 \[−0.035, +0.022\] |
| RoboRewardBench | MAE ↓ | 0.443 | 0.458 | +0.015 \[−0.018, +0.032\] |
| JevAny-Robotics-Public | skill | 0.567 | 0.590 | +0.023 \[+0.001, +0.046\] |

The two-view gain is the largest, but its confidence interval just touches 0, and it comes mostly from one family of constructed questions (AIRoA's reversed actions). Three-view gets worse, consistent with RH20T's constructed questions going the wrong way on the dev set as training proceeds. Single-view gets significantly worse. Failure and success questions get significantly better. At the default threshold, the continued model calls 87.5% of questions "rising".

So we are still releasing the model in this post. What the extra round shows is that more data can move a capability onto a specific weakness, but gains from constructed questions do not necessarily transfer, and they can make other views worse. Next we plan to use real failures and real regressions (rather than reversed playback or reversed actions), add more truncated clips, and replay all of the RoboRewardBench-style scoring questions. If you have annotated real-robot rollouts, we would love to hear from you.

# Experimental details

**One harness.** Every model runs in our harness, with identical questions and media, and every model answered every question. Confidence intervals are 95% intervals from a paired cluster bootstrap (resampling by episode or task group). Test sets: RoboPulse 1,800 pairs, RoboPulse++ 7,720 pairs, RoboRewardBench 2,831 episodes (checked item by item against the test split of teetone/RoboReward on the Hub), and the 13 question types of JevAny-Robotics-Public.

**How the specialists answer questions they were not trained on.** All of the rules below were fixed before scoring.

- RoboReward: whole-episode 1–5 scoring only. For pairwise questions, we take the expected score of (before, after) minus that of (after, before). It was not trained to judge pairs, so scoring near chance on RoboPulse does not mean it cannot judge progress.
- Robometer: debiased frame-level progress; the 1–5 cut points for RoboRewardBench are fitted on val.
- Robo-Dopamine: we replicate GRMInference from PRM-as-a-Judge (vLLM). On RoboPulse++ we fixed incremental as the primary readout and also report forward and position-bias-removed variants; the reference start is the first frame of the episode, and a blank image is used when there is no goal image. RoboRewardBench, policy ranking, and the success questions use first-to-last-frame progress, with cut points fitted on RoboReward val; the fitted cut points are such that it never predicts a 5. The 146 requests out of 23,559 that exceeded 8,192 tokens were rerun with images scaled to ≤ 76,800 pixels.

**RoboReward's numbers do not match the leaderboard.** We measure MAE 0.358 / 0.407 for RoboReward-8B / 4B, while the paper and leaderboard report 0.665 / 0.845. The test set is the same; free generation and prefilled readout give 0.357 and 0.358; zero-shot Qwen3-VL-4B is 0.939 for us against 1.032 in the paper. An independent effort, RoboRMBench, also reports much lower RoboReward error than the leaderboard. We could not obtain the leaderboard's evaluation setup, so this post reports only our own measurements and does not compare against it.

**Robo-Dopamine overlap check.** Using our decontamination of the GRM data, we mark the RoboPulse questions that overlap with GRM episodes (861 / 1,800). Overlapping questions are easier for every model, by about 5 points; the numbers above are on the non-overlapping subset.

**Calibration metrics.** NLL and Brier are computed over the five scores; ECE uses 10 equal-width confidence bins, with confidence taken as the probability of the top-scoring option. Robo-Dopamine and Robometer do not emit a probability for each score and are excluded from the calibration comparison.

**Policy ranking.** Pairs of policies within the same RoboArena session that received different human scores: 134 sessions, 805 pairs. Each policy takes the mean expected score over its views; agreement with the human ranking counts as correct, and ties count as half.

**Speed.** One A100-40GB, batch 1, median end-to-end latency. All accuracies in this post use the PyTorch reference kernel; with flash-linear-attention + causal-conv1d they are unchanged within numerical error (RoboPulse 71.4%, RoboRewardBench MAE 0.446). The 27B models run the reference kernel across 2 GPUs.

# How to use it

The model is on Hugging Face: [SimpleJev/JevAny-Robotics-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Robotics-4B-LoRA). The evaluation set: [SimpleJev/JevAny-Robotics-Public](https://huggingface.co/datasets/SimpleJev/JevAny-Robotics-Public). The code: [SimpleJev/JevAny](https://github.com/SimpleJev/JevAny).

Start the server with <code>--media full</code> (or set <code>JEVANY_MEDIA=full</code>). Then images are used at native resolution and videos with all frames, matching training and evaluation. We verified on 300 RoboRewardBench videos and 300 RoboPulse pairs that the probabilities this produces match the ones reported here exactly. JevAny's default compresses media and is clearly worse (see above). This option is still being merged into the JevAny main branch, so follow the model card for installation.

If you use it for a binary decision (say, whether progress is rising), set the threshold on your own data rather than using 0.5.

# References

1\. RoboPulse. arXiv:2603.21669.

2\. RoboPulse++. arXiv:2608.14284.

3\. RoboReward: RoboReward and RoboRewardBench. arXiv:2601.00675.

4\. Robometer. arXiv:2603.02115.

5\. Robo-Dopamine. arXiv:2512.23703; Robo-Dopamine 2.0. arXiv:2608.15680.

6\. RoboRMBench. arXiv:2609.05401.

7\. PRM-as-a-Judge (Robo-Dopamine inference code).

8\. JevAny-Robotics-4B-LoRA model card. [https://huggingface.co/SimpleJev/JevAny-Robotics-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Robotics-4B-LoRA)

9\. JevAny. [https://github.com/SimpleJev/JevAny](https://github.com/SimpleJev/JevAny)
