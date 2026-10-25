window.ROBOTICS_CONTENT = {
  en: String.raw`Decision models (Jev-style, System One) answer questions differently from generative models: give one a state and a set of options, and it returns a probability for every option in a single forward pass, without generating text. We wanted to know how far a 4B decision model gets at judging robot manipulation when it is trained only on public data with our own recipe.

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

9\. JevAny. [https://github.com/SimpleJev/JevAny](https://github.com/SimpleJev/JevAny)`,

  zh: String.raw`<p class="post-links"><strong>模型</strong> <a href="https://huggingface.co/SimpleJev/JevAny-Robotics-4B-LoRA">JevAny-Robotics-4B-LoRA</a><span class="sep">·</span><strong>评测集</strong> <a href="https://huggingface.co/datasets/SimpleJev/JevAny-Robotics-Public">JevAny-Robotics-Public</a><span class="sep">·</span><strong>代码</strong> <a href="https://github.com/SimpleJev/JevAny">SimpleJev/JevAny</a></p>

决策模型（Jev 式、System One）回答问题的方式和生成式模型不同：给它一个状态和一组选项，它一次前向就给出每个选项的概率，不生成文本。我们想知道，只用公开数据和我们的训练方法，这样一个 4B 的决策模型在机器人操作的评判上能做到什么程度。

# 1 结论

1. **公开数据就能把 4B 决策模型训得很强。** 在 [RoboPulse](https://arxiv.org/abs/2603.21669)、[RoboRewardBench](https://arxiv.org/abs/2601.00675) 和人工打分的真机 rollout（RoboArena）上，它超过我们测过的所有多模态 Jev 式模型，包括 27B 的；RoboPulse++ 和给策略排序上与 27B 打平。专用的机器人奖励模型在各自的主场仍然更强：Robo-Dopamine 在细粒度进度判断上（RoboPulse 90% 对 71%），RoboReward 在 1–5 打分上（MAE 0.358 对 0.443）。
2. **它的概率可信，也快。** RoboRewardBench 上它的 NLL 是所有测过的模型里最低的（1.02，RoboReward-8B 1.39）；单张 A100 上每次判断 185 ms，Robo-Dopamine-8B 316 ms，27B judge 用两张卡也要 732–838 ms。
3. **短板清楚，而且看起来能补。** 细粒度进度、RoboRewardBench 的合成题、多相机视角是它最弱的地方。加 2.2 万道新真机题续训后，两相机进度的 AUROC 从 0.65 升到 0.77（置信区间刚好碰到 0），但单视角变差，整体差不多持平，所以发布版仍是第一版（v0.2）。

# 2 一个模型，回答一段 rollout 的所有问题

给它一段机器人 rollout 和一个问题，它对每个可能的答案给出概率。一段 rollout 上你会问的问题，它都能回答：

- **进度**：两个时刻之间，任务有没有往前推进？
- **整段打分**：任务完成得怎么样，1–5 分？
- **步骤成败**：这个子任务成功了吗？
- **失败类型**：拿错物体、放错位置、没抓住，还是根本没动？
- **计划检查**：动手之前，这个计划对不对，错在哪？
- **失败定位**：rollout 在哪一步出了错？
- **偏好**：两段 rollout 哪段更好？

在每一帧都问一次“任务完成到哪了”，就得到一条进度曲线（图 1）。成功的那段，玉米落到盘子上时分数升到 5；失败的那段，分数一直停在 1。

![Progress curves on two real-robot rollouts](img/figure-001.png)

***图 1** 两段 RoboArena 真机 rollout（RoboRewardBench 的 RoboArena 子集，DROID 平台）的逐帧打分：对视频的每个前缀问“如果在这里结束，任务完成得怎么样（1–5）”，曲线是期望分。两段都选自模型最终分数和人工打分一致的例子。*

# 3 一次前向给出所有选项的概率

每道题由状态（任务指令、图像或视频）和一个带类型的问题组成：是非题、多选题或 1–5 打分题，选项和判分标准都写在题里。模型把状态和所有选项一起编码，由 pointer head 在选项上打分，直接输出一个概率分布，不逐词生成。

这带来两个对机器人评判很有用的性质。一是选项再多也只算一次，延迟不随答案长度增长。二是输出本身就是概率，可以直接当奖励信号、拿来筛数据，或者按置信度决定什么时候交给人复核。

[JevAny-Robotics-4B](https://huggingface.co/SimpleJev/JevAny-Robotics-4B-LoRA) 从我们开源的 [JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) 热启动。后者在通用决策数据上训练过，底座是 Qwen3.5-4B。我们只训 LoRA（r = 8，共 1,760 万可训练参数）和 pointer head：一个 epoch、5,780 步、全局 batch 32、学习率 1e-5，在 4 张 A100-40GB 上约 13–14 小时。图像用原分辨率；视频先重采样成约 1–2 fps、8–32 帧的短片，再全部帧输入。

# 4 训练数据只来自公开数据集的 train split

我们想回答的问题是：不采新数据、不做人工标注，只把现有公开数据改写成决策题，能把一个 4B 模型训到什么程度。

来源是 [RoboReward](https://arxiv.org/abs/2601.00675)、[Robo-Dopamine GRM](https://arxiv.org/abs/2512.23703)、[Guardian](https://arxiv.org/abs/2512.01946)（BDV2 / RLBench）、[DenseReward](https://arxiv.org/abs/2607.13033)、[RoboFAC](https://arxiv.org/abs/2505.12224)、[ViFailback](https://arxiv.org/abs/2512.02787) 和 [RoboVQA](https://arxiv.org/abs/2311.00899) 的 train split，涵盖真机和仿真、图像和视频。每个来源原本的标签形式不同：整段 1–5 分、帧级进度、成败标签、失败类型、规划是否正确。我们把它们统一改写成 7 类题：成对进度、相邻帧方向、1–5 打分、执行成败、失败类型与规划检查、失败定位、轨迹偏好，共 184,942 道（图 2）。

![Training data pipeline](img/figure-002.png)

***图 2** 训练数据的构造。只用各数据集的 train split，对 4 个测试集去重后改写成 7 类决策题，再做一个 epoch 的 LoRA SFT。*

去重分两步。第一步是 episode 级黑名单：凡是出现在 RoboPulse、RoboPulse++、RoboRewardBench 或 JevAny-Robotics-Public 测试集里的 episode，一律不用。第二步是帧级哈希：把训练帧和 168,861 张测试帧做 64 位感知哈希比对，命中一帧就丢掉整个 episode。两步加上规则过滤和查重，共去掉约 2 万道疑似污染的题、约 1 万道不合规则的题和约 1,500 道重复题。

需要说明的是，去重去掉的是同一个 episode，去不掉同一个场景和同一台机器人。RoboPulse 和 RoboRewardBench 与训练数据出自同一批公开数据集，所以测试题对我们来说并不是完全陌生的场景。对比里的专用奖励模型也是这样：RoboReward 在 RoboReward train 上训练，Robo-Dopamine 在 GRM 数据上训练。JevAny-Robotics-Public 的 5 个成败集合里有 2 个（BDV2、RoboVQA）的 train split 在我们的训练数据里，下文报这一项时会标出来。

训练集 JevAny-Robotics-Train-v0.2 还在确认部分来源的再发布许可，确认后公开。

# 5 公开数据能把 4B 训到哪里

我们把 8 个模型放进同一套 harness，题目和媒体完全相同。一边是支持多模态的 Jev 类决策模型：JevAny-27B、OneJev-27B、OmniJev-4B；另一边是专门为机器人训练的奖励模型：[Robo-Dopamine-GRM-2.0-8B](https://arxiv.org/abs/2608.15680)、RoboReward-8B / 4B、[Robometer-4B](https://arxiv.org/abs/2603.02115)（图 3）。

![Model comparison across seven benchmarks](img/figure-003.png)

***图 3** 8 个模型在 7 项测试上的成绩。没有一个模型在所有行都赢；JevAny-Robotics-4B 在每一行都靠前。*

主要数字如下。

| 测试 | 指标 | 我们 4B | Robo-Dopamine-8B | RoboReward-8B | Robometer-4B | JevAny-27B | OneJev-27B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RoboPulse（1,800 对） | acc / AUROC | 71.1 / 0.806 | **90.1 / 0.953** | 49.9 / 0.499ᵃ | 63.6 / 0.700 | 56.6 / 0.627 | 60.4 / 0.651 |
| RoboPulse++（7,720 对） | AUROC | **0.728** | 0.717（去偏 0.730） | 0.489ᵃ | 0.630 | 0.692 | 0.700 |
| RoboRewardBench（2,831 段） | MAE ↓ | 0.443 | 1.023ᵇ | **0.358** | 0.777 | 0.692 | 0.805 |
| RoboArena 真机（1,000 段） | MAE ↓ | **0.687** | 0.928ᵇ | 0.968 | 0.834 | 0.867 | 0.919 |
| 策略排序（805 对） | 与人工一致 % | 88.4 | 78.3ᵇ | 85.3 | 75.5 | 88.3 | **88.8** |
| 成败判断（5 个集合） | 平均 AUROC | 0.872ᶜ | 0.723 | 0.688 | 0.756 | 未跑 | **0.903** |
| 能回答的题型（13 类） |  | 13 | 6 | 6 | 6 | 13 | 13 |

ᵃ RoboReward 只做整段打分，成对题是我们改造的。ᵇ Robo-Dopamine 用首帧到末帧的进度当分数，切点在 val 上拟合。ᶜ 5 个集合里有 2 个的 train split 在我们的训练数据里。改造规则见「实验细节」。RoboReward-4B 和 OmniJev-4B 的数字见模型卡。

和同类的多模态 Jev 类模型比，这个 4B 模型在 RoboPulse、RoboRewardBench 和 RoboArena 真机上都超过了参数量约为它 7 倍的 JevAny-27B 和 OneJev-27B。差距最大的是 RoboPulse：71.1% 对 56.6% 和 60.4%。[RoboPulse++](https://arxiv.org/abs/2608.14284) 上我们的数字更高，但差距在置信区间内；策略排序上三者打平。成败判断和轨迹偏好上 OneJev-27B 仍然更好（AUROC 0.903 对 0.872，配对差 −0.031 \[−0.052, −0.012\]；偏好 0.836 对 0.771）。

和专用奖励模型比，结论取决于题目是不是它们的主场。

- **Robo-Dopamine 在细粒度进度上领先很多。** RoboPulse 90.1% 对 71.1%，配对差 −19.1 \[−21.3, −16.8\]。RoboPulse 的输入格式（3 视角 + 起止参考图）和它的训练格式完全一致。在连续步的 RoboPulse++ 上两者打平（配对差 +0.011 \[−0.026, +0.045\]）。
- **RoboReward 在 RoboRewardBench 整体上更好。** MAE 0.358 对 0.443，差距来自合成的 OXE 题，这正是它训练数据的构造方式。只看 1,000 段人工打分的真机视频（RoboArena），我们的误差是所有模型里最低的：0.687 对 RoboReward-8B 的 0.968，配对差 −0.281 \[−0.380, −0.190\]。
- **专用模型只答得了 6 类题。** 成败判断的 AUROC 我们比最好的奖励模型高 0.116；失败类型、规划检查和失败定位，它们本来就不回答。

我们的理解是，只用公开数据，一个小的决策模型可以做到什么题都答、每项都靠前；但在某一种固定格式上，专门为这种格式训练的模型仍然更强。

# 6 概率可信，速度也快

奖励信号不只要判得对，还要知道自己什么时候不确定。RoboRewardBench 的每段视频都要在 1–5 分上给出分布，我们比较了所有能输出这个分布的模型（图 4）。

![Reliability diagram on RoboRewardBench](img/figure-004.png)

***图 4** RoboRewardBench 上的可靠性图。横轴是模型对最高分选项的置信度，纵轴是实际命中率。RoboReward 在真机视频上明显过度自信。*

| 模型 | NLL ↓ | ECE ↓ | Brier ↓ | RoboArena NLL ↓ | RoboArena ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| **JevAny-Robotics-4B** | **1.021** | 0.103 | **0.498** | **1.143** | 0.065 |
| RoboReward-8B | 1.388 | 0.227 | 0.563 | 2.082 | 0.347 |
| RoboReward-4B | 1.212 | 0.175 | 0.532 | 1.649 | 0.240 |
| JevAny-27B | 1.158 | **0.061** | 0.598 | 1.220 | **0.050** |
| OneJev-27B | 1.252 | 0.090 | 0.649 | 1.287 | 0.131 |

我们的 NLL 是所有输出概率的模型里最低的，比 RoboReward-8B 低 0.367 \[0.220, 0.517\]。差别主要在真机视频上：RoboArena 子集里，RoboReward-8B 平均有 77% 的把握，实际判对 42%；我们平均 58% 的把握，判对 52%。两者精确命中率接近（62.1% 对 63.5%），所以差距不在判得对不对，而在概率是否可信。

这里有两点要说清楚。第一，我们的曲线也在对角线下方，概率并不是「80% 把握就 80% 判对」。第二，按 ECE 算，JevAny-27B 比我们更低（0.061 对 0.103）。我们能说的是 NLL 和 Brier 最低，不是校准在每个指标上都最好。

这些概率也能拿来给策略排序。RoboArena 里同一个 session 会让几个策略做同一个任务。我们用期望分给两个策略排序，在 805 对上和人工排序一致 88.4%，和两个 27B 打平，比 RoboReward-4B 高 4.7 \[1.4, 8.0\] 个点。

速度上，一次前向的好处是实打的。下表是单张 A100-40GB、batch 1 的端到端延迟中位数。

| 模型 | RoboPulse 每对 | RoboRewardBench 每段 |
| --- | --- | --- |
| JevAny-Robotics-4B，flash-linear-attention + causal-conv1d | **185 ms** | 279 ms |
| JevAny-Robotics-4B，PyTorch 参考内核 | 294 ms | 392 ms |
| Robo-Dopamine-8B（vLLM） | 316 ms | – |
| RoboReward-4B / 8B，单次前向 | – | **142 / 160 ms** |
| RoboReward-8B，正常生成 | – | 361 ms |
| 27B Jev 类模型，参考内核，2 卡 | 732–838 ms | – |

我们比 Robo-Dopamine-8B 和 27B 模型都快，但在 RoboRewardBench 上比 RoboReward 慢。原因是我们按原分辨率、全部帧输入，每段视频的视觉 token 中位数是 1,433，是 RoboReward（564）的约 2.5 倍。把输入压到 JevAny 默认的 8 帧、512×512 总像素，延迟能降到 187 ms，但 RoboRewardBench MAE 从 0.443 涨到 0.607，RoboArena 从 0.687 涨到 1.289。

# 7 这些结论的边界

## 7.1 细粒度进度还差很远

RoboPulse 上 71.1% 对 Robo-Dopamine 的 90.1%，是全文最大的差距。我们检查过这不是测试重叠造成的：只看 episode 不在 Robo-Dopamine 公开训练数据里的 939 题，它仍然是 87.5%，我们 68.8%。分来源看，它在全部 9 个来源上都领先。我们的理解是，小幅进度变化需要大量同格式的密集监督，而公开数据里这类题我们只用了一部分。

RoboPulse++ 上还有一个阈值问题。训练时上升 / 下降是 1:1，测试里 81% 是上升。默认阈值 0.5 下准确率只有 60.4%，低于「全判上升」的 81.1%。排序质量（AUROC）和平衡准确率才是有意义的指标，实际用时应该在自己的数据上定阈值。

## 7.2 RoboRewardBench 的合成题

RoboRewardBench 的 OXE 部分有一批构造题：换掉任务指令的成功视频、从中间截断的视频。RoboReward 正是在同样构造的数据上训练的，在这两类题上明显更好（MAE 0.223 对 0.297、0.418 对 0.584）。我们常把截断视频判成「快做完了」。另外几个小子集（如 utokyo\_xarm\_bimanual、stanford\_hydra）误差很高，训练里这些子集的打分题只有几百道。

## 7.3 其他发现

- **多视角**：同时看 2–3 个相机的题仍是最弱的一类。RoboPulse++ 上，单视角 AUROC 0.758，两视角 0.646，三视角 0.672。
- **成败判断和偏好**：JevAny-Robotics-Public 上 OneJev-27B 更好。而且 5 个成败集合里有 2 个的 train split 在我们的训练数据里，这一项对我们是偏有利的。
- **速度**：全量媒体的代价是每段视频的 token 多，再加上 Qwen3.5 的线性注意力层比 Qwen3-VL 的注意力慢，所以 RoboRewardBench 上我们每段 279 ms，RoboReward 单次前向 142–160 ms。
- **机器人之外的校准变差**：在通用决策题上，准确率大致保持在起点 JevAny-Qwen3.5-4B 的水平，但概率变得过度自信。通用场景请用 [JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) 本身。

# 8 第二轮训练的线索

这些弱点能不能靠数据补？我们做了一轮续训（内部叫 v0.3.1）：从本文的模型出发，加入 22,051 道新的真机题（AIRoA、AgiBot、RoboCOIN、RoboMIND、RH20T），并回放一部分旧数据。结果是有涨有跌。

| 测试 | 指标 | 本文模型 | 续训后 | 差 \[95% CI\] |
| --- | --- | --- | --- | --- |
| RoboPulse | acc | 71.1 | 72.2 | +1.2 \[0.0, +2.4\] |
| RoboPulse++ 两视角 | AUROC | 0.646 | 0.770 | +0.124 \[−0.005, +0.230\] |
| RoboPulse++ 单视角 | AUROC | 0.758 | 0.726 | −0.031 \[−0.056, −0.009\] |
| RoboPulse++ 三视角 | AUROC | 0.672 | 0.586 | −0.086 \[−0.186, +0.021\] |
| RoboPulse++ 全部 | AUROC | 0.728 | 0.720 | −0.008 \[−0.035, +0.022\] |
| RoboRewardBench | MAE ↓ | 0.443 | 0.458 | +0.015 \[−0.018, +0.032\] |
| JevAny-Robotics-Public | skill | 0.567 | 0.590 | +0.023 \[+0.001, +0.046\] |

两视角的提升最大，但置信区间刚好碰到 0，而且主要来自一类构造题（AIRoA 的反向动作）。三视角变差，和 RH20T 的构造题在开发集上越训越反一致。单视角显著变差。失败和成败类的题显著变好。默认阈值下，续训版把 87.5% 的题都判成上升。

所以我们仍然发布本文的版本。这轮续训说明的是：加数据能把能力搬到特定的弱项上，但构造题带来的提升不一定能迁移，也可能让别的视角变差。

# 9 实验细节

**同一套 harness。** 所有模型都在我们的 harness 里跑，题目和媒体完全相同，每个模型都答完了全部题。置信区间是配对 cluster bootstrap（按 episode 或 task group 重采样）的 95% 区间。测试集：RoboPulse 1,800 对，RoboPulse++ 7,720 对，RoboRewardBench 2,831 段（与 HF 上 [teetone/RoboReward](https://huggingface.co/datasets/teetone/RoboReward) 的 test split 逐条核对一致），JevAny-Robotics-Public 13 个题型。

**专用模型怎么答它们没被训练过的题。** 下面的规则都是打分前定下的。

- RoboReward：只做整段 1–5 打分。成对题用两帧视频（前，后）的期望分减去（后，前）的期望分。它不是为成对判断训练的，所以 RoboPulse 上接近随机并不说明它不会判进度。
- Robometer：用去偏后的逐帧进度；RoboRewardBench 的 1–5 切点在 val 上拟合。
- Robo-Dopamine：复刻 [PRM-as-a-Judge](https://arxiv.org/abs/2603.21669) 的 GRMInference（vLLM）。RoboPulse++ 上事先定 incremental 为主读出，另报 forward 和去位置偏置两种；参考起点是 episode 第一帧，没有目标图时用空白图。RoboRewardBench、策略排序和成败题用首帧到末帧的进度，切点在 RoboReward val 上拟合，拟合出的切点让它从不预测 5 分。超过 8,192 token 的请求（23,559 个里有 146 个）把图像缩到 ≤ 76,800 像素后重跑。

**RoboReward 的数字和榜单对不上。** 我们测得 RoboReward-8B / 4B 的 MAE 是 0.358 / 0.407，论文和榜单是 0.665 / 0.845。测试集是同一个；自由生成和预填读出分别是 0.357 和 0.358；零样本的 Qwen3-VL-4B 在我们这里是 0.939，论文是 1.032。另一篇独立工作 [RoboRMBench](https://arxiv.org/abs/2609.05401) 报告的 RoboReward 误差也远低于榜单。我们没能拿到榜单的评测设置，所以本文只报自己实测的数字，不和榜单比。

**Robo-Dopamine 的重叠检查。** 用我们对 GRM 数据做的去重结果，标出与 GRM episode 重叠的 RoboPulse 题（861 / 1,800）。重叠题对每个模型都更容易，大约高 5 个点，不重叠的结果见上文。

**校准指标。** NLL 和 Brier 在 5 个分数上算；ECE 用 10 个等宽的置信度桶，置信度取最高分选项的概率。Robo-Dopamine 和 Robometer 不直接输出每个分数的概率，不参与校准比较。

**策略排序。** RoboArena 里同一 session、人工分数不同的策略对，共 134 个 session、805 对。每个策略取各视角期望分的均值，和人工排序一致算对，平局算半个。

**速度。** 单张 A100-40GB、batch 1，端到端延迟取中位数。本文的准确率都用 PyTorch 参考内核算；换成 flash-linear-attention + causal-conv1d 后，准确率在数值误差范围内不变（RoboPulse 71.4%，RoboRewardBench MAE 0.446）。27B 模型用参考内核、分在 2 张卡上。

# 10 怎么用

模型在 Hugging Face 上：[SimpleJev/JevAny-Robotics-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Robotics-4B-LoRA)。评测集：[SimpleJev/JevAny-Robotics-Public](https://huggingface.co/datasets/SimpleJev/JevAny-Robotics-Public)。代码：[SimpleJev/JevAny](https://github.com/SimpleJev/JevAny)。

请用 <code>--media full</code> 启动服务（或设 <code>JEVANY_MEDIA=full</code>）。这样图像用原分辨率，视频用全部帧，和训练、评测时一致。我们在 300 段 RoboRewardBench 视频和 300 对 RoboPulse 题上核对过，用它得到的概率和本文报告的完全一致。JevAny 的默认设置会压缩媒体，效果明显更差（见上文）。这个选项还在合并进 JevAny 主分支的过程中，安装方式以模型卡为准。

要拿它做二分类判断（比如进度是否上升），建议在自己的数据上定阈值，不要直接用 0.5。

# 参考文献

1\. Ji, Y., Liu, Y., Tan, H., et al. *PRM-as-a-Judge: A Dense Evaluation Paradigm for Fine-Grained Robotic Auditing*. arXiv:2603.21669, 2026. 提出 RoboPulse. [https://arxiv.org/abs/2603.21669](https://arxiv.org/abs/2603.21669)

2\. Lee, T., Wagenmaker, A., Pertsch, K., et al. *RoboReward: General-Purpose Vision-Language Reward Models for Robotics*. arXiv:2601.00675, 2026. 提出 RoboReward、RoboRewardBench、RoboArena. [https://arxiv.org/abs/2601.00675](https://arxiv.org/abs/2601.00675)

3\. Tan, H., Chen, S., Xu, Y., et al. *Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation*. arXiv:2512.23703, 2025. [https://arxiv.org/abs/2512.23703](https://arxiv.org/abs/2512.23703)

4\. Pacaud, P., Garcia, R., Chen, S., & Schmid, C. *Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models*. arXiv:2512.01946, 2025. [https://arxiv.org/abs/2512.01946](https://arxiv.org/abs/2512.01946)

5\. Fang, Y., Dong, W., Liu, J., et al. *DenseReward: Dense Reward Learning via Failure Synthesis for Robotic Manipulation*. arXiv:2607.13033, 2026. [https://arxiv.org/abs/2607.13033](https://arxiv.org/abs/2607.13033)

6\. Ye, Z., Lu, W., Ye, M., et al. *RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction*. arXiv:2505.12224, 2025. [https://arxiv.org/abs/2505.12224](https://arxiv.org/abs/2505.12224)

7\. Zeng, X., Zhou, X., Li, Y., et al. *Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols*. arXiv:2512.02787, 2025. 提出 ViFailback. [https://arxiv.org/abs/2512.02787](https://arxiv.org/abs/2512.02787)

8\. Sermanet, P., Ding, T., Zhao, J., et al. *RoboVQA: Multimodal Long-Horizon Reasoning for Robotics*. arXiv:2311.00899, 2023. [https://arxiv.org/abs/2311.00899](https://arxiv.org/abs/2311.00899)

9\. Xu, Y., Jin, H., Zhou, R., et al. *Robo-Dopamine 2.0: History-Conditioned and OOD-Aware Process Reward Modeling for Robotic Manipulation*. arXiv:2608.15680, 2026. [https://arxiv.org/abs/2608.15680](https://arxiv.org/abs/2608.15680)

10\. Liang, A., Korkmaz, Y., Zhang, J., et al. *Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons*. arXiv:2603.02115, 2026. [https://arxiv.org/abs/2603.02115](https://arxiv.org/abs/2603.02115)

11\. Liu, Y., Shen, Y., Chen, R., et al. *PRM-as-a-Judge 1.5: A Toolkit for Robot Process Assessment*. arXiv:2608.14284, 2026. 提出 RoboPulse++. [https://arxiv.org/abs/2608.14284](https://arxiv.org/abs/2608.14284)

12\. Jeung, W., Yoon, S., Hong, H., et al. *Same Trajectory, Contradictory Rewards (RoboRMBench): Paraphrase Fragility in Vision Language Reward Models*. arXiv:2609.05401, 2026. [https://arxiv.org/abs/2609.05401](https://arxiv.org/abs/2609.05401)`
};
