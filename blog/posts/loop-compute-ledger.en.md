A looped language model reuses the same block of layers. The parameter count stays fixed while the computation can be made deeper, one pass at a time. Training is expensive: the [Parcae](https://arxiv.org/abs/2604.12946) recipe runs about eight loops per sample on average. We trained more than sixty small models on one laptop GPU to see how many of those loops are actually doing work, and what the saved compute can buy.

The models have 86.5M and 144M parameters. Training uses at most 2B tokens from FineWeb-Edu.

> Loss is validation cross-entropy, and lower is better. At this scale, a loss drop of 0.01 is roughly equivalent to training on 2% to 5% more data. Rerunning a recipe with a new seed moves the loss by 0.001 to 0.01, so we treat any gap inside 0.01 as a tie. The figures are redrawn from the tables below.

# Takeaway

A model trained with the original Parcae recipe collects 94% of its gain by loop 4. After that the hidden state almost stops moving, and the later loops mostly idle.

We changed the recipe in two places. Every sample runs exactly four loops, and every loop receives a gradient. From loop 2 on, the loop block attends only to the most recent 64 tokens. On the same training data, this lean recipe matches the original, and it is slightly better on the smaller model. Training is 1.25 to 1.4 times faster, generation is 2 to 3 times faster, and the KV cache is about one third as large.

The advantage has edges. It shrinks as training gets longer, and at 2B tokens the original recipe pulls slightly ahead by looping more. At this small scale, a plain Transformer with the same parameter count and the same training time beats every looped variant.

# The model and the recipe

A plain Transformer uses each layer once. A looped model turns the middle layers into a block and sends the same sequence through that block again. Each extra loop adds depth without adding parameters. Our layout matches Parcae: two prelude layers and two coda layers run once, and a two-layer loop block repeats for T loops. Every loop reads the prelude output again.

Each extra loop costs another forward pass, and during training it can cost a backward pass too. Parcae is the strongest public training recipe we compared against, so the rest of this note calls it the original recipe. It draws a loop count for each sample, with mean 8. To save memory and compute, only the last four loops receive gradients; the earlier loops are forward-only. At inference the usual choice is 8 to 12 loops.

![The model, and the three training recipes compared here. The original-recipe row is drawn for a sample that drew 9 loops.](img/loop/figure-1.svg)

(The prelude and the coda each run once. The two-layer loop block repeats for T loops, and each loop reads the prelude output again.)

The question is an accounting one. Is a mean of eight loops necessary, is it worth training only the last four, and does every loop need the full context?

# After four loops

Start with what an original-recipe model does at inference. Take the 144M model trained on 1B tokens and evaluate it at 1 through 32 loops. From 1 loop to 4, loss falls from 3.50 to 3.26. From 4 to 8 it falls another 0.016. Past 8 loops the curve is flat. The first four loops collect 94% of the gain between 1 and 8. All seven original-recipe models we trained at this size behave this way.

![Validation loss of an original-recipe model as the number of inference loops changes.](img/loop/figure-2.svg)

(144M parameters, 1B training tokens, two seeds. The horizontal axis is the loop count on a log scale. Gray lines are the two seeds; navy is their mean.)

| Inference loops | Seed 43 | Seed 44 | Mean |
| --- | --- | --- | --- |
| 1 | 3.5066 | 3.4944 | 3.5005 |
| 2 | 3.3393 | 3.3357 | 3.3375 |
| 3 | 3.2840 | 3.2822 | 3.2831 |
| 4 | 3.2622 | 3.2610 | 3.2616 |
| 6 | 3.2487 | 3.2478 | 3.2483 |
| 8 | 3.2463 | 3.2451 | 3.2457 |
| 12 | 3.2458 | 3.2446 | 3.2452 |
| 16 | 3.2459 | 3.2447 | 3.2453 |
| 24 | 3.2461 | 3.2449 | 3.2455 |
| 32 | 3.2461 | 3.2449 | 3.2455 |

The hidden state tells the same story. Each new loop changes it by about half to two thirds as much as the previous loop, and by loop 12 the change is on the order of bf16 rounding. The model settles at a fixed state. Further loops do not move it. That state is a fixed point. Of the eight loops the original recipe runs on average, the later ones barely affect the output, and they cost as much as the early ones.

# Two changes

That measurement suggests two changes, which together we call the lean recipe. First, train with exactly four loops, and backpropagate through all four. There is no random loop count and no forward-only prefix. Second, the loop block sees the full context on loop 1 only. Loops 2 through 4 see the most recent 64 tokens. The prelude and the coda are unchanged.

On the same training data, four loops of the lean recipe match the original recipe at its best loop count, which is 12. For the 144M model trained on 1B tokens, the lean recipe is 0.005 lower on both seeds, inside the noise. For the 86.5M model trained on 0.5B tokens, it is better on all three seeds, by 0.025 on average. Training is also about 1.4 times faster. The next two sections say why neither change gives up accuracy. The speed accounting comes after that.

| Model | Data | Seed | Original (best loop count) | Fixed 4 | Lean |
| --- | --- | --- | --- | --- | --- |
| 86.5M | 0.5B tokens | 43 | 3.5502 | 3.5396 | 3.5319 |
| 86.5M | 0.5B tokens | 44 | 3.5475 | 3.5419 | 3.5201 |
| 86.5M | 0.5B tokens | 45 | 3.5537 | 3.5440 | 3.5232 |
| 144M | 1B tokens | 43 | 3.2458 | 3.2433 | 3.2413 |
| 144M | 1B tokens | 44 | 3.2446 | 3.2407 | 3.2392 |
| 144M | 2B tokens | 43 | 3.0937 | — | 3.1024 |
| 144M | 2B tokens | 44 | 3.0927 | — | 3.0981 |

Validation loss. The original recipe is scored at its best count from 1 to 32 loops, which is 12 in every case. The other two recipes are scored at 4 loops. Recipes in the same row share a seed, so they share initialization and data order. The test set agrees on the direction.

# Why four loops are enough

We measured how much each loop moves the hidden state. On the original recipe, the change halves at loops 2, 3, and 4, and the state settles. On the lean recipe, those three loops keep making large edits. The state does not converge.

![How much each loop moves the hidden state.](img/loop/figure-3.svg)

(Relative change is the size of the difference across a loop, divided by the size of the state entering that loop. The two main models have 144M parameters, were trained on 2B tokens, and are averaged over 32 validation segments. The fixed-4 full-attention model was trained on 1B tokens.)

| Model | Loop 2 | Loop 3 | Loop 4 |
| --- | --- | --- | --- |
| Original (2B tokens) | 0.37 | 0.18 | 0.09 |
| Lean (2B tokens) | 0.62 | 0.56 | 0.63 |
| Fixed 4, full attention (1B tokens) | 0.70 | 0.54 | 0.46 |

The difference comes from which loops receive a gradient. The original loop count is random with mean 8, but only the last four loops are trained. Only about one sample in ten draws 4 loops or fewer, and only those samples train the loops that start from the initial state. Every other sample trains loops that are already near the fixed point. The model learns to converge quickly and then stop. With four fixed loops, every step from the initial state to the output has a gradient, so the model learns to use each of them.

Two controls support this, each run once. At a matched training time, four loops with a gradient on every loop beat six loops whose first three are forward-only, by 0.023 loss. The extra loops are worth less than training every loop. In the other control, the original random schedule is kept, but only the last loop receives a gradient. The gain from looping, measured as the loss drop from 1 loop to 8, shrinks from 0.23 to 0.11. The model uses the loop half as much.

Fixing the count has a cost: the model can only run four loops. Fewer or more loops are clearly worse. Eight loops are 0.07 worse than four, and one loop is 0.8 worse. An original-recipe model can run anywhere from 6 to 32 loops with almost the same loss. The lean recipe gives up the usual selling point of a looped model, which is spending more inference loops for a better answer. The curve above says the original recipe was already limited on that point: past eight loops there is nothing left to buy. [Huang et al. (2026)](https://arxiv.org/abs/2610.06833) report both facts on models from 100M to 1.6B parameters.

![Validation loss against the number of inference loops, for both recipes.](img/loop/figure-4.svg)

(144M parameters, 1B tokens, mean of two seeds. The lean recipe is best at 4 loops, 0.005 below the original recipe's best score. Every other loop count is clearly worse.)

| Inference loops | Original, seed 43 | Seed 44 | Lean, seed 43 | Seed 44 |
| --- | --- | --- | --- | --- |
| 1 | 3.5066 | 3.4944 | 4.0478 | 4.0400 |
| 2 | 3.3393 | 3.3357 | 3.6453 | 3.6365 |
| 3 | 3.2840 | 3.2822 | 3.4292 | 3.4358 |
| 4 | 3.2622 | 3.2610 | 3.2413 | 3.2392 |
| 6 | 3.2487 | 3.2478 | 3.4620 | 3.4139 |
| 8 | 3.2463 | 3.2451 | 3.3057 | 3.3114 |
| 12 | 3.2458 | 3.2446 | 3.3387 | 3.3439 |
| 16 | 3.2459 | 3.2447 | 3.3495 | 3.3513 |
| 24 | 3.2461 | 3.2449 | 3.3538 | 3.3521 |
| 32 | 3.2461 | 3.2449 | 3.3541 | 3.3518 |

# How far the loop needs to look

The second change comes from an ablation. On the 86.5M model, we removed one piece of compute at a time from the loop block of the fixed-4 model, to see which pieces can go.

![Change in validation loss after removing one piece of the loop block.](img/loop/figure-5.svg)

(The baseline is four fixed loops with full attention. Negative means better. 86.5M parameters, 0.5B tokens, one run each. The gray band is the seed range, about plus or minus 0.01.)

| Change to the loop block | Validation loss | Versus baseline | Train time |
| --- | --- | --- | --- |
| Baseline: fixed 4, full attention | 3.5396 | 0 | 1.99 h |
| Loops 2–4 see the last 64 tokens | 3.5319 | −0.008 | 1.80 h |
| All 4 loops see the last 64 tokens | 3.5317 | −0.008 | 1.70 h |
| Loops 2–4 see the last 256 tokens | 3.5385 | −0.001 | 1.81 h |
| Loops 2–4 at half width (heads and MLP) | 3.5684 | +0.029 | 1.76 h |
| Loops 2–4 drop attention, MLP only | 3.5870 | +0.047 | 1.60 h |
| Loops 2–4 reuse the KV from loop 1 | 3.5705 | +0.031 | 1.93 h |
| Only 2 loops | 3.5653 | +0.026 | 1.56 h |
| Reference: original recipe | 3.5502 | +0.011 | 2.59 h |
| Reference: plain Transformer, 6 layers | 3.5540 | +0.014 | 1.33 h |

Only the shorter attention window is free. Loops 2 through 4 can see the last 64 tokens or the last 256 without a worse loss, and even restricting loop 1 to 64 tokens does not hurt. Every other cut is clearly worse: half width by 0.029, MLP only by 0.047, reusing loop 1's KV by 0.031, and stopping at two loops by 0.026.

So each loop still needs its full width and its own attention. It does not need distant tokens. The prelude and the coda, four layers of full attention, carry the long-range information. The 64-token window reproduced on the other two seeds and on the 144M model. Keeping full attention on loop 1, as the lean recipe does, is the conservative choice.

There is a cost. Tokens that copy content from more than 500 positions earlier have about 0.02 higher loss, and they are about 3% of tokens. The other tokens have slightly lower loss. Downstream scores are too noisy at this scale to separate the two recipes.

# The ledger

Together, the two changes raise training throughput on the 144M model from 35k to 49k tokens per second, a factor of 1.42. Part of that is an inefficiency in the current original-recipe implementation. It draws a loop count per sample, so every sample in a batch waits for the longest one. Padding burns 15% to 20% of the compute. Drawing one count for the whole batch would make the original recipe about 15% faster, and the lean advantage would fall to 1.25 times.

Faster training can be spent on a lower loss. We gave the original recipe less data so that wall-clock time matched, and both runs finished the learning-rate decay. At 5.7 hours the lean loss is 0.11 lower. At 11.3 hours it is 0.06 lower. On the 86.5M model, 1.8 hours gives a gap of 0.18. All six paired repeats agree on the direction. This comparison uses the original implementation as it stands. Fixing the padding inefficiency would, by a rough estimate, close a third to a half of the gap.

![Validation loss at a matched training time.](img/loop/figure-6.svg)

(Each point is a finished training run. The original-recipe points use different amounts of data. Where several seeds exist, the point is their mean. The 86.5M panel also includes a 6-layer Transformer with the same parameter count and no loop.)

| Model | Train time | Seed | Original (data) | Lean (data) | Gap |
| --- | --- | --- | --- | --- | --- |
| 86.5M | about 1.8 h | 43 | 3.6979 (345M tokens) | 3.5319 (500M tokens) | −0.166 |
| 86.5M | about 1.8 h | 44 | 3.7024 | 3.5201 | −0.182 |
| 86.5M | about 1.8 h | 45 | 3.7081 | 3.5232 | −0.185 |
| 144M | about 5.7 h | 43 | 3.3521 (704M tokens) | 3.2413 (1B tokens) | −0.111 |
| 144M | about 5.7 h | 44 | 3.3547 | 3.2392 | −0.116 |
| 144M | about 11.3 h | 43 | 3.1654 (1.41B tokens) | 3.1024 (2B tokens) | −0.063 |

Other points. The original 144M recipe takes 8.07 hours for 1B tokens (loss 3.2452) and 15.98 hours for 2B tokens (loss 3.0932). The original 86.5M recipe takes 2.61 hours for 0.5B tokens (loss 3.5505). A plain Transformer at 86.5M, seed 43, takes 1.33 hours for 0.5B tokens (loss 3.5540) and 1.99 hours for 0.75B tokens (loss 3.4673).

Generation is where the gap is larger. A looped model stores a separate KV cache for each loop, so eight loops means eight copies. The lean recipe stores four loops, and three of them store only the last 64 tokens. Timed on the 144M layout, the lean recipe generates a token 2.3 to 3.1 times faster than the original recipe at 8 loops, with about one third of the cache. At a context of 16k tokens it is about as fast as a plain 6-layer Transformer of the same size, and the caches match, because both have only six layers that see the full context. At 2,000 tokens the plain Transformer is still about 20% faster.

![Time to generate one token, and the size of the KV cache.](img/loop/figure-7.svg)

(Measured on the 144M layout, 16k tokens of context, one sequence at a time. The fixed-4 bar changes only the loop count, so the two savings can be read separately.)

| Layout | 2k tokens, batch 1 | 2k tokens, batch 64 | KV cache | 16k tokens, batch 1 | 16k tokens, batch 8 | KV cache |
| --- | --- | --- | --- | --- | --- | --- |
| Original, 8 loops | 4.66 ms | 20.17 ms | 120 MB | 25.07 ms | 27.01 ms | 960 MB |
| Fixed 4 | 3.09 ms | 12.47 ms | 72 MB | 15.26 ms | 16.23 ms | 576 MB |
| Lean | 2.00 ms | 7.01 ms | 37 MB | 8.11 ms | 8.63 ms | 289 MB |
| All 4 loops, window 64 | 1.67 ms | 5.31 ms | 26 MB | 5.80 ms | 6.12 ms | 194 MB |
| Plain Transformer, 6 layers | 1.57 ms | 6.39 ms | 36 MB | 7.87 ms | 8.13 ms | 288 MB |

The timing uses bf16, a static KV cache, and a CUDA graph around the whole step. Local-attention loops use a 64-slot ring cache. The time depends on the layout, not on the weight values, so the weights are random. The cache sizes are for a batch of one.

This is our own benchmark, not a measurement inside an inference engine such as vLLM. An original-recipe model can also share one KV cache across loops, which is what Huginn does. Doing that with a fixed loop count hurts quality sharply (Huang et al., 2026). We did not measure that variant.

# Where this stops

## The longer the run, the smaller the gap

At matched wall-clock time, the gap falls from 0.18 on the smaller model, to 0.11 at 5.7 hours on the larger one, to 0.06 at 11.3 hours. Matching the amount of data instead of the time gives the same trend. At 144M parameters and 2B tokens, the original recipe at 12 loops is 0.005 to 0.009 better than the lean recipe, on both seeds. It spends three times the compute to get there. If it is also limited to 4 loops, it is still 0.013 to 0.014 worse.

More training seems to make extra loops more valuable, which matches the scaling reported in the Parcae paper. Whether fixing the count at more than four loops keeps the advantage is something we tried once, on the smaller model: eight fixed loops beat the original recipe by 0.034. That still needs a longer run.

## At this scale a plain Transformer is the better buy

On the 86.5M model we also trained a 6-layer Transformer with the same parameter count and no loop. At about two hours it has a lower loss than every looped variant, 0.03 to 0.04 below the lean recipe. At this size the loop itself is not yet worth it. The lean recipe closes part of the gap and does not reverse it. This comparison was run once, on the smaller model.

## The scale is small

Everything here is at most 144M parameters and 2B tokens, on one dataset and one laptop GPU. Most of the parameters sit in the embedding table. The compute-heavy part is 19.4M parameters in the smaller model and 43.7M in the larger one. These runs do not say what happens at a larger model or a longer training run.

# Prior work

Most of the phenomena here have been seen in other settings.

- Drawing a random loop count and training only the last few loops comes from [Huginn](https://arxiv.org/abs/2502.05171) (Geiping et al., 2025). Parcae keeps that choice. Both Parcae and Huang et al. (2026) report that loss stops falling once inference uses more loops than the training average.
- Fixed-count training is common. [Ouro](https://arxiv.org/abs/2510.25741) and [LT2](https://arxiv.org/abs/2605.20670) both fix the count at 4. Huang et al. compare the two schedules from 100M to 1.6B parameters: at a matched mean, a fixed count beats a random one, but the model gets worse if inference uses a different count. The appendix of [Loop growth](https://arxiv.org/abs/2609.19107) also compares random and fixed counts.
- [Iso-depth](https://arxiv.org/abs/2604.21106) compares, at matched compute, training only the last few loops against training every loop. Training only the last loops wins there, because the saved compute buys more data. We compare a different split of the same compute: more loops with gradients on only some of them, against fewer loops with a gradient on every one.
- Local attention on later loops is used by [LT2](https://arxiv.org/abs/2605.20670), [PLT](https://arxiv.org/abs/2510.24824), and [LPT](https://arxiv.org/abs/2610.02383). PLT and LPT let later loops see 64 tokens while also reading the global KV from the first loop. Sharing the first loop's KV directly is the [MoR](https://arxiv.org/abs/2507.10524) choice. PLT reports that sharing KV alone loses accuracy, and we see the same thing.

What we add is three measurements in one setup: a joint account of training time, generation speed, and loss, including implementation overhead; a loop-by-loop look at the hidden state that explains why a fixed count does not lose; and an ablation of which compute inside the loop can be removed.

# Setup

Data. FineWeb-Edu, read in order. The validation set and the test set are held-out shards.

Layout. Two prelude layers, a two-layer loop block, and two coda layers. Each loop reads the prelude output, which is the Parcae injection. The vocabulary has 32,768 tokens.

Two sizes. 86.5M: width 512, 4 heads, MLP 2048, and 19.4M parameters outside the embedding. 144M: width 768, 6 heads, MLP 3072, and 43.7M parameters outside the embedding.

Training. Sequence length 2048, about 520k tokens per step. The optimizer is Muon, with learning rate 0.008, constant for the first half and then linear to zero. That is Parcae's default. Every recipe uses this setting; nothing was tuned per recipe.

Original recipe. The loop count of each sample is drawn from a Poisson distribution with mean 8. Backpropagation covers at most the last four loops.

Local attention. A causal sliding window of 64 tokens, implemented with PyTorch FlexAttention.

Repeats. The main comparison uses 3 seeds at 86.5M and 2 seeds at 144M. Within a seed, every recipe shares initialization and data order. Each ablation is a single run.

Evaluation. Every model is scored at 1, 2, 3, 4, 6, 8, 12, 16, 24, and 32 loops. Every number is a finished run through the learning-rate decay, not a midpoint checkpoint.

Hardware. One laptop RTX 5090 (24 GB), bf16, torch.compile.

The numbers come from the evaluation record written at the end of each run. The raw timing and per-loop diagnostics are in the project log.

# References

Geiping et al. (2025). Huginn. arXiv. [https://arxiv.org/abs/2502.05171](https://arxiv.org/abs/2502.05171)

Huang et al. (2026). arXiv. [https://arxiv.org/abs/2610.06833](https://arxiv.org/abs/2610.06833)

Iso-depth (2026). arXiv. [https://arxiv.org/abs/2604.21106](https://arxiv.org/abs/2604.21106)

Loop growth (2026). arXiv. [https://arxiv.org/abs/2609.19107](https://arxiv.org/abs/2609.19107)

LPT (2026). arXiv. [https://arxiv.org/abs/2610.02383](https://arxiv.org/abs/2610.02383)

LT2 (2026). arXiv. [https://arxiv.org/abs/2605.20670](https://arxiv.org/abs/2605.20670)

MoR (2025). arXiv. [https://arxiv.org/abs/2507.10524](https://arxiv.org/abs/2507.10524)

Ouro (2025). arXiv. [https://arxiv.org/abs/2510.25741](https://arxiv.org/abs/2510.25741)

Parcae (2026). arXiv. [https://arxiv.org/abs/2604.12946](https://arxiv.org/abs/2604.12946)

PLT (2025). arXiv. [https://arxiv.org/abs/2510.24824](https://arxiv.org/abs/2510.24824)
