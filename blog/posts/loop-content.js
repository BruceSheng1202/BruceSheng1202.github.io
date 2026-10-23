window.LOOP_CONTENT = { en: String.raw`A looped language model reuses the same block of layers. The parameter count stays fixed while the computation can be made deeper, one pass at a time. Training is expensive: the [Parcae](https://arxiv.org/abs/2604.12946) recipe runs about eight loops per sample on average. We trained more than sixty small models on one laptop GPU to see how many of those loops are actually doing work, and what the saved compute can buy.

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

PLT (2025). arXiv. [https://arxiv.org/abs/2510.24824](https://arxiv.org/abs/2510.24824)`, zh: String.raw`循环语言模型把同一组层反复使用，参数不变，计算却能一轮轮加深。代价是训练很贵。[Parcae](https://arxiv.org/abs/2604.12946) 的训练配方让每条样本平均循环 8 轮。我们在一张笔记本 GPU 上训练了 60 多个小模型，想弄清楚这 8 轮里有多少是必要的，省下来的算力又能换来什么。

模型 8650 万和 1.44 亿参数，训练数据最多 20 亿 token，数据是 FineWeb-Edu。

> 文中的 loss 是验证集上的交叉熵，越低越好。在我们的规模上，loss 低 0.01，大约相当于多训 2% 到 5% 的数据。同一个配方换一个随机种子重新训练，loss 会有 0.001 到 0.01 的出入，所以差距在 0.01 以内的，一律算作持平。图是按文末的表重新画的。

# 结论

Parcae 原配方训出来的模型，循环 4 轮就拿到了 94% 的收益，之后隐状态几乎不再变化，后面几轮基本在空转。

我们把训练改了两处：每次固定循环 4 轮，每一轮都计算梯度；循环块从第 2 轮起只看最近 64 个 token。同样的训练数据下，效果和原配方持平，较小的模型上还略好。训练快 1.25 到 1.4 倍，生成文本快 2 到 3 倍，KV 缓存约为原来的三分之一。

这些优势有边界。训练越久，优势越小，训到 20 亿 token 时原配方靠多循环几轮略微反超。而在我们这么小的规模上，参数量相同的普通 Transformer 用同样的训练时间，效果比所有循环版本都好。

# 循环模型和它的训练配方

普通 Transformer 的每一层只用一次。循环语言模型把中间几层做成一个循环块，让同一个序列反复穿过它。每多循环一轮，就相当于多了一层深度，参数却没有增加。我们用的结构和 Parcae 相同：开头 2 层和结尾 2 层各跑一次，中间 2 层组成循环块，重复 T 轮，每一轮都会重新读入开头两层的输出。

循环块多跑一轮，就多一份计算，训练时还要加上反向传播。Parcae 是目前公开的效果较好的循环模型训练配方，下文称为原配方。它给每条样本随机抽一个循环轮数，平均 8 轮；为了省显存和计算，只有最后 4 轮计算梯度，前面的轮次只做前向。推理时一般循环 8 到 12 轮。

![模型结构，以及本文比较的三种训练方式。原配方这一行画的是一条抽到 9 轮的样本。](img/loop/figure-1.svg)

(开头两层和结尾两层各跑一次。循环块两层重复 T 轮，每一轮重新读入开头两层的输出。)

这篇笔记逐项检查这笔算力：平均 8 轮是不是必要，只训练最后 4 轮划不划算，每一轮是不是都要看完整的上下文。

# 4 轮之后

先看原配方训出来的模型在推理时做了什么。我们取一个 1.44 亿参数、训练了 10 亿 token 的模型，让它分别循环 1 到 32 轮，测验证 loss。从 1 轮到 4 轮，loss 从 3.50 降到 3.26；从 4 轮到 8 轮只再降了 0.016；8 轮以后几乎是一条平线。前 4 轮拿到了 1 到 8 轮全部收益的 94%。我们训练的 7 个这个规模的原配方模型都是如此。

![原配方模型在推理时循环不同轮数的验证 loss。](img/loop/figure-2.svg)

(1.44 亿参数、训练 10 亿 token，两个种子。横轴是推理轮数，用对数刻度。灰线是两个种子，海军蓝是平均。)

| 推理时循环轮数 | 种子 43 | 种子 44 | 平均 |
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

隐状态的变化也是这样。每多循环一轮，隐状态的改动大约只有上一轮的一半到三分之二，到第 12 轮已经小到和 bf16 的舍入误差相当。模型很快停在一个固定的状态上，再循环下去状态也不会变，数学上把这种状态叫作不动点。原配方训练时平均跑的 8 轮里，后面几轮对结果几乎没有贡献，花的算力却和前几轮一样多。

# 两处改动

据此，我们对原配方做了两处改动，合称精简配方。第一，训练时每次固定循环 4 轮，4 轮都计算梯度，不再随机抽轮数，也不再有只做前向的轮次。第二，循环块只在第 1 轮看完整的上下文，第 2 到第 4 轮只看最近 64 个 token；开头和结尾的 4 层不变。

同样的训练数据下，精简配方循环 4 轮的效果，和原配方在它最好的轮数（12 轮）下一样好。1.44 亿参数的模型训练 10 亿 token，两个种子上精简配方都低 0.005，在误差范围内；8650 万参数的模型训练 5 亿 token，三个种子上精简配方都更好，平均低 0.025。训练也快了约 1.4 倍。下面两节分别解释这两处改动为什么不损失效果，然后再算速度的账。

| 模型 | 训练数据 | 种子 | 原配方（最好的轮数） | 固定 4 轮 | 精简配方 |
| --- | --- | --- | --- | --- | --- |
| 8650 万参数 | 5 亿 token | 43 | 3.5502 | 3.5396 | 3.5319 |
| 8650 万参数 | 5 亿 token | 44 | 3.5475 | 3.5419 | 3.5201 |
| 8650 万参数 | 5 亿 token | 45 | 3.5537 | 3.5440 | 3.5232 |
| 1.44 亿参数 | 10 亿 token | 43 | 3.2458 | 3.2433 | 3.2413 |
| 1.44 亿参数 | 10 亿 token | 44 | 3.2446 | 3.2407 | 3.2392 |
| 1.44 亿参数 | 20 亿 token | 43 | 3.0937 | — | 3.1024 |
| 1.44 亿参数 | 20 亿 token | 44 | 3.0927 | — | 3.0981 |

验证 loss。原配方取 1 到 32 轮中最好的一档，都是 12 轮；另两种取 4 轮。同一行的几种配方用同一个种子，初始化和数据顺序相同。测试集上方向一致。

# 为什么 4 轮够用

我们测了两种模型在每一轮让隐状态改变了多少。原配方的模型，第 2、3、4 轮的改动每轮减半，很快停了下来。精简配方的模型，这三轮的改动一直很大，每一轮都在大幅改写隐状态，并没有收敛。

![每一轮让隐状态改变了多少。](img/loop/figure-3.svg)

(相对变化是这一轮前后隐状态之差的大小，除以上一轮隐状态的大小。两个主要模型都是 1.44 亿参数、训练 20 亿 token，在 32 段验证文本上取平均。只固定 4 轮、不加局部注意力的模型训练了 10 亿 token。)

| 模型 | 第 2 轮 | 第 3 轮 | 第 4 轮 |
| --- | --- | --- | --- |
| 原配方（20 亿 token） | 0.37 | 0.18 | 0.09 |
| 精简配方（20 亿 token） | 0.62 | 0.56 | 0.63 |
| 固定 4 轮、全注意力（10 亿 token） | 0.70 | 0.54 | 0.46 |

我们的理解是，差别来自训练时梯度经过了哪些轮次。原配方的轮数随机，平均 8 轮，但只有最后 4 轮有梯度。只有约十分之一的样本会抽到 4 轮以内，这时从初始状态出发的前几轮才会被训练到；其余样本训练的都是已经接近不动点的那几轮。模型于是学会了尽快收敛，然后停住。固定 4 轮时，从初始状态到输出的每一轮都有梯度，模型就学会了把每一轮都用上。

两个对照实验支持这个解释，各做了一次。第一个在相同的训练时间里比较两种用法：循环 4 轮、4 轮都算梯度，比循环 6 轮、前 3 轮不算梯度的 loss 低 0.023，多跑的那几轮不如让每一轮都参与训练。第二个保持原配方随机抽轮数的做法，但只让最后 1 轮算梯度，模型从循环中得到的收益，也就是从循环 1 轮到 8 轮 loss 的降幅，从 0.23 缩到了 0.11，模型对循环的利用少了一半。

固定轮数也有代价，模型只能循环 4 轮。少跑或多跑，loss 都明显变差，循环 8 轮时比 4 轮高 0.07，只循环 1 轮时高 0.8。原配方的模型则可以在 6 到 32 轮之间随意选，效果差不多。这等于放弃了循环模型的一个卖点，即推理时多循环几轮来换更好的结果。不过上面的曲线也说明，原配方在这一点上本来就很有限，超过 8 轮就没有收益了。[Huang et al.（2026）](https://arxiv.org/abs/2610.06833) 在 1 亿到 16 亿参数的模型上观察到了同样的两点。

![两种配方在推理时循环不同轮数的验证 loss。](img/loop/figure-4.svg)

(1.44 亿参数、训练 10 亿 token，两个种子的平均。精简配方在 4 轮时最好，比原配方最好的成绩还低 0.005；换成其他任何轮数都明显变差。)

| 推理时循环轮数 | 原配方 · 种子 43 | 种子 44 | 精简配方 · 种子 43 | 种子 44 |
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

# 循环块看多远

第二处改动来自一组削减实验。我们在 8650 万参数的模型上，从固定 4 轮模型的循环块里各削掉一部分计算，看哪些可以省。

![从循环块里削掉一部分计算之后，验证 loss 相对基准的变化。](img/loop/figure-5.svg)

(基准是固定 4 轮、全注意力。负数表示更好。8650 万参数，训练 5 亿 token，每种各训练一次。灰色区域是种子误差的范围，大约正负 0.01。)

| 循环块的改动 | 验证 loss | 相对基准 | 训练时间 |
| --- | --- | --- | --- |
| 基准：固定 4 轮、全注意力 | 3.5396 | 0 | 1.99 小时 |
| 第 2–4 轮只看最近 64 个 token | 3.5319 | −0.008 | 1.80 小时 |
| 4 轮都只看最近 64 个 token | 3.5317 | −0.008 | 1.70 小时 |
| 第 2–4 轮只看最近 256 个 token | 3.5385 | −0.001 | 1.81 小时 |
| 第 2–4 轮宽度减半（注意力头和 MLP 都减半） | 3.5684 | +0.029 | 1.76 小时 |
| 第 2–4 轮去掉注意力，只留 MLP | 3.5870 | +0.047 | 1.60 小时 |
| 第 2–4 轮复用第 1 轮的 KV | 3.5705 | +0.031 | 1.93 小时 |
| 只循环 2 轮 | 3.5653 | +0.026 | 1.56 小时 |
| 参照：原配方 | 3.5502 | +0.011 | 2.59 小时 |
| 参照：普通 Transformer（6 层，不循环） | 3.5540 | +0.014 | 1.33 小时 |

只有缩短注意力范围不伤效果。第 2 到第 4 轮只看最近 64 个或 256 个 token，loss 都没有变差；连第 1 轮也只看 64 个 token，结果也一样。其他削减都明显变差。后几轮的宽度减半，loss 高 0.029；后几轮去掉注意力只留 MLP，高 0.047；后几轮直接复用第 1 轮的 KV，高 0.031；只循环 2 轮，高 0.026。

所以循环块的每一轮都需要完整的宽度和自己的注意力，只是不需要看远处的 token，远处的信息由开头和结尾那 4 层完整注意力来处理。只看最近 64 个 token 的做法，在另外两个种子和 1.44 亿参数的模型上都复现了。精简配方保留了第 1 轮的完整注意力，是偏保守的选择。

代价也有。逐个 token 拆开看，需要从 500 多个 token 以前复制内容的 token，loss 高了约 0.02，这类 token 约占 3%；其余 token 的 loss 略低。下游任务的分数在这个规模上波动太大，看不出两种配方的差别。

# 算账

两处改动合起来，1.44 亿参数模型的训练速度从每秒 3.5 万 token 提高到 4.9 万，快 1.42 倍。这里面有一部分来自原配方现有实现的一个低效之处。它给每条样本单独抽轮数，同一批里的样本要陪着轮数最多的那条一起跑，15% 到 20% 的计算花在补齐上。如果改成整批抽同一个轮数，原配方自己能快约 15%，精简配方的优势降到 1.25 倍。

更快的训练可以换成更低的 loss。我们让原配方少训一些数据，使它的训练时间和精简配方相同，两边都完整走完学习率的衰减，再来比较。同样训练 5.7 小时，精简配方的 loss 低 0.11；同样训练 11.3 小时，低 0.06；8650 万参数的模型同样训练 1.8 小时，低 0.18。6 组重复的方向全部一致。这里比的是原配方的现有实现，如果修掉上面那个低效之处，我们粗略估计差距会缩小三分之一到一半。

![同样的训练时间，三种模型的验证 loss。](img/loop/figure-6.svg)

(每个点是一次完整的训练。原配方的几个点对应不同的训练数据量；有多个种子时画的是平均值。8650 万参数那一幅另加了参数量相同、不循环的 6 层 Transformer。)

| 模型 | 训练时间 | 种子 | 原配方（训练数据） | 精简配方（训练数据） | 差 |
| --- | --- | --- | --- | --- | --- |
| 8650 万参数 | 约 1.8 小时 | 43 | 3.6979（3.45 亿 token） | 3.5319（5 亿 token） | −0.166 |
| 8650 万参数 | 约 1.8 小时 | 44 | 3.7024 | 3.5201 | −0.182 |
| 8650 万参数 | 约 1.8 小时 | 45 | 3.7081 | 3.5232 | −0.185 |
| 1.44 亿参数 | 约 5.7 小时 | 43 | 3.3521（7.04 亿 token） | 3.2413（10 亿 token） | −0.111 |
| 1.44 亿参数 | 约 5.7 小时 | 44 | 3.3547 | 3.2392 | −0.116 |
| 1.44 亿参数 | 约 11.3 小时 | 43 | 3.1654（14.1 亿 token） | 3.1024（20 亿 token） | −0.063 |

其他点：原配方 1.44 亿参数训练 10 亿 token 用 8.07 小时（3.2452），20 亿 token 用 15.98 小时（3.0932）；8650 万参数训练 5 亿 token 用 2.61 小时（3.5505）。普通 Transformer（8650 万参数，种子 43）训练 5 亿 token 用 1.33 小时（3.5540），7.5 亿 token 用 1.99 小时（3.4673）。

生成文本时差别更大。循环模型每循环一轮，就要为这一轮单独存一份 KV 缓存，原配方循环 8 轮就要存 8 份。精简配方只循环 4 轮，后 3 轮又只需要存最近 64 个 token。我们按 1.44 亿参数模型的结构测了生成一个 token 的耗时，精简配方比原配方循环 8 轮快 2.3 到 3.1 倍，KV 缓存约为三分之一。上下文 1.6 万 token 时，它和同样大小的普通 6 层 Transformer 差不多快，缓存也一样大，因为两者都只有 6 层需要看完整的上下文。上下文只有 2000 token 时，普通 Transformer 的耗时还要少约 20%。

![生成一个 token 的耗时，以及 KV 缓存大小。](img/loop/figure-7.svg)

(按 1.44 亿参数模型的结构测量，上下文 1.6 万 token，一次生成一条。固定 4 轮那一档只固定了轮数、没加局部注意力，用来看两处改动各省了多少。)

| 结构 | 2000 token 一次 1 条 | 2000 token 一次 64 条 | KV 缓存 | 1.6 万 token 一次 1 条 | 1.6 万 token 一次 8 条 | KV 缓存 |
| --- | --- | --- | --- | --- | --- | --- |
| 原配方（循环 8 轮） | 4.66 ms | 20.17 ms | 120 MB | 25.07 ms | 27.01 ms | 960 MB |
| 固定 4 轮 | 3.09 ms | 12.47 ms | 72 MB | 15.26 ms | 16.23 ms | 576 MB |
| 精简配方 | 2.00 ms | 7.01 ms | 37 MB | 8.11 ms | 8.63 ms | 289 MB |
| 4 轮都只看 64 个 token | 1.67 ms | 5.31 ms | 26 MB | 5.80 ms | 6.12 ms | 194 MB |
| 普通 Transformer（6 层） | 1.57 ms | 6.39 ms | 36 MB | 7.87 ms | 8.13 ms | 288 MB |

bf16，静态 KV 缓存，整个生成步骤用 CUDA graph 捕获，局部注意力的轮次用 64 格的环形缓存。耗时只取决于结构，与权重的具体数值无关，所以用的是随机权重。KV 缓存是一次 1 条时的大小。

这是我们自己写的测速程序，没有在 vLLM 这类推理框架里测过。原配方的模型还可以让各轮共用一份 KV 缓存来省内存，Huginn 用过这个办法；固定轮数的模型这样做效果会大幅下降（Huang et al., 2026）。这一项我们没有测。

# 边界

## 训练越久，优势越小

同样训练时间下的差距，从较小模型的 0.18，到较大模型训练 5.7 小时的 0.11，再到训练 11.3 小时的 0.06，一路在缩小。按同样的训练数据比，趋势也一样。1.44 亿参数的模型训到 20 亿 token 时，原配方循环 12 轮的 loss 比精简配方低 0.005 到 0.009，两个种子都是如此，原配方反超了。它要循环 12 轮才做到，是精简配方的 3 倍计算；如果它也只循环 4 轮，仍比精简配方差 0.013 到 0.014。

看起来训练量越大，多循环几轮越值得，这和 Parcae 论文里的缩放规律一致。固定循环更多轮能不能保住优势，我们只在较小的模型上试过一次，固定 8 轮比原配方低 0.034，还需要在更大的训练量上验证。

## 在这个规模上，普通 Transformer 更划算

在 8650 万参数的模型上，我们还训练了一个参数量相同、不循环的 6 层 Transformer。同样训练约 2 小时，它的 loss 比所有循环版本都低，比精简配方低 0.03 到 0.04。在这么小的规模上，循环本身还不划算，精简配方缩小了差距，但没有扭转。这个对比只在较小的模型上做了一次。

## 规模很小

所有结论都来自最多 1.44 亿参数、20 亿 token 的训练，只用了一个数据集（FineWeb-Edu）和一张笔记本 GPU。这两个模型的参数大部分在词嵌入上，真正参与计算的分别只有 1940 万和 4370 万。更大的模型、更长的训练会怎样，这里的数据回答不了。

# 已有工作

这里的大部分现象，别人在别的设置下也见过。

- 随机抽循环轮数、只训练最后几轮，这套做法来自 [Huginn](https://arxiv.org/abs/2502.05171)（Geiping et al., 2025），Parcae 沿用了它。Parcae 和 Huang et al.（2026）都报告过，推理轮数超过训练时的平均轮数之后，loss 就不再下降。
- 固定轮数训练很常见，[Ouro](https://arxiv.org/abs/2510.25741) 和 [LT2](https://arxiv.org/abs/2605.20670) 都固定循环 4 轮。Huang et al. 在 1 亿到 16 亿参数上比较过，平均轮数相同时，固定轮数比随机轮数好，但换一个轮数推理就会明显变差。[Loop growth](https://arxiv.org/abs/2609.19107) 的附录也比较过随机和固定的轮数。
- [Iso-depth](https://arxiv.org/abs/2604.21106) 在计算量相同的条件下比较过只训练最后几轮和每轮都训练，发现只训练最后几轮更好，因为省下的计算可以用来多训数据。我们比的是另一件事：同样的计算量，是用来多跑几轮但只训练其中一部分，还是少跑几轮、每轮都训练。
- 后几轮用局部注意力，[LT2](https://arxiv.org/abs/2605.20670)、[PLT](https://arxiv.org/abs/2510.24824) 和 [LPT](https://arxiv.org/abs/2610.02383) 都做过。PLT 和 LPT 让后几轮只看 64 个 token，同时读取第一轮的全局 KV。让后几轮直接共用第一轮的 KV 是 [MoR](https://arxiv.org/abs/2507.10524) 的做法；PLT 报告过单靠共享 KV 会损失效果，我们也看到了这一点。

我们补充的有三点：在同一套设置里，把训练时间、生成速度和效果放在一起算账，包括实现上的开销；用每一轮隐状态的变化，解释固定轮数为什么不吃亏；以及循环块里哪些计算能省、哪些不能省的一组对照。

# 实验细节

数据。FineWeb-Edu，按顺序读取。验证集和测试集各是一个不参与训练的数据分片。

结构。开头 2 层，循环块 2 层，结尾 2 层。循环块每一轮读入开头两层的输出，也就是 Parcae 的注入方式。词表 32768。

两个规模。8650 万参数：宽 512，4 个注意力头，MLP 2048，不含词嵌入的参数 1940 万。1.44 亿参数：宽 768，6 个注意力头，MLP 3072，不含词嵌入的参数 4370 万。

训练。序列长 2048，每步约 52 万 token。优化器 Muon，学习率 0.008，前一半保持不变，后一半线性降到 0，与 Parcae 的默认设置相同。所有配方用同一套超参数，没有单独调过。

原配方。每条样本的循环轮数取自均值为 8 的泊松分布，最多对最后 4 轮做反向传播。

局部注意力。PyTorch FlexAttention 的因果滑动窗口，宽 64 个 token。

重复次数。主要对比在 8650 万参数上用 3 个种子，在 1.44 亿参数上用 2 个种子。同一个种子下，各配方的初始化和数据顺序相同。削减实验各做一次。

评测。每个模型分别在循环 1、2、3、4、6、8、12、16、24、32 轮时测验证 loss。每个数字都来自一次走完学习率衰减的完整训练，不是中途的检查点。

硬件。一张 RTX 5090 笔记本 GPU（24 GB），bf16，torch.compile。

数字取自每次训练结束时的评测记录。测速和逐轮诊断的原始输出记在项目的研究日志里。

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

PLT (2025). arXiv. [https://arxiv.org/abs/2510.24824](https://arxiv.org/abs/2510.24824)` };
