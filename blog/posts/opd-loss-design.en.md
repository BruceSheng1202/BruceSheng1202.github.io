Over the past half year, on-policy distillation (OPD) has become a new focus in large-model post-training research. Among its most prominent applications, Qwen3 ([Qwen Team, 2025](https://arxiv.org/abs/2505.09388)) uses OPD to transfer capabilities from a large model to a smaller one; Thinking Machines Lab ([Lu, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/)) uses it to improve post-training efficiency and performance; and MiMo ([Ma et al., 2026](https://arxiv.org/abs/2606.30406)) and DeepSeek-V4 ([DeepSeek-AI, 2026](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)) use it to merge the capabilities of multiple RL teacher models into a single model. A growing body of work also explores combining OPD with SFT/RL, using advantage information for improved self-distillation, and truncating or filtering tokens, trajectories, and rounds ([Song & Zheng, 2026](https://arxiv.org/abs/2604.00626)). Here, we focus on how to formulate the loss between the teacher and student policies: Must it be reverse KL? Should we look only at the sampled token? If so, must we use the K1 estimator? Can traditional representation distillation work?

**TL;DR**: 

1. RKL is not the only choice, but it may still be the best one. New estimators have theoretical advantages and are recommended in practice;
2. The sampled token is not the only choice. The teacher's Top-K and tail-corrected variants also have benefits;
3. Representation distillation can be used for OPD, but it is difficult: CKA does not adapt well to the OPD setting, and using it as a supplement to KL does not necessarily help.

# 1 Preliminary
Conventional OPD works as follows. Given a student model $ \pi_\theta $ and a teacher model $ \pi_T $, we let the student reason over a prompt $ x $. For the next-token prediction of every token $ y_t $, the two models produce probabilities $ p_t $ and $ q_t $ for $ y_t $ based on the prompt $ x $ and prefix $ s_t $. We then optimize the log ratio between these two probabilities as a loss term (RKL):

$$
{\mathrm{KL}}\left(\pi_{\theta}(\cdot|s_t) ,\Vert \pi_T(\cdot| s_t)\right)
=
\mathbb{E}_{y_t \sim \pi_{\theta}(\cdot \mid s_t)}
\left[
\log \pi_{\theta}(y_t \mid s_t)
-
\log \pi_T(y_t \mid s_t)
\right].
$$

There are two ways to perform the optimization. One treats this loss as a penalty term in the reward, thereby viewing OPD as a policy-gradient RL method with dense rewards (PG-style OPD). The other directly backpropagates through the loss (GKD-style OPD). These two methods are fundamentally different in their optimization, yet they are often conflated.

PG-style OPD usually treats the reverse KL above as a reward and uses it in a policy-gradient update. Here, reverse KL is sampled-token KL: whichever token the student samples is the token on which supervision is applied. Reverse KL also uses the conventional K1 estimator. A stop-gradient is applied before it is passed into the reward.

This article focuses on GKD-style OPD. When calculating the loss, it does not consider only the token sampled by the student; that Monte Carlo estimator is unbiased but has very high variance. Because we can obtain full-vocabulary logits from both models at the current token prediction, the theoretically unbiased, zero-variance loss is the expectation of RKL over the entire vocabulary, namely the probability-weighted log ratio:

GKD-style:

$$
D\left(
\pi_{\theta}(\cdot \mid s_t),
\pi_T(\cdot \mid s_t),
y_t
\right)
=
\sum_{v \in V}
\pi_T(v \mid s_t)
\log
\frac{
\pi_T(v \mid s_t)
}{
\pi_{\theta}(v \mid s_t)
}.
$$

However, this full-vocabulary RKL often causes GPU out-of-memory errors during backpropagation because the vocabularies $ V $ of frontier large models are enormous, commonly containing one or two hundred thousand tokens (the Qwen3 family has 150,000; [Qwen Team, 2025](https://arxiv.org/abs/2505.09388)). Offline processing can solve this problem, but it is still awkward to implement. One approach that has attracted attention is Top-K truncation: calculate the loss using only the K largest logits in the vocabulary ([Fu et al., 2026](https://arxiv.org/pdf/2603.25562)). This avoids the high variance of sampled KL, but it also discards probability mass in the tail and therefore biases the loss value.

In practice, Top-K also introduces accompanying engineering problems. If K is large, rollout efficiency and backpropagation memory may become bottlenecks. If K is small, the bias in the loss grows, raising further questions: What if the student's Top-K and the teacher's Top-K barely overlap? What if the sampled token does not fall inside the Top-K?

For these problems, current research suggests that using the teacher's Top-K set may be better, and that the sampled token should preferably be merged into it. In engineering implementations, SGLang in slime can retrieve only Top-k. Missing values can be partially recovered by retrieving a larger K or by running an additional teacher-side forward pass. Empirically, a small value such as -12 can also be used as a biased but acceptable fallback approximation, or one can directly take the intersection.

From the perspective of KL divergence, current loss designs offer no perfect choice, either in theory or in practice:

1. Sampled-token KL is unbiased but has high variance. Extreme values can easily cause training to collapse, and engineering tricks treat the symptoms rather than the cause;
2. Full-vocabulary KL is unbiased and has zero variance, but it is often unavailable or expensive and laborious to implement;
3. Top-K truncation is biased and has zero variance, while also creating a host of engineering issues and hyperparameter choices.

Leaving these engineering complications aside, there are three cleaner options:

1. Can we avoid RKL? Try other loss forms;
2. Can sampled RKL be made more reliable? Try other estimator forms;
3. Can we avoid distilling logits? Try representation distillation.

# 2 Reverse KL vs Other Loss?
Reverse KL is certainly not the only choice, but it may still be the best one available. Foundational works such as MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)) and GKD ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)) had already explored forward KL, JS divergence, and other options. Combining the two KL directions, taking their midpoint, or extending them to more general f-divergences (Hellinger distance ([Xie et al., 2025](https://arxiv.org/abs/2510.11615)) and $ \alpha $-$ \beta $ divergence ([Wang et al., 2025](https://arxiv.org/abs/2505.04560))) is nothing new either. The key question is: What are the benefits and drawbacks?

![](img/opd/figure-000.png)

(This figure comes from MiniLLM; many related articles contain similar illustrations.)

Across conventional knowledge distillation, from the neural-network era and the BERT era to offline LLM distillation and post-training OPD, there has been something close to a consensus: forward KL is mean-seeking. In other words, it makes the student imitate the teacher's entire distribution, so it cannot become sufficiently sharp at probability peaks and cannot closely follow probability valleys. Reverse KL is mode-seeking: it imitates only the positions where the teacher assigns the highest probability and cannot fit a multimodal distribution. Motivated by this distinction, many papers have tried adaptive weighted combinations of the two KL divergences. From each paper's theoretical account and narrative, these combinations have some justification and practical effect, but ultimately they remain heuristic ideas.

Despite drawbacks such as overconfidence, strong benchmark performance has kept RKL the first choice for OPD. In fact, RKL may have other advantages. The following figure extends the toy example from Adaptive KL ([Wu et al., 2025](https://arxiv.org/pdf/2404.02657)) to larger vocabularies. Although FKL and RKL may appear to “arrive at the same destination by different routes” during optimization, in the large output spaces of modern LLMs, RKL is clearly better at bringing the student's probabilities toward the teacher's. Theoretically, [Luong et al. (2026)](https://arxiv.org/pdf/2604.00223v1) show that RKL reduces optimization complexity, especially when the student has less capacity than the teacher.

![](img/opd/figure-001.png)

(As the output space grows, distillation requires more epochs to converge, and FKL and RKL approach the target in two fundamentally different ways.)

# 3 Better Estimators?
RKL may still be the best loss for OPD. Are there better estimators that make distillation more stable? Yes, there are:

![](img/opd/figure-002.png)

EMA-PG ([Zhang & Ba, 2026](https://arxiv.org/pdf/2602.04417)) summarizes the K1-K3 estimators for RKL discussed by [Schulman (2020)](http://joschu.net/blog/kl-approx.html) and the K3++ estimator from DeepSeek-V3.2 ([DeepSeek-AI, 2025](https://arxiv.org/pdf/2512.02556)), analyzes their bias and variance in both estimated values and gradients, and proposes the K4 and K5 estimators. Through computational corrections, sampled RKL can be made unbiased in both its loss value and gradient. In OPD research, Skill-SD ([Wang et al., 2026](https://arxiv.org/pdf/2604.10674)) also discusses the use of K3++ in Appendix D.2 as an importance-weighted version of the K3 estimator. At present, we have not seen these better estimators used in other OPD studies (knowledge current to June 2026).

The so-called TopkReverseKL and TopkForwardKL in the figure above are combinations of sampled KL and Top-K-truncated KL. This method corrects the theoretical bias of Top-K-truncated KL while also reducing the high variance of sampled KL, suggesting that it is better when truncating at a small k.

![](img/opd/figure-003.png)

[Fu et al. (2026)](https://arxiv.org/pdf/2603.25562) compare truncated Top-K KL with EMA-PG-style tail correction for OPD in their appendix. As the figure above shows, the performance gap between them depends strongly on the value of k and the sharpness of the distribution. Inspection of their reproduction code also shows that it does not use the correct K4 estimator.

Therefore, using these estimators and tail corrections to improve OPD stability remains a worthwhile direction in engineering practice. Even so, the high variance of sampled-token KL is not substantially reduced by the theoretical benefits of the computation. This drawback remains as long as the on-policy student continues to sample. This is annoying. Can we compute KL without logits instead?

# 4 Representation Distillation
## 4.1 Traditional methods and [Ke zhou qiu jian](https://www.chinestudy.com/blog/chinese-idioms-kezhouqiujian): MSE and Cosine Loss
Before OPD, researchers had already tried just about everything in off-policy knowledge distillation: hidden states from every layer, attention maps, and almost anything else inside the model could be distilled. These losses were often added on top of KL over the logits, in the hope that the student and teacher could declare, “Our wills align!” and perform better. The OPD survey ([Song & Zheng, 2026](https://arxiv.org/abs/2604.00626)) also looks to representation distillation as a way to distill across vocabularies and models.

The most common forms of representation distillation in conventional offline policy distillation are mean squared error (MSE) and cosine-similarity loss. Before obtaining the student and teacher's full-vocabulary logits for the current token prediction, we extract the hidden states from one of their layers (say, simply, the final layer), directly calculate the MSE or cosine loss between the two sets of vectors, and backpropagate to minimize it.

The idea is simple: match the student's representations to the teacher's, either in value or direction, and they should look similar after training. But in LLM post-training, those representations change as training proceeds. Forcing them to match is [ke zhou qiu jian (marking the boat to find the sword)](https://www.chinestudy.com/blog/chinese-idioms-kezhouqiujian).

First, to calculate these two losses, the two hidden-state vectors must have the same dimensionality. This confines them to distillation between models at the same scale: the teacher may be the student after its capabilities have been strengthened through SFT/RL, as in MiMo's MOPD and related settings ([Ma et al., 2026](https://arxiv.org/abs/2606.30406)). It cannot accommodate a Qwen3-style large-to-small setting. Academia, of course, has its “cut the foot to fit the shoe” workarounds. Flex-KD ([Saadi & Wang, 2025](https://openreview.net/forum?id=aiMINHhIiQ)) uses dimensional correlations to hard-select the n most relevant hidden-state dimensions so that their number can match the student's dimensionality. Another approach learns a projection matrix before training ([Miles et al., 2024](https://arxiv.org/pdf/2403.06213)) to perform a semantic transformation between teacher and student.

But this produces the first ke zhou qiu jian problem. Whether we hard-cut the most relevant dimensions or map them through a bridge matrix, we assume that the mismatch can be corrected by some static mechanism. Yet the student model is being dynamically trained and changed. Whether the mechanism changes or remains fixed, neither option is satisfactory.

Directly calculating these two losses also hides another risk: Does every feature in the teacher and student hidden states have the same meaning? Because of representation superposition, the layer-by-layer propagation of semantics resembles a multi-constraint linear program ([Xiong, 2026](https://arxiv.org/pdf/2603.01227)). After post-training, both the semantic strength and meaning of every dimension have changed to a greater or lesser extent. Directly calculating MSE or cosine similarity at this point forces the model to approximate a biased target.

Is there a way around this problem?

## 4.2 Centered Kernel Alignment: An Imperfect Similarity
Relational knowledge distillation ([Park et al., 2019](https://arxiv.org/pdf/1904.05068)) is one way to address this kind of problem. Instead of forcing a match on every training sample, it distills according to similarity relations among samples. Centered Kernel Alignment (CKA) ([Kornblith et al., 2019](https://arxiv.org/abs/1905.00414)) is a representative and popular method in this family. Following [Dasgupta and Cohn (2025)](https://openreview.net/forum?id=IcVSKhVpKu), we first extract hidden-state vectors from one layer of each model and treat every token as a sample. The tokens in the entire batch therefore form two hidden-state matrices, $ H^s $ and $ H^t $. After normalizing the matrices, we multiply each by itself to obtain Gram matrices $ K $, and then calculate CKA:

$$
\mathcal{L}_{CKA} = 1\ -CKA(H^t,H^s) = 1\ - \ \frac{tr(K^tK^s)}{||K^t||_F\ ||K^s||_F}.
$$

This loss resembles Pearson correlation but is based on the Hilbert-Schmidt Independence Criterion (HSIC). With a linear kernel, it reduces to the expression above, and optimizing it is also equivalent to optimizing maximum mean discrepancy (PCKA ([Zhou et al., 2024](https://arxiv.org/pdf/2401.11824))).

If the mathematics sounds complicated, the simple explanation is that CKA measures the similarity between the two models across a large collection of token predictions. Although the predictions come from different reasoning contexts, the models' hidden-state responses at similar positions should be as similar as possible. For example, if the proposition “1+1=2” appears in one answer, the predictive response at the “2” should be similar to the response at “1+1=2” in another answer.

Its advantage is that multiplying each hidden-state matrix by itself to form a Gram matrix eliminates the dimensional mismatch between the two models' hidden states:

$$
K^t = \tilde{H}^t{\tilde{H}^t}^T, \ K_s = \tilde{H}^s{\tilde{H}^s}^T \in \mathbb{R}^{N\times N},
$$

where $N$ is the number of samples.

This loss design is therefore naturally suited to distillation between models of different sizes: **as long as the layers correspond, distillation is possible regardless of dimensionality**.

Considering the degree of information processing and semantic relevance of the “same layer,” we use the final layer—the layer immediately before the representation is translated through the vocabulary into human language. These final hidden states already contain all of the model's thinking about prompt $ x $ and the current prefix $ s_t $, plus its response just before it is spoken. This makes them the most natural target for distillation. Architectural differences between models of different sizes also make the final layer the best choice for avoiding cross-layer mismatch.

Another benefit of CKA or representation distillation is that it raises the upper bound of what distillation can learn. When KL divergence is calculated from logits, the softmax function erases the magnitude of the logits themselves. As a result, even in the ideal case where the probability mass is perfectly aligned, the logits and the model content that precedes them remain unaligned. In the hidden states, this appears as a constant-related error term, leaving the distilled models “outwardly alike but inwardly apart.”

This method nevertheless carries risks and drawbacks. Effectiveness does not imply rigor, and the following issues ultimately led us to abandon further development of this project:

1. Both the numerator and denominator of CKA are HSIC estimators, whose distributional assumption is that samples are independent and identically distributed. Tokens in a batch may include different rollouts of the same question, creating severe sample autocorrelation. Even if data parallelism and the number of samples ensure that this does not occur within a microbatch, tokens in the same reasoning sequence are still generated autoregressively. First-differencing the sequence's token-level hidden states cannot eliminate the problem. In other words, this loss objective is biased in its engineering implementation, lacks rigor in its distributional assumptions, and cannot be fixed at the root by the obvious engineering remedies.
2. Another engineering risk of CKA lies in its batch-level accumulation, which is not as flexible as KL. In actual training, KL divergences over all tokens can be accumulated until the entire batch is complete and only then used for a gradient update. CKA, however, is constructed as a ratio and is not additive. Its batch-level engineering implementation is therefore the mean of microbatch CKAs. Because its expectation cannot pass through the nonlinearity under Jensen's inequality, this mean is not equivalent to batch-level CKA, causing an objective mismatch. The problem follows from [Dasgupta and Cohn (2025)](https://openreview.net/forum?id=IcVSKhVpKu), but to our knowledge it has never been discussed or resolved by that work or related papers.
3. When representation distillation is used alone, training of the vocabulary projection matrix (the LM head) is almost always skipped. In tied-weight models, the LM head can be influenced only indirectly through the input-embedding layer. This creates another ke zhou qiu jian problem: the model's thinking has changed to become more like the teacher's, but its voice remains the same. It has not learned how to translate latent reasoning into reasoning expressions as the teacher does. Moreover, CKA depends on Gram matrices calculated from hidden-state matrices, which conceals semantic drift in the vocabulary. In theory, an orthogonal matrix can allow the LM head to drift without CKA noticing. From these two points, we conclude that representation distillation may allow the student and teacher to declare, “Our wills align!” Even so, it is best positioned as a supplement to KL rather than a replacement.
4. On-policy sampling in OPD is a double-edged sword. It brings many of OPD's benefits, but it also introduces one of its most worrying drawbacks: prefix bias in long reasoning. The student may generate a reasoning path that the teacher considers impossible under its original policy and fail to correct itself despite repeated hmmm/wait reflections. Yet, over a long reasoning response, the teacher is still forced to score every token—like “treating a dead horse as if it were still alive.” Supervision at that point is beyond rescue. Although CKA measures similarity among token predictions, so many noisy tokens can bias and destabilize its value, potentially inflating it while the information that truly ought to be learned receives too little weight.

Thus, although representation distillation avoids the design and estimation problems of KL distillation, it still has several clear limitations: the ke zhou qiu jian problem in conventional methods (MSE and cosine loss), and CKA's invalid distributional assumptions in engineering implementations, objective mismatch, risk of semantic drift, and inflated oscillations. We may still be missing a representation-distillation loss that works across a broad range of OPD settings.

## 4.3 Can Representation Distillation Cross Vocabularies or Models?
No. Although representation distillation bypasses the LM head and directly aligns model internals, it does not change the key constraints that cross-vocabulary or cross-model settings impose on OPD. Consider the simplest example: the number “120” may be one token in one model's vocabulary but three tokens, {1, 2, 3}, in another model's vocabulary. For either logits KL or hidden states, one vector must then correspond to three vectors. Recent work can merge the probabilities of three tokens into one through Markov conditional probabilities ([Niu et al., 2026](https://arxiv.org/abs/2606.09456)), or retain only token segmentations that form a “common language” between the two models while discarding tokens on which they disagree ([Sun et al., 2026](https://arxiv.org/abs/2605.07711)). But these cross-vocabulary methods solve the broader OPD problem; they do not solve it through representation distillation. Although CKA does not require one-to-one token correspondence, it does require consistent samples. Different tokenization schemes cause both the number and indices of tokens along a reasoning chain to differ, already creating a computational obstacle. In short, representation distillation across vocabularies or models still cannot avoid token alignment and therefore cannot solve the problem by itself.

# 5 Experiments and Analysis
We have introduced several loss designs for OPD: full-vocabulary RKL, Top-K-truncated RKL, multiple estimators for sampled-token KL, EMA-PG-style tail correction, and the CKA loss for representation distillation. How do they actually perform?

## 5.1 Math Reasoning
We use two pairs of 1.5B models from JustRL ([He et al., 2025](https://arxiv.org/abs/2512.16649)), based on Qwen 2.5 Math, to perform strong-to-weak training within the same architecture in the mathematics domain. All evaluations sample 32 times with a 32k reasoning-length limit to reduce statistical noise as much as possible. We also use the more accurate Bayes@K ([Hariri et al., 2026](https://openreview.net/forum?id=PTXi3Ef4sT)) as the metric and plot 90% confidence intervals. As a rough heuristic, non-overlapping intervals suggest a significant difference in performance.

First, in terms of training dynamics, pure CKA progresses substantially more slowly than pure KL OPD. The sampled RKL here uses the K4 estimator.

![](img/opd/figure-004.png)

In the final evaluation, pure on-policy CKA struggles to compete with KL OPD and also underperforms off-policy CKA. It is worth noting, however, that with K4 RKL and the teacher's Top-K distillation, the trained student surpasses the teacher to some extent.

![](img/opd/figure-005.png)

In theory, CKA can be an effective supplement to KL OPD. In our experimental analysis, however, their gradient directions in the final hidden layer are almost orthogonal. After trying multiple weighting schemes, the resulting training performance usually falls somewhere between the two. In the On-Policy CKA + sampled K4 or Top50 experiments in the table above, performance improves only slightly over pure KL OPD in some cases and declines in others.

We also ran K1/K5 experiments and the EMA-PG-style tail correction, namely the Top50 + K4 experiment:

1. K1 has a theoretically biased gradient and its estimates can easily collapse, but after clipping its results are similar;
2. Because K5 is itself FKL, it is prone to oscillations from extreme values, but its training results are not much worse;
3. Although EMA-PG is theoretically superior, it yields only a slight improvement.

## 5.2 Clinical Reasoning
On the medical diagnostic reasoning problems that medical-algorithm researchers care about, pure CKA holds its own. The combined losses also perform about as well as K4 and Top-50; all of them successfully distill the capabilities. One caveat is that, for cost reasons, our HealthBench grader is gpt-oss-120b, with MF1=0.66. This evaluation model falls some distance short of the official default, GPT-4.1 ([Arora et al., 2025](https://arxiv.org/abs/2505.08775)), and is a relatively weak and somewhat unreliable grader.

![](img/opd/figure-006.png)

## 5.3 Large-to-Small Distillation: Math Reasoning
One major motivation for using CKA as a loss is its independence from hidden-state dimensionality, which allows a teacher and student of different sizes to be distilled as long as they share the same vocabulary. In the experiment below, OpenMath-Nemotron-7B teaches a 1.5B student. The combined CKA and K4 loss does perform better than either loss alone, but it transfers only a small portion of the teacher's capabilities.

![](img/opd/figure-007.png)

In fact, strong-to-weak distillation is not particularly difficult in OPD; the truly difficult problem is large-to-small distillation. Even with strong baselines such as G-OPD ([Yang et al., 2026](https://arxiv.org/abs/2602.12125)), large-to-small distillation often fails and may even cause capability regression. Even when a strengthened Qwen3-4B teacher distills a Qwen3-1.7B student, the student often cannot learn all of the teacher's performance, perhaps because of their size and architectural differences. We also do not yet have a conclusion about the performance differences produced by choosing base, instruct, or post-trained models as the student and teacher. A plausible conclusion is that distillation becomes harder as the models differ more in size and architecture. Because of those differences, a capability gap between student and teacher may always remain, no matter how the distillation is performed.

For example, in the figure below, a 14B teacher distills a 1.5B student. Because of the large size and architectural gap, the CKA loss decreases but hidden-norm alignment fails, training becomes abnormal, and the model's reasoning capability ultimately regresses substantially.

![](img/opd/figure-008.png)

## 5.4 Analysis of CKA Training Dynamics
Continuing the analysis of the training dynamics above, the motivation for CKA representation distillation is to have the student and teacher declare “Our wills align!”—rather than remain “outwardly alike but inwardly apart.” The latter has a precise theoretical meaning here: pure KL distillation aligns the student's output probability mass with the teacher's, but does not align their logits or hidden states. A constant-related error term therefore remains in the hidden states.

Our experimental analysis confirms this. In strong-to-weak experiments using CKA, the norms of the two models' final-layer hidden states align and converge quickly. With pure KL, a somewhat larger gap persists throughout training.

![](img/opd/figure-009.png)

![](img/opd/figure-010.png)

In the method-design section, we noted that CKA implementations suffer from sample autocorrelation and token noise. These problems ultimately appear as loss spikes during training. Inflated CKA values and spikes are not new in conventional representation distillation; the main remedy is to filter out certain samples, which is one possible way to improve CKA. Our attempts also revealed several interesting phenomena.

First, the higher the sampling temperature, the larger the loss oscillations. This is intuitive: at a higher sampling temperature, the student is more likely to generate absurd tokens, increasing the proportion of noise in the teacher's supervision. Once a harmful sample appears, the loss at that step surges.

![](img/opd/figure-011.png)

In practice, loss spikes can sometimes be severe: CKA loss may increase by tens or even hundreds of times within a single step, only to return to normal at the next step.

![](img/opd/figure-012.png)

Analysis of the anomalous tokens shows that those causing spikes are mostly punctuation marks and discourse-connective words, with almost no relevance to reasoning. In engineering practice, however, data parallelism and the number of samples can reduce the autocorrelation problem and substantially alleviate loss spikes. Taking a per-token first difference of hidden states also changes the objective fundamentally and almost entirely removes the spikes, while producing almost no difference in distillation performance.

![](img/opd/figure-013.png)

Beyond first differences, CKA has several alternative calculation forms; for example, its similarity may take a square-root form or a squared form. In practice, they produce nearly identical results.

# 6 Conclusion
Despite its drawbacks, the RKL conventionally used in OPD remains the best choice in both theory and practice. From the perspectives of estimator and loss construction, new estimators such as K4/K5 may offer slight advantages, but sampled-token KL still has high variance. Top-K and its corrected variants may offer theoretical advantages and occasionally perform slightly better, but the bias-variance problem remains. We recommend using them in practice, but they do not solve the fundamental problem.

Representation distillation for OPD is difficult. Conventional MSE and cosine loss suffer from the ke zhou qiu jian problem, while CKA, a newer popular choice, performs reasonably well but is still not a good fit for OPD. Suitable loss constructions remain to be explored, but representation distillation should supplement KL rather than replace it.

# References

Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos, S., Geist, M., & Bachem, O. (2024). *On-policy distillation of language models: Learning from self-generated mistakes*. International Conference on Learning Representations. [https://arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649)

Arora, R. K., Wei, J., Hicks, R. S., Bowman, P., Quiñonero-Candela, J., Tsimpourlas, F., Sharman, M., Shah, M., Vallone, A., Beutel, A., Heidecke, J., & Singhal, K. (2025). *HealthBench: Evaluating large language models towards improved human health*. arXiv. [https://arxiv.org/abs/2505.08775](https://arxiv.org/abs/2505.08775)

Dasgupta, S., & Cohn, T. (2025). *Improving language model distillation through hidden state matching*. International Conference on Learning Representations. [https://openreview.net/forum?id=IcVSKhVpKu](https://openreview.net/forum?id=IcVSKhVpKu)

DeepSeek-AI. (2025). *DeepSeek-V3.2: Pushing the frontier of open large language models*. arXiv. [https://arxiv.org/abs/2512.02556](https://arxiv.org/abs/2512.02556)

DeepSeek-AI. (2026). *DeepSeek-V4: Towards highly efficient million-token context intelligence* [Technical report]. [https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)

Fu, Y., Huang, H., Jiang, K., Liu, J., Jiang, Z., Zhu, Y., & Zhao, D. (2026). *Revisiting on-policy distillation: Empirical failure modes and simple fixes*. arXiv. [https://arxiv.org/abs/2603.25562](https://arxiv.org/abs/2603.25562)

Gu, Y., Dong, L., Wei, F., & Huang, M. (2024). *MiniLLM: On-policy distillation of large language models*. International Conference on Learning Representations. [https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)

Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026). *Don't Pass@k: A Bayesian framework for large language model evaluation*. International Conference on Learning Representations. [https://openreview.net/forum?id=PTXi3Ef4sT](https://openreview.net/forum?id=PTXi3Ef4sT)

He, B., Qu, Z., Liu, Z., Chen, Y., Zuo, Y., Qian, C., Zhang, K., Chen, W., Xiao, C., Cui, G., Ding, N., & Liu, Z. (2025). *JustRL: Scaling a 1.5B LLM with a simple RL recipe*. arXiv. [https://arxiv.org/abs/2512.16649](https://arxiv.org/abs/2512.16649)

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. In *Proceedings of the 36th International Conference on Machine Learning*. [https://arxiv.org/abs/1905.00414](https://arxiv.org/abs/1905.00414)

Lu, K. (2025). *On-policy distillation*. Thinking Machines Lab. [https://thinkingmachines.ai/blog/on-policy-distillation/](https://thinkingmachines.ai/blog/on-policy-distillation/)

Luong, H.-C., Tran, D. B., & Chen, L. (2026). *Diversity-aware reverse Kullback-Leibler divergence for large language model distillation*. arXiv. [https://arxiv.org/abs/2604.00223](https://arxiv.org/abs/2604.00223)

Ma, W., Wei, J., Zhao, L., Zhang, H., Xiao, B., Li, L., Yang, Q., Gao, B., Wang, Y., Li, R., Dong, J., Sui, Z., & Luo, F. (2026). *MOPD: Multi-teacher on-policy distillation for capability integration in LLM post-training*. arXiv. [https://arxiv.org/abs/2606.30406](https://arxiv.org/abs/2606.30406)

Miles, R., Elezi, I., & Deng, J. (2024). \(V_kD\): Improving knowledge distillation using orthogonal projections. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. [https://arxiv.org/abs/2403.06213](https://arxiv.org/abs/2403.06213)

Niu, Y., Xiao, H., Liu, D., Wang, Z., Gong, D., Wang, Y., & Li, J. (2026). *Breaking the tokenizer barrier: On-policy distillation across model families*. arXiv. [https://arxiv.org/abs/2606.09456](https://arxiv.org/abs/2606.09456)

Park, W., Kim, D., Lu, Y., & Cho, M. (2019). Relational knowledge distillation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. [https://arxiv.org/abs/1904.05068](https://arxiv.org/abs/1904.05068)

Qwen Team. (2025). *Qwen3 technical report*. arXiv. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

Saadi, K., & Wang, D. (2025). *Flexible feature distillation for large language models*. [https://openreview.net/forum?id=aiMINHhIiQ](https://openreview.net/forum?id=aiMINHhIiQ)

Schulman, J. (2020). *Approximating KL divergence*. [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)

Song, M., & Zheng, M. (2026). *A survey of on-policy distillation for large language models*. arXiv. [https://arxiv.org/abs/2604.00626](https://arxiv.org/abs/2604.00626)

Sun, J., Zheng, M., Song, M., Zhong, Q., Cheng, Y., Feng, B., Liu, P., Fang, J., & Wang, X. (2026). *SimCT: Recovering lost supervision for cross-tokenizer on-policy distillation*. arXiv. [https://arxiv.org/abs/2605.07711](https://arxiv.org/abs/2605.07711)

Wang, G., Yang, Z., Wang, Z., Wang, S., Xu, Q., & Huang, Q. (2025). ABKD: Pursuing a proper allocation of the probability mass in knowledge distillation via \\(\alpha\\)-\\(\beta\\)-divergence. In *Proceedings of the 42nd International Conference on Machine Learning*. [https://arxiv.org/abs/2505.04560](https://arxiv.org/abs/2505.04560)

Wang, H., Wang, G., Xiao, H., Zhou, Y., Pan, Y., Wang, J., Xu, K., Wen, Y., Ruan, X., Chen, X., & Qi, H. (2026). *Skill-SD: Skill-conditioned self-distillation for multi-turn LLM agents*. arXiv. [https://arxiv.org/abs/2604.10674](https://arxiv.org/abs/2604.10674)

Wu, T., Tao, C., Wang, J., Yang, R., Zhao, Z., & Wong, N. (2025). Rethinking Kullback-Leibler divergence in knowledge distillation for large language models. In *Proceedings of COLING 2025*. [https://arxiv.org/abs/2404.02657](https://arxiv.org/abs/2404.02657)

Xie, X., Xue, Z., Wu, J., Li, J., Wang, Y., Hu, X., Liu, Y., & Zhang, J. (2025). *LLM-oriented token-adaptive knowledge distillation*. arXiv. [https://arxiv.org/abs/2510.11615](https://arxiv.org/abs/2510.11615)

Xiong, B. (2026). *The lattice representation hypothesis of large language models*. International Conference on Learning Representations. [https://arxiv.org/abs/2603.01227](https://arxiv.org/abs/2603.01227)

Yang, W., Liu, W., Xie, R., Yang, K., Yang, S., & Lin, Y. (2026). *Learning beyond teacher: Generalized on-policy distillation with reward extrapolation*. arXiv. [https://arxiv.org/abs/2602.12125](https://arxiv.org/abs/2602.12125)

Zhang, L., & Ba, J. (2026). *EMA policy gradient: Taming reinforcement learning for LLMs with EMA anchor and Top-k KL*. arXiv. [https://arxiv.org/abs/2602.04417](https://arxiv.org/abs/2602.04417)

Zhou, Z., Shen, Y., Shao, S., Gong, L., & Lin, S. (2024). *Rethinking centered kernel alignment in knowledge distillation*. arXiv. [https://arxiv.org/abs/2401.11824](https://arxiv.org/abs/2401.11824)
