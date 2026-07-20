window.OPD_CONTENT = { en: String.raw`Over the past half year, on-policy distillation (OPD) has become a new focus in large-model post-training research. Among its most prominent applications, Qwen3 ([Qwen Team, 2025](https://arxiv.org/abs/2505.09388)) uses OPD to transfer capabilities from a large model to a smaller one; Thinking Machines Lab ([Lu, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/)) uses it to improve post-training efficiency and performance; and MiMo ([Ma et al., 2026](https://arxiv.org/abs/2606.30406)) and DeepSeek-V4 ([DeepSeek-AI, 2026](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)) use it to merge the capabilities of multiple RL teacher models into a single model. A growing body of work also explores combining OPD with SFT/RL, using advantage information for improved self-distillation, and truncating or filtering tokens, trajectories, and rounds ([Song & Zheng, 2026](https://arxiv.org/abs/2604.00626)). Here, we focus on how to formulate the loss between the teacher and student policies: Must it be reverse KL? Should we look only at the sampled token? If so, must we use the K1 estimator? Can traditional representation distillation work?

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

Zhou, Z., Shen, Y., Shao, S., Gong, L., & Lin, S. (2024). *Rethinking centered kernel alignment in knowledge distillation*. arXiv. [https://arxiv.org/abs/2401.11824](https://arxiv.org/abs/2401.11824)`, zh: String.raw`近半年来 OPD 的研究成为了大模型后训练新的热点。追溯其最主要的应用，Qwen3 ([Qwen Team, 2025](https://arxiv.org/abs/2505.09388)) 用它来做能力传输大蒸小，ThinkingMachineLab ([Lu, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/)) 用它来做后训练能力提升获得比 RL 更高的性价比，MiMo ([Ma et al., 2026](https://arxiv.org/abs/2606.30406)) 和 DeepseekV4 ([DeepSeek-AI, 2026](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf)) 用它来将多个 RL 教师模型的能力合版到一个模型，以及一系列关于将 OPD 与 SFT/RL 融合、利用优势信息做自蒸馏提升、token/轨迹/轮次的截断筛选等角度的研究 ([Song & Zheng, 2026](https://arxiv.org/abs/2604.00626))。在此我们关注于 teacher 和 student 两个策略间的损失制定问题：一定要用 reverse KL 吗？只看 sampled token 吗？用的话必须用 K1 估计量吗？传统的表征蒸馏能不能做？

懒得看全文？这是结论：

1. RKL 不是唯一选择但可能仍然是最佳选择，新估计量存在理论优势，实践上推荐使用；
2. Sampled token 不是唯一选择，teacher Top-K 及其尾部修正变体也有好处；
3. 表征蒸馏 OPD 能做，但不好做：CKA 在 OPD 水土不服，作为 KL 补充不一定有好处。

# 1 Preliminary
传统的 OPD 是这样做的：给定一个 student 模型 $ \pi_\theta $ 和一个 teacher 模型 $ \pi_T $，让 student 对 prompt $ x $ 进行推理，那么对于每一个 token $ y_t $ 的 next token prediction，我们都有两个模型基于 prompt $ x $ 和前缀 $ s_t $ 关于 $ y_t $ 给出的预测概率 $ p_t $ 和 $ q_t $，从而把两个概率的 log ratio 作为损失项 (RKL) 去做优化：

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

而做优化的方式有两种：一种是将这个 loss 看作是 reward 的 penalty 项，即将 OPD 视为是一种具有密集奖励的 policy gradient RL 方法 (PG-style OPD)，另一种是直接将这个损失反向传播 (GKD-style OPD)。这两种方法在优化上其实是截然不同的，但很多人将他们混为一谈。

PG-style OPD 通常将上面的 reverse KL 作为奖励并用于策略梯度更新。在此，reverse KL 采用的是 sampled token KL，即 student 采样到什么 token，就在什么 token 上做监督。并且，reverse KL 使用的是传统的 K1 估计量。在传入 reward 之前，还会加上 stop gradient。

本文关注的是 GKD-style OPD，其在计算损失时并不只考虑 student 采样到的 token，这样的 Monte Carlo 估计量无偏但方差很大。 由于我们可以在当前 token prediction 上得到两个模型全词表的 logits，所以理论上无偏无方差的损失是整个词表上 RKL 的期望，即概率加权的 log ratio：

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

但是这样的全词表 RKL 往往会在反向传播时造成显存 OOM，因为前沿大模型的词表 $ V $ 太大了，动辄十几二十万 (Qwen3 系列是 15 万，[Qwen Team, 2025](https://arxiv.org/abs/2505.09388))。尽管我们可以通过一些离线处理的方法来解决，但工程上依然比较麻烦。因此一个引发关注的做法是截取 Top-K，即只计算词表上 logits 最大的 K 个来计算 ([Fu et al., 2026](https://arxiv.org/pdf/2603.25562))。虽然这样防止了 sampled KL 的高方差，但也舍去了尾部的概率质量造成了损失值上的有偏。

在实际操作中，取 Top-K 还会引发伴随的工程问题。如果 K 值大了，rollouts 的效率和反向传播时的显存都可能会成为问题。而如果 K 值小了，损失上的 bias 也会随之增大，并引发以下问题：student 的 Top-K 和 teacher 的 Top-K 几乎不相同，怎么办？sampled token 不落在 Top-K 里又怎么办？

对于这样的问题，当前研究表明使用 teacher 的 Top-K 集可能更好，并且最好把 sampled token 也合并进去。在工程实现上，slime 中 sglang 只能取 Top-k，缺失值可以通过多取 K 或者 teacher 端二次传播部分补齐，经验上使用一个较小的值 (如-12) 作为填补也可以作为一个有偏但可妥协的兜底近似，也可直接取交集。

从 KL divergence 的角度来看，当前的损失设计从理论到实操并没有完美的选择：

1. sampled token KL 无偏高方差，容易极值导致训练崩溃，工程 trick 治标不治本；
2. 全词表 KL 无偏无方差但往往不可得，或者工程上费时耗力；
3. Top-K 截断有偏无方差，还引起一堆工程问题和超参设计。

既然如此，除去繁杂的工程手段，相对优雅的几种方案就是：

1. 不用 RKL 可以吗：其他 loss 形式
2. sampled RKL 能不能更靠谱点：其他估计量形式
3. 不蒸 logits 行不行：尝试表征蒸馏

# 2 Reverse KL vs Other Loss？
Reverse KL 当然不是唯一选择，但可能仍是当前的最优选择。早在方法奠基文章 MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543))、GKD ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)) 等作品中早已探讨过了使用 forward KL、JS-divergence 等选择，将两种 KL 进行结合、取其中点、或者是拓展到更广义的 f-divergence (Hellinger distance ([Xie et al., 2025](https://arxiv.org/abs/2510.11615)), $ \alpha $-$ \beta $divergence ([Wang et al., 2025](https://arxiv.org/abs/2505.04560))) 也不是什么新鲜事。关键是，好处/坏处是什么？

<!-- 这是一张图片，ocr 内容为：TARGET DISTRIBUTION 0.4 FORWARD KLD REVERSE KLD(OURS) LLI R 0.2 AY 0.0 2.5 12.5 10.0 5.0 7.5 0.0 THE TOY FIGURE 2: WEFIT OY EXPERIMENT. A GAUSSIAN MIXTURE DISTRIBUTION WITH A SINGLE GAUSSIAN DISTRIBUTION USING FOR- WARD KLD AND REVERSE KL KLD. -->
![](img/opd/figure-000.png)

(本图源自 MiniLLM，许多相关文章都有类似图)

在传统知识蒸馏中，无论是神经网络时期、Bert 时期、LLM 离线蒸馏时期和后训练 OPD 时期，大家几乎有一个共识：forward KL 是 mean seeking，i.e. 即它会让 student 模型全面模仿 teacher 的分布，因此在概率高峰它难以尖锐，而在概率低谷它也无法贴近。而 reverse KL 是 mode-seeking，即它只模仿 teacher 概率最高的位置，无法拟合多峰分布。另出于这样的考虑，许多文章开始尝试两种 KL divergence 的自适应加权结合，从他们各自的理论和故事角度，其加权结合均有一定的合理性和实际效果，但终究是启发式的想法。

尽管存在过度自信等缺点，Benchmark 上的优异表现让 RKL 仍成为了 OPD 的首选，但事实上 RKL 可能还存在其他优势。下图依照 Adaptive KL ([Wu et al., 2025](https://arxiv.org/pdf/2404.02657)) 的 toy example 进行词表拓展，虽然 FKL 和 RKL 在优化上看似殊途同归，但在现代 LLM 的大词表 output space 下，RKL 使 student 对 teacher 的概率趋近明显是更优的。理论上 [Luong et al. (2026)](https://arxiv.org/pdf/2604.00223v1) 证明 RKL 降低了优化的复杂性，尤其是当 student 容量小于 teacher 的场景。

<!-- 这是一张图片，ocr 内容为：EPOCH 50 EPOCH 1 EPOCH 10 EPOCH 30 PROBABILITY 0.04 V150 0.02 0.02 0.02 0.02 0.02 0.00- 0.00 0.00 0.00 0.00 50 100 150 50 100 150 150 150 50 100 150 50 0 100 0 100 O 50 0 EPOCH 1 EPOCH 50 EPOCH 5 EPOCH 10 EPOCH 30 PROBABILITY 0.004 V1500 0.004 0.002 0.002 0.002 0.002 0.002 0.000- 0.000 0.000 0.000 0.000 1.0 1.5 0.5 1.0 0.5 1.0 0.5 0.0 1.01.5 1.5 1.5 0.5 1.5 0.5 1.0 0.0 0.0 0.0 0.0 X10 X103 X10 X10 X103 EPOCH 5 EPOCH 50 EPOCH 1 EPOCH 30 EPOCH 10 V15000 PROBABILITY 0.0004 0.0004 0.0002 0.0002 0.0002 0.0002 0.0002 0.0000 0.0000 0.0000 0.0000 0.0000 1.5 L.01.5 0.5 1.0 0.5 1.0 1.5 0.5 1.5 0.5 1.0 0.5 1.0 0.0 1.5 0.0 0.0 0.0 0.0 X104 X10 X104 X104 X104 EPOCH 1000 EPOCH 2500 EPOCH 50 EPOCH 500 LE-FPOCH 250 1E-5 LE- LE-5 V 三 150000 PROBABILITY 4 5.0 2 2 2.5 HATER HER HEREESTEROOBAODBOOBAODE 牛肉馆 0 0.0- 0 1.0 0.5 1.0 1.0 1.5 1.5 0.5 1.5 1.0 0.5 0.5 1.01.5 0.5 1.5 0.0 0.0 0.0 0.0 0.0 X105 X105 X105 X10 X105 EPOCH 5000 EPOCH 2000 EPOCH 1000 EPOCH 500 EPOCH 100 LE-5 LE-5* 1E-5------ LE- V3000000 PROBABILITY 2 1 0 0 0 2 I 2 2 O 2 0 0 0 3 X105 X105 X105 X10 X105 RKL FKL TEACHER -->
![](img/opd/figure-001.png)

(当输出空间不断变大，蒸馏收敛需要的 epoch 也变大，而 FKL 和 RKL 存在两种截然不同的趋近方式)

# 3 Better Estimators？
既然 RKL 可能仍是目前 OPD 最优的损失选择，是否有更优估计量来实现更稳定的蒸馏呢？有的同学，有的：

<!-- 这是一张图片，ocr 内容为：TABLE I, TOKEN FEVEL KL BSTINATORS. FOREVILY, UE OMIT THE CONDITENING OF THE OR PAST OR PAS AR STOP R THE ESTI E ESIMATORS BEING HIGHTED  IN BLUE ARE THE KL  ESTIMATORS THAT PROVIDE BOTH UNBIASED  VALUES ANBIASED [DF(TE , TRER)] EXPRESSION GRADIENT UNBIASED?MEMORY OVERALL UNBIASED? VALUE VOKL(TE,TTRER) XXXX KL(ME,TREF) ((LALO REVERSE KL:EXACT -LOGW KL(TE, TTREF) O(1)O 0 VOKL(TE, TREF) (LOG W) 2/2 ERRE[(LOGW) 2 / 2  O(1) VEKL(TREF, TE) KL(NO,TREF) O(1) LOGW+W+W-1 VOKL(NO, TRET) 人人入 K3++ KL(TE,TREF) O(1( I (-LOGW+W+1) VOKL(TE, TREF) K4 KL(TE, TREF) (1 (1 R.SG(-LOGW) VEKL(TE,TREF) KL(NE, TREF) O(K) ALG 1 TOPKREVERSEKL (LAI)O. KL(TREF,TE) FORWARD KL:EXACT 70KL(TREF,TO) EJEV TRER(J)[LOG W(J)] VOKL(TRET,TTE) KL(TREF, TE) (T)O K5 SG(W)LOGW+LOGTR O(KE) VEKL(TREF,TTE) KL(TREF, TE) ALG 2 TOPKFORWARDKL -->
![](img/opd/figure-002.png)

EMA-PG ([Zhang & Ba, 2026](https://arxiv.org/pdf/2602.04417)) 归纳了 [Schulman (2020)](http://joschu.net/blog/kl-approx.html) 对 RKL 的 K1-K3 形式估计量和 DeepSeekV32 ([DeepSeek-AI, 2025](https://arxiv.org/pdf/2512.02556)) 的 K3++ 估计量，分析其估计值和梯度上的 bias & variance，并提出了 K4 和 K5 估计量。通过计算上的修正，可以保证 sampled RKL 在损失值和梯度上的无偏。在 OPD 的研究中，Skill-SD ([Wang et al., 2026](https://arxiv.org/pdf/2604.10674)) 亦在附录 D.2 中讨论了 K3++ 的使用，作为一种对 K3 估计量的重要性加权。此外目前在其他 OPD 研究中，暂时未见这些更优估计量的应用 (知识截止于 26 年 6 月)。

上图中所谓 TopkReverseKL 和 TopkForwardKL 实为 sampled KL 和 Top-K 截断 KL 的结合。这一方法修正了 Top-K 截断 KL 理论上的有偏，亦降低了 sampled KL 的高方差，启示在小 k 截断下这一方法更好。

<!-- 这是一张图片，ocr 内容为：TOP-K KL GRADIENT ERROR VS. SAMPLING BATCH SIZE (SYNTHETIC I.D.SETTING) RELATIVE RMSE OF GRAD ERROR TOP-32 MASS TOP-32MASS0.8 TOP-32 MASS0.9 0.5 TOP-32MASS0.2 K0 K0 K0 K-0 10 1 10- K128 10-2 K128 W.K WW...  W...28 K128 100  102  103  104 10010102 103   104 10010102 103 104 103 102  102  102  103 104 BATCH TOKENS(SAMPLE SIZE) BATCH TOKENS(SAMPLE SIZE) BATCH TOKENS(SAMPLE SIZE) BATCH TOKENS(SAMPLE SIZE) K16.TOP-K K4,TOP-K K32.TOP-K K256,TOP-K K8,TOP-K K128,TOP-K K64,TOP-K K0(SAMPLED KL) K256,TRUNCATED K32,TRUNCATED K16.TRUNCATED K128,TRUNCATED K4,TRUNCATED K8. TRUNCATED K64,TRUNCATED CRITICAL SAMPLE SIZE AL  RERINET, BUT OUPERFORMS THNCATE;  WE       CRADERFORD A CETAH  ERTTICAMPERMSTE;  WE     BTARANE   -->
![](img/opd/figure-003.png)

[Fu et al. (2026)](https://arxiv.org/pdf/2603.25562) 在附录中对比了截断 Top-K KL 和 EMA-PG 式的尾部修正在 OPD 上的表现，但二者的表现差异正如上图，与 k 的取值以及分布的尖锐程度非常相关。究其复现代码，其也没有使用正确的 K4 估计量。

因此，利用这些估计量、尾部修正来提升 OPD 的稳定性， 仍然是工程实践中值得尝试的一个方向。尽管如此，sampled token KL 的高方差并没有因为计算方式上的理论效益而得到显著的降低，这一坏处仍然伴随着 on-policy 的 student 采样一直存在。好烦啊，那不用 logits 去算 KL 行不行？

# 4 Representation Distillation
## 4.1 传统方法的刻舟求剑：MSE 和 Cosine Loss
在 OPD 之前，off-policy 知识蒸馏早已雕花过了各种蒸馏：每一层的 hidden states 可以蒸馏，attention map 可以蒸馏，模型内部啥都可以蒸馏。他们往往作为 logits KL 的补充，以希望 student 和 teacher “我们意念合一！”，从而实现更好的表现。OPD survey ([Song & Zheng, 2026](https://arxiv.org/abs/2604.00626)) 亦期待通过表征蒸馏实现跨词表、跨模型的蒸馏。

传统的离线策略蒸馏最常用的表征蒸馏形式是最小均方误差 MSE 和余弦相似度损失 Cosine loss，即在我们得到 student 和 teacher 关于当前 token prediction 的全词表 logits 预测之前，我们取出其某一层 (不妨就最后一层吧) 的 hidden states，并直接计算两组向量之间的 MSE 或者 Cosine Loss，并反向传播来做最小化。

这个想法非常简单直接：数值或者语义方向上让 student 去趋近 teacher，训完包相似的。但是在大模型后训练场景下，问题产生了微妙的变化：动态训练已经改变了表征含义，强行近似无异于刻舟求剑。

首先，要能计算这两种 loss，我们需要这两对 hidden states 的维度数量一致，这就注定了它们只能适应相同尺度的模型蒸馏，即 teacher 可能是一个经过 SFT/RL 得到能力强化的 student 模型，正如 MiMO 的 MOPD 等研究中的场景 ([Ma et al., 2026](https://arxiv.org/abs/2606.30406))，这就无法适应 Qwen3 那样大蒸小的场景了。 诚然，academia 有削足适履的方法：Flex-KD ([Saadi & Wang, 2025](https://openreview.net/forum?id=aiMINHhIiQ)) 用维度相关性硬选出最相关的 n 个 hidden states 维度，从而可以跟 student 的维度数量 match。还有在训练之前学习到一个映射矩阵 ([Miles et al., 2024](https://arxiv.org/pdf/2403.06213))，可以把 teacher 和 student 之间做到语义转换。

但是这里会出现第一个刻舟求剑：无论是筛选出最相关硬切，还是通过桥梁矩阵映射，都假定了这个 mismatch 是可以通过某种静态机制矫正的。而 student 模型正在被动态训练和改变，机制变不变都不好。

并且直接计算这俩 loss 还有一个隐患：teacher 和 student 在 hidden states 上每一个特征的语义一致吗？事实上由于其超位特性，其语义的逐层传导类似于多约束线性规划 ([Xiong, 2026](https://arxiv.org/pdf/2603.01227)) 形式，做完后训练每个维度的语义强度和含义几乎已经发生了或大或小的改变，此时直接计算 MSE 或者是余弦相似度，都是在对着一个有偏目标做强行近似。

既然如此，有没有能够绕开这个问题的方法呢？

## 4.2 Centered Kernel Alignment: 不完美的相似度
Relational knowledge distillation ([Park et al., 2019](https://arxiv.org/pdf/1904.05068)) 就是用于解决这一类问题的办法，即我们不在每一个学习样本上做强行近似，而是根据样本间的相似性关系来实现蒸馏。Centered Kernel Alignment (CKA) ([Kornblith et al., 2019](https://arxiv.org/abs/1905.00414)) 就是这一类方法中典型而热门的方法。具体的，我们延续 [Dasgupta and Cohn (2025)](https://openreview.net/forum?id=IcVSKhVpKu) 的思路，首先取出两个模型某一层的 hidden states 向量，并将每个 token 视为一个样本，从而整个 batch 上的 token 拼出一个 hidden states 矩阵,$ H^s

 $和$ H^t
 $。对这两个矩阵归一化后自乘计算 Gram 矩阵 $ K $，即可计算 CKA：

$$
\mathcal{L}_{CKA} = 1\ -CKA(H^t,H^s) = 1\ - \ \frac{tr(K^tK^s)}{||K^t||_F\ ||K^s||_F}.
$$

这一损失形式类似于 Pearson correlation，只是依赖于 Hilbert-Schmidt Independence Criterion (HSIC)，在线形核下可以以上式进行等价简便计算，它的优化还等价于最大均值差异的优化 (PCKA ([Zhou et al., 2024](https://arxiv.org/pdf/2401.11824)))。

搞不懂这些复杂的数学，简单来说就是它测度了两个模型在一大堆 token 预测上的相似度：尽管基于不同的推理语境，但在相似的位置上两个模型在 hidden states 上的思考反应应该得是尽可能相似的。例如在一个问题回答里出现了 “1+1=2” 这个命题，在“2”这个位置上的预测反应，应该和在另一个回答中的“1+1=2”上相似。

它的好处在于自乘 Gram 矩阵之际，两个模型 hidden states 的 dimension mismatch 的问题被消除：

$$
K^t = \tilde{H}^t{\tilde{H}^t}^T, \ K_s = \tilde{H}^s{\tilde{H}^s}^T \in \mathbb{R}^{N\times N},
$$

$N$ 是样本量

从而这种损失设计天然适用于不同尺寸模型间的蒸馏：**只要是相对应的同一层，就可以蒸馏，维度无关**。

考虑到“同一层”的信息处理程度和语义相关，我们采取最后一层，即词表翻译成人话的前一层。这最后一层 hidden states 已经包含了整个模型对问题 prompt $ x $ 和当前前缀 $ s_t $ 下所有的思考+即将脱口而出的反应，无疑是最值得蒸馏的。并且由于不同尺寸模型间架构设计的差异，也让最后一层成为了避免层间错配的最优选择。

使用 CKA 或者表征蒸馏的好处还在于它实际上提升了蒸馏学习的上限：在使用 logits 计算 KL divergence 时，softmax 函数抹平了 logits 本身的量级强度，导致即便理想状态下概率质量已经完全对齐，但 logits 及其以前的模型内容都没对齐，即在 hidden states 表现为有一个常数相关的 error term，最终蒸馏“貌合神离”。

但使用这一方法同样有风险和缺点，有效不代表严谨，也是这些原因让我们最终放弃了这个项目的推进：

1. CKA 的分子分母都是 HSIC 估计量，其对样本的分布假设是独立同分布。而 batch 上的 token 可能包含了同一问题的不同 rollout，这将会造成样本的严重自相关。即便是通过数据并行和采样数实现 microbatch 上不存在这一现象，同一推理 sequence 内的 token 也是自相关生成的，即便对序列 token 的 hidden states 做一阶差分也无法根治。也就是说，这个损失目标从工程实现上是有偏的，从分布假设上来说并不严谨，并且容易想到的工程解决方法也无法根治。
2. CKA 的工程计算风险还存在于其 batch 级别的累加，并不能像 KL 那样自由。在实际训练中，所有 token 上的 KL divergence 都进行累加，直到整个 batch 完成后再进行梯度更新。但 CKA 作为 ratio 构造，其可加性不复存在，因此 batch 上 CKA 的工程实现实际上是 microbatch CKA 的均值。而其期望又因为 Jensen 不等式无法非线性传递，不等价于 batch 级 CKA，进而目标错配。这一问题延续自 [Dasgupta and Cohn (2025)](https://openreview.net/forum?id=IcVSKhVpKu)，但据我们所知从未被其及其相关论文讨论和解决。
3. 如果单独使用表征蒸馏，几乎都跳过了对词表翻译矩阵 (LM head) 的训练，只能在 tied weights 模型中通过影响 input embedding 层来间接影响 LM head。这也会造成一个刻舟求剑：模型的思考已经改变得更像 teacher，但是它的发声表达还是老样子，也就是没有学会如何像 teacher 一样将 latent reasoning 翻译成推理表达。而 CKA 依赖于对 hidden states 矩阵的 Gram 矩阵计算，这隐藏了词表上的语义漂移，即理论上可以存在一个正交矩阵让 LM head 存在漂移而不自知。从这两点我们总结：表征蒸馏或许可以让 student 和 teacher 意念合一，但其定位最好是作为 KL 的补充而非代替。
4. OPD 的 on-policy 采样是一把双刃剑，它既带来了 OPD 的诸多好处，也带来了最令人担忧的坏处之一：长推理下的 prefix bias。即 student 在原策略下生成 teacher 认为绝无可能的推理路径，反复反思 hmmm/wait 仍无法纠正，但在长推理回答长度下 teacher 还要被迫给每个 token 打分，相当于“死马当活马医”，以此监督实在回天无力。尽管 CKA 测度的是 token 预测间相似度，但如此多的噪声 token 也容易造成其数值上的偏差和不稳定，可能造成其虚高的现象，真正该学的信息难被重视。

因此，尽管表征蒸馏绕过了 KL 蒸馏的设计/估计等问题，但本身仍有不少缺陷无法否认：传统蒸馏方式 (MSE, cosine loss) 的刻舟求剑，CKA 工程实现上的分布假设失效、目标错配、语义漂移风险和虚高震荡。在表征蒸馏领域，目前可能仍然缺少一个真正能适应广泛 OPD 问题的损失设计。

## 4.3 表征蒸馏能够跨词表/跨模型吗？
不行。虽然表征蒸馏绕过了 LM head 直接对模型内部进行对齐，但跨词表/跨模型对 OPD 的关键制约并没有改变。最简单的例子，在推理中的一个数字，“120”，在一些模型的词表里面是一个 token，而在另一个模型里面可能是 {1, 2, 3} 三个 token，这样无论是 logits KL 还是 hidden states，都会存在一个向量要对应三个向量的问题。虽然最新研究通过马尔可夫条件概率可以将三个 token 的概率值归并到一个进行 ([Niu et al., 2026](https://arxiv.org/abs/2606.09456))，或者只取两个模型有“共同语言”的 token 划分而抛弃那些存在分歧的 token ([Sun et al., 2026](https://arxiv.org/abs/2605.07711))，但这些跨词表的研究方法解决的是广泛的 OPD 问题，而不是通过表征蒸馏解决。尽管我们使用 CKA 并不要求 token 级别的一一对应，但 CKA 依赖于样本一致，不同的 token 划分方式将导致在一条推理链上 token 数量和 index 都无法对应，在计算上就已经造成了困难。总而言之，利用表征蒸馏去做不同词表/模型的蒸馏，几乎终究绕不过 token 对应，因而无法解决问题。

# 5 Experiments and Analysis
继上文，我们介绍了 OPD 中不同的损失设计，包括全词表 RKL、Top-K 截断 RKL、sampled token KL 的多种估计量、EMA-PG 式尾部校准，还有表征蒸馏中的 CKA 损失，那么它们的表现究竟如何？

## 5.1 Math Reasoning
在此，我们选取 JustRL ([He et al., 2025](https://arxiv.org/abs/2512.16649)) 的两对 1.5B 模型 (Qwen 2.5 Math 为基) 在 Math 领域进行同架构强蒸弱训练，所有评估都以 32k 的推理长度采样 32 次以尽可能缓解统计噪声的影响。还采用了更准确的 Bayes@K ([Hariri et al., 2026](https://openreview.net/forum?id=PTXi3Ef4sT)) 作为指标，且画出了 90% 置信区间。从假设检验角度可以不严谨地认为，区间不重合约等于表现存在显出差异。

首先，从训练动态上看，纯 CKA 的训练进展显著落后于纯 KL OPD，这里的 sampled RKL 是 K4 估计量。 

<!-- 这是一张图片，ocr 内容为：AI ME24 AIME24 AMC23 AMC23 DEEPSEEK DEEPSEEK NEMOTRON NEMOTRON 0.88 0.775 0.70 0.87 0.50 0.750 0.68 0.86 0.45 0.85 0.725 0.66 22 10.84 22 0.700 0.675 0.82 0.35 0.60 0.650 0.81 0.58 0.625 0.30 0.80 O 50 150 150 50 200 50 150 100 200 150 50 200 100 100 100 200 TRAIN STEP TRAIN STEP TRAIN STEP TRAIN STEP SAMPLED KL_ONLY CKA PLUS SAMPLED KL CKA_PLUS_TOP50_KL TOP50 KL_ONLY CKA_ONLY FIGURE 3: THIS  ISURE SHOWS THE TRAINING PERFOM THE IN THE DISTLAFON FROM THE TEACHER MODEL -NEMON- A JUSTRL-DEEPSEEK-L.5B) TO STUDENT MODE((OPENMATH- NEMOTRON-1 SB AND DEEPSEEK-RI-DISTIL-QWEN--  WHERE I WITH MERELY CKA LOSS BRINGS SLIGHTLY WEAK RESULTS. -->
![](img/opd/figure-004.png)

而从最终评估结果来看，纯 on-policy CKA 的表现难以与 KL OPD 竞争，也低于 off-policy CKA 的表现。但值得一提的是，在 K4 RKL 和 Teacher's Top-K 蒸馏下，student 的训练结果是一定程度上超越教师的。

<!-- 这是一张图片，ocr 内容为：TABLE I:MAIN RESULLS ON THE DEEPSEEK AND NEMOTRON PAIRS, FACH CELL REPORS MEAN @32 WITH PARENTHESESES TEACHER:JUSTRL-DEEPSEEK-1.5B TEACHER:JUSTRL-NEMOTRON-1.5B MEAN@32 STUDENT:DEEPSEEK-R1-DISTILL-QWEN-1.5B STUDENT:OPENMATH-NEMOTRON-1.5B (PASS@32) HMMT26 AIME26 HMMT26 HMMT25 HMMT25 AIME26 AIME25 AIME25 30.49 14.96 53.85 30.00 48.33 20.94 14.90 23.13 (76.67) (39.39) (53.33) STUDENT (54.55) (46.67) (63.33) (63.33) (83.33) 38.13 62.50 36.35 24.15 21.46 35.04 38.44 59.38 (43.44) (63.33) (57.58) (39.39) (70.00) (83.33) TEACHER (83.33) (66.67) TRAINED MODELS 23.58 34.27 35.63 39.79 20.52 37.31 63.44 61.35 (73.33) (57.58) (86.67) (70.00) (60.00) (83.33) (36.36) SAMPLED K4 (50.00) 34.90 38.16 60.00 39.90 34.06 20.83 63.33 23.48 TEACHER TOP50 (60.61) (63.33) (39.39) (83.33) (46.67) (76.67) (73.33) (80.00) 37.12 37.92 34.90 22.19 63.85 22.54 35.42 60.21 (73.33) (83.33) (39.39) (60.00) (63.64) TEACHER TOP512 (46.67) (76.67) (83.33) 38.65 61.46 21.31 35.63 37.03 21.77 35.10 60.10 OFF-POLICY CKA (80.80) (39.39) (40.00) (86.87) (73.33) (73.33) (60.61) (66.67) 37.71 21.56 58.75 37.12 28.96 20.64 61.04 31.88 ON-POLICY CKA (73.33) (83.33) (33.33) (56.67) (66.67) (63.64) (50.00) (86.67) 35.31 38.85 34.79 37.97 23.67 21.25 61.56 62.60 +SAMPLED K4 (54.55) (43.33) (36.36) (86.67) (80.00) (66.67) (70.00) (66.67) 24.05 37.50 34.48 21.04 35.83 63.54 36.65 61.56 +TOP50 (83.33) (83.33) (73.33) (70.00) (36.36) (57.58) (46.67) (60.00) -->
![](img/opd/figure-005.png)

理论上 CKA 可以作为 KL OPD 的有效补充，但在实验分析中我们发现，二者在最后一层 hidden states 的梯度方向几乎正交，但尝试了多种加权方案之后结果往往是训练表现介于二者之间。在上表 On-Policy CKA + sampled K4 或者 Top50 的实验中，表现也仅仅比纯 KL OPD 实验微弱的提升，部分还存在下降。

我们当然也做了 K1/K5 的实验，以及 EMA-PG 式的尾部修正，就是 Top50 + K4 的实验：

1. K1 理论上梯度有偏、估计易崩溃，但加上 clipping 之后效果差不多；
2. K5 由于本身是 FKL 容易出现极值震荡，但训练效果并没有逊色太多；
3. 而 EMA-PG 虽然理论上优越，但也仅有微弱提升。

## 5.2 Clinical Reasoning
在医疗算法同学们关心的医疗诊断推理问题上，纯 CKA 倒是不落下风，合并损失也和 K4 及 Top-50 差不多，全都蒸进去了。但值得注意的是，出于成本考虑，这里我们使用的 healthbench grader 是 gpt-oss-120b，MF1=0.66。这个评估模型跟官方默认的 GPT4.1 ([Arora et al., 2025](https://arxiv.org/abs/2505.08775)) 存在一定差距，属于是一个偏弱/偏不靠谱的 grader。

<!-- 这是一张图片，ocr 内容为：DISTILLATION TABLE 2: ON-POLICY ENT MODEL TRAINED TE TEACHER MODEL(AQ-MEDAI/CLINALIGN-4B) TO THE RAW STUDENT FROM 7) ON CLINICAL DIAGNOSIS TASK. THE TRAINING AND TESTING DATASET IS ANISHA2102/RAR-MEDICINE AND (QWEN/QWEN3-4B-INSTRUCT-2507) OPENAIRHEALTHBENCH RESPECTIVELY WHILE THE TESTING JUDGE MODEL IS OPENAILGPT-OSS-120B. LE THE TES AXIS(CAPABILITY DIMENSIONS) MODEL OVERALL CONTEXT AWARENESS COMMUNICATION QUALITY INSTRUCTION FOLLOWING COMPLETENESS ACCURACY 0.3457 0.5397 0.3853 0.3569 0.4417 STUDENT 0.6583 0.5462 0.4638 0.3968 0.5058 0.6017 0.4904 TEACHER 0.4588 CKA 0.5519 0.3949 0.4982 0.4793 0.6250 0.5547 0.4589 0.3952 SAMPLED K4 0.4767 0.4983 0.6022 0.4791 0.5514 0.3895 0.6014 0.4575 0.5018 TOP 50 0.5628 0.3984 CKA+K4 0.4597 0.4777 0.6131 0.4957 CKA+TOP 50 0.4534 0.3910 0.5473 0.4993 0.5992 0.4787 -->
![](img/opd/figure-006.png)

## 5.3 大蒸小：数学推理
我们采用 CKA 作为损失的一大动机是其对 hidden states 维度无关，因此可以适配非同一尺寸的 teacher 对 student 进行蒸馏 (词表相同即可)。下表中实验用 OpenMath-Nemotron-7B 作为教师去教学 1.5B student 模型，CKA 和 K4 合并损失表现确实是比单独损失实验表现要好，但也仅仅蒸馏进去了一小部分。

<!-- 这是一张图片，ocr 内容为：TABLE 3:DISTILL KNOWLEDGE FROM A LARGER TEACHER (NVIDIA/OPENMATH-NEMOTRON-7B) TO A SMALLER R STUDENT MODEL (NVIDIA/OPENMATH-NEMOTRON-1.5B) IS TYPICALLY MORE DIFFICULT THAN STRONG-TO-WEAK DISTILLATION WITH A SAME ARCHITECTURE.HOWEVER,WITH THE COMBINED LOSS OF CKA AND K4 RKL, THE DISTILLATION RESULT IS BETTER THAN PURE CKA OR RKL DISTILLATION. MODEL AIME26 HMMT26 AIME25 HMMT25 0.3155 0.5373 0.4578 0.3078 STUDENT_1.5B 0.6549 0.5902 0.4157 0.4055 TEACHER_7B 0.4833 0.3039 0.5324 0.2951 CKA 0.3200 0.3235 0.5059 0.4755 SAMPLED K4 0.5480 0.3209 0.3314 0.5059 CKA+K4 -->
![](img/opd/figure-007.png)

事实上，OPD 领域内强蒸弱不是什么难事，真正难的是大蒸小。在 G-OPD ([Yang et al., 2026](https://arxiv.org/abs/2602.12125)) 等强 baseline 的实验中，大蒸小也常常无法成功，甚至可能存在能力的倒退。即便使用一个 Qwen3-4B 的强化教师去蒸馏一个 Qwen3-1.7B 的 student，往往也无法完全学习到 teacher 全部的表现，或许因为尺寸架构。并且，student/teacher 是选择 base/instruct，还是后训练后的模型，表现差异我们也暂无定论。但一个可能正确的结论是，尺寸架构差异越大，蒸馏效果可能越差；由于尺寸架构差异，再怎么蒸，student 和 teacher 能力始终会有一个 gap 存在。

例如下图，使用 14B teacher 去蒸馏 1.5B student 时，由于尺寸架构差异较大，虽然 CKA loss 在下降，但 hidden norm 对齐失败，训练出现异常，最终模型推理能力出现了较大幅度的倒退。

<!-- 这是一张图片，ocr 内容为：14B TEACHER 1.5B STUDENT (CKA-ONLY,469 STEPS) 105 CKA_LOSS STUDENT NORM 0.18 TEACHER_NORM 新疆国机械械整备一个人民规模糊模糊精致的整体中国机构,川第一种 0.16 100 0.14 MEAN NORM 95 CKA LOSS 0.12 HIDDENME 0.10 90 0.08 0.06 85 WUNNWIRMRUNNSUNMANSURNAMY 0.04 0 200 400 100 300 TRAINING STEP -->
![](img/opd/figure-008.png)

## 5.4 CKA 的训练动态分析
继上图的训练动态观察：CKA 做表征蒸馏的动机是希望蒸馏效果“意念合一”而非“貌合神离”，所谓“貌合神离”在理论上就是纯 KL 蒸馏会让 student 模型在输出概率质量上和 teacher 对齐，但 logits 和 hidden states 不对齐，即在 hidden states 上会存在一个常数相关的 error term。

这在实验分析上也证实了：有使用 CKA 进行蒸馏的强蒸弱实验，两个模型最后一层 hidden states 的 norm 会尽快对齐和贴合，而基于纯 KL 的始终会有稍大的 gap。

<!-- 这是一张图片，ocr 内容为：I VS KL LOSS-OPENMATH-NEMOTRON-1.5B-CKA ONLY, 468 STEPS HIDDEN NORM STUDENT NORM TEACHER NORM 100 CKA LOSS 0.008 98 0.006 WHIWHUNAUNANN HIDDEN MEAN NORM CKA LOSS 96 004 94 0.002 92 0.000 0 100 300 200 400 STEP -->
![](img/opd/figure-009.png)

<!-- 这是一张图片，ocr 内容为：-OPENMATH-NEMOTRON-1.5B-SAMPLED KL ONLY,468 STEPS HIDDEN NORM VS KL  LOSS 0.7 102 STUDENT NORM TEACHER NORM KL LOSS (SAMPLED (TOPK-0)) 0.6 100 0.4 HIDDEN MEAN NORM 98 LOSS 0.3 96 0.2 0.1 0.0 0.1 100 200 300 400 STEP -->
![](img/opd/figure-010.png)

在方法设计部分我们提到了 CKA 的实现应用存在样本自相关和 token 噪声的问题，这最终体现在训练中的 loss spike 上。事实上，CKA 的虚高和 spike 问题在传统表征蒸馏上并非新鲜事，解决办法主要是对样本进行筛选去除，这是改进 CKA 表现的一种思路，但在我们的尝试中还有一些有趣的现象。

首先，采样温度越高，loss 震荡越大。这是符合直觉的，因为采样温度越高，student 产生离谱 token 的概率也就越高，从而 teacher 监督的噪声比例也会随之增大。一旦有恶性样本出现，该 step 的 loss 就会激增。

<!-- 这是一张图片，ocr 内容为：CKA_LOSS 0.016 T-0.5 T-0.7 T-1.0 0.014 0.012 0.010 CKA_LOSS 0.008 0.006 MANMIN 0.004 0.002 80 40 20 100 STEP -->
![](img/opd/figure-011.png)

在实际观测中，loss spike 情况有时非常严重：CKA loss 可能单步内激增几十上百倍，但下一步又回归。

<!-- 这是一张图片，ocr 内容为：HS OPD TRAINING CURVES(1750 STEPS,FULL DATASET) 1.00 0.06 OKA LOSS 0.05 0.99 0.04 0.98 CKA SIMILARITY CKA LOSS 0.03 0.97 0.02 0.96 0.01 0.95 CKA SIMILARITY 0.00 0.94 GRAD NORM 1.75 1.50 1.25 GRADIENT NORM 1.00 0.75 0.50 0.25 LUNTERI 0.00 0 250 1000 1750 500 750 1500 1250 TRAINING STEP -->
![](img/opd/figure-012.png)

对异常 token 分析，发现引发 spike 的 token 大多是一些标点符号和表示语义承接的词，几乎与推理无关。 但工程上可以通过数据并行和采样数来解决自相关问题，loss spike 得到较大缓解；以及对 hidden states 做逐 token 的一阶差分，由于目标形式的根本改变，loss spike 也几乎消失了，蒸馏效果几乎没有差异。

<!-- 这是一张图片，ocr 内容为：OPD: (RED) (DASHED) NEMOTRON HS_O (BLUE) LED KL CKA_ONLY (SOLID) CKA+SAMP BASE SEGDIF VS 3 MEAN,95% CI) TRAIN/LOSS VS STEP (TOTAL DISTILLATION LO BAYES&32  (AIME24+AMC23 M N LOSS) EVA PONFIG CKA SAMP LED K I 0.78 LAAM 0.76 AYES PASSE32 .74 0.72 0.70 10.0 SEQDIFF 200 100 50 100 200 ROLLOUT_ID (TRAIN STEP) VUL BUPUM0.7073 LAST-O.7602 BEST-0.7663 D629 -->
![](img/opd/figure-013.png)

除了一阶差分，CKA 的计算形式也有一些变体，例如其相似度可以取根号式或者平方式，最终效果基本一样。

# 6 Conclusion
OPD 中传统使用的 RKL，尽管有一些缺点，但理论和实践上仍然是最优选择。从估计量和损失构造角度，新估计量如 K4/K5 可能存在微弱优势，但 sampled token KL 方差依然大；top K 及其修正变体可能理论上存在优势且有时表现略微更好，但 bias/variance 问题依然存在。实践上推荐大家使用，但它们并未解决本质问题。

表征蒸馏 OPD 并不好做：传统的 MSE 和 Cosine Loss 会有刻舟求剑问题，而新宠 CKA 在 OPD 场景表现虽然尚可也仍然水土不服。合适的损失构造仍有待探索，但表征蒸馏不应作为 KL 的替代，而是补充。

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


` };
