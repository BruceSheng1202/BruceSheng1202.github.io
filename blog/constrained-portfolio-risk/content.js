window.POST_CONTENT = {
  en: String.raw`<div class="editor-note">
<p class="editor-note-label">Editor's note</p>
<p>Three years ago I was still doing high-dimensional constrained portfolio work in MATLAB and R. The code is in <a href="https://github.com/microfan1/CROWN">CROWN</a>, the paper came out in <a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2025.2602832">JASA</a>, and the Misc section of my homepage still mentions it. What I was reading at the time was a printout a senior student handed me: Jianqing Fan's <a href="https://www.taylorfrancis.com/books/mono/10.1201/9780429096280/statistical-foundations-data-science-jianqing-fan-runze-li-cun-hui-zhang-hui-zou">Statistical Foundations of Data Science</a>. Before graduating I helped proofread its Chinese translation, which is now <a href="http://fj.tsxcfw.com/bookshow.asp?id=3230017">in print</a>.</p>
<p>LLM agents are strong at mathematics and code, work like this needs no multi-GPU cluster, and I am still learning the same advanced statistics. So a question kept nagging: can the strongest AI write a paper good enough for one of the four top statistics journals?</p>
<p>Taking the high-dimensional portfolio problem as the topic, I wrote my requirements out as a prompt and let Claude Opus 5.5 Max run in autoresearch mode. Within a day it produced a paper, presented here as a blog post. I rarely adjusted the agent's behaviour or decisions: the idea, the experiments, the paper and this blog post were all written by Claude. I only did a rough review and proofread, so I will not claim the contribution as mine.</p>
<p>The study looks plausible, but I cannot vouch for whether it has real research or practical value. What bothers me more is something else: if a study like this can appear in a day, could one determined person submit to a top statistics journal every day, and would the review system buckle under the load? As human experts find it harder to keep up with AI-generated work, how to separate validity from contribution and decide acceptance is the question this post left me with.</p>
</div>

Fit a minimum-variance portfolio on a year of daily returns and the optimizer tells you how risky the portfolio is. That number is too low, often by a lot. In one of the examples below — 500 stocks, 250 days, short sales banned — the optimizer reports about one eighteenth of the variance the portfolio actually carries.

The reason is the same one that makes training error lower than test error. The covariance matrix is estimated from the same data the optimizer uses to pick the weights, and the optimizer is very good at finding combinations whose risk the sample happens to underestimate. Without constraints the size of this gap has been known for a long time. Real portfolios are constrained — no short sales, position limits, sector bounds, tracking-error budgets — and once a constraint binds, the classical formula no longer applies.

This post is about what replaces it. The answer fits in one line: count the holdings, not the assets.

# 1 Takeaways

- **Count the holdings, not the assets.** With n days of data, divide the in-sample variance by (1 − d/m)<sup>2</sup>, where m = n − 1 and d is the number of stocks held minus one. In general d is the dimension of the face of the constraint set on which the optimizer lands, and the optimizer reports it for free.
- **With fewer stocks than days, this is provably accurate** for every covariance matrix and every set of linear constraints, with an error that shrinks like 1/√m. In 20 simulated markets the corrected value is within 2.6% of the true variance on average; the uncorrected value is up to 150 times too low.
- **Counting all the assets overshoots.** The classical correction, which uses the number of stocks p instead of d, puts the risk 1.3 to 3.1 times too high at p/n = 0.5 and up to 18.6 times too high at p/n = 0.8. Once p ≥ n it is not defined at all.
- **With more stocks than days, one number decides whether the portfolio breaks.** The optimizer finds a portfolio with zero in-sample risk exactly when the statistical dimension of the constraint cone exceeds m. Below that line the correction still works.
- **For tracking-error mandates, the same idea sets the budget.** On NASDAQ data, capping the in-sample tracking error at the target overshoots it by 27% to 132% at the median; calibrating the cap with d brings this to 0% to 12%. With more stocks than days there is a floor that no budget can push below.

# 2 Why the optimizer's number is too low

Two things have to happen together.

First, the covariance matrix is estimated, and estimated badly. For 500 stocks it has 125,250 distinct entries; a year of daily data gives 250 × 500 = 125,000 numbers. Every entry carries real noise.

Second, the same data both choose the portfolio and grade it. Fix a portfolio in advance and its sample variance is an unbiased estimate of its true variance: sometimes high, sometimes low, right on average. The bias comes from choosing. The simplest version: take 500 stocks with exactly the same true variance, estimate each variance from 250 days, and pick the stock with the smallest estimate. In a quick simulation the winner's estimated variance averages 0.75 of its true value. No single estimate is wrong; the winner is just the stock whose noise happened to help.

A minimum-variance optimizer does this on a much larger scale. It does not only pick quiet stocks; it also builds hedges out of correlations that look strong in the sample. Two stocks whose sample correlation is 0.8 but whose true correlation is 0.3 look like a good long–short pair. Out of sample the hedge does not hedge.

Without constraints the damage is known precisely. The in-sample variance is too low by a factor of about 1 − p/n ([El Karoui, 2010](https://arxiv.org/abs/1211.2917)), and the variance you actually bear is too high by about 1/(1 − p/n), so the two differ by about (1 − p/n)<sup>−2</sup>. Dividing by that factor was proposed as a correction for risk forecasts ([Shepard, 2009](https://arxiv.org/abs/0908.2455)). At p/n = 0.4 the gap is a factor of 2.8; at p/n = 0.8 it is 25.

Constraints are known to help ([Jagannathan and Ma, 2003](https://www.nber.org/papers/w8922)), but the algebra behind the classical formula needs a closed-form solution, and with inequality constraints there is none. The standard workaround is the jackknife ([Basak, Jagannathan and Ma, 2009](https://doi.org/10.1287/mnsc.1090.1001)): refit the portfolio n times and see how far the refits scatter.

# 3 Count the face, not the assets

With three stocks and no short sales, every allowed portfolio is a point in a triangle. Wherever the optimizer lands, it sits on one face of that triangle: in the interior (all three held), on an edge (two held) or at a corner (one held). On that face the problem looks unconstrained. A stock that is not held has its weight pinned at zero; the others slide freely. The number of free directions is d = holdings − 1.

![Long-only weights for three stocks form a triangle](img/figure-001.svg)

***Figure 1** The optimizer lands on an edge: two stocks are held, one direction is free, so d = 1.*

That suggests replacing the classical p − 1 free parameters by d:

**R<sub>out</sub> ≈ R<sub>in</sub> / (1 − d/m)<sup>2</sup>, with m = n − 1,**

where R<sub>in</sub> is the in-sample variance the optimizer reports and R<sub>out</sub> is the variance the portfolio really has. The rule extends to any linear constraints. With upper and lower bounds on each weight, d counts the weights strictly between their bounds, minus one; every extra constraint that binds, such as a sector limit at its cap, removes one more direction.

Back to the opening example: 500 stocks with AR(1) correlations, 250 days, short sales banned. The optimizer holds about 192 stocks, so d/m ≈ 0.77. The in-sample variance is 1/18.6 of the truth, and dividing by (1 − 0.77)<sup>2</sup> recovers the truth to within 0.1% on average. Counted in assets the problem looks hopeless, with p/n = 2; counted in free directions it is not.

The obvious objection: d is chosen by the data. We just saw that choosing creates bias, so why should a chosen number correct it? The lasso has the same structure. Which coefficients are non-zero is decided by the data, and yet the lasso's degrees of freedom equal the number of non-zero coefficients ([Tibshirani and Taylor, 2012](https://doi.org/10.1214/12-AOS1003)). The constrained portfolio is a least-squares problem in disguise, and d is exactly its degrees of freedom: near the solution, the face does not change under small perturbations of the data, and for Gaussian returns that holds with probability one.

What was missing was a proof that this works where portfolios live: random returns rather than a fixed design, and a hard constraint rather than a penalty. [Bellec (2023)](https://arxiv.org/abs/2008.11840) proved the (1 − df/n)<sup>−2</sup> estimate for convex penalties that take finite values, which rules constraints out. We reach constraints as the limit of steeper and steeper penalties. Two results come out:

- **A finite-sample bound.** For p < n, the expected relative error of the corrected estimate is at most C/√m, where C depends only on p/n — not on the covariance matrix, not on the constraints, and not on whether the constraints bind in the population.
- **An exact identity.** It says why the correction works: the bias of R<sub>in</sub> is of first order, about 2d/m in relative terms, and the bias of the corrected estimate is of second order.

# 4 How accurate is it?

Figure 2 covers 20 simulated markets: independent stocks, AR(1) correlations, and one- and three-factor markets with all-positive or mixed-sign loadings. Before correction, the ratio of in-sample to true variance falls almost exactly on the curve (1 − d/m)<sup>2</sup>, whatever the market: the size of the bias depends on d/m and on nothing else. After correction every point sits at 1, including the one at d/m = 0.92, where the in-sample variance was 150 times too low.

![Accuracy of the correction in 20 simulated markets](img/figure-002.svg)

***Figure 2** Open circles: in-sample over true variance. Filled: corrected over true variance. Every market, before and after.*

Using p instead of d is not a harmless approximation:

| Market | p/n = 0.2 | p/n = 0.5 | p/n = 0.8 |
| --- | --- | --- | --- |
| Independent stocks | 1.01 / 0.99 | 1.35 / 1.02 | 4.26 / 1.02 |
| AR(1) | 1.10 / 1.00 | 1.85 / 1.00 | 7.88 / 1.02 |
| One factor, mixed signs | 1.13 / 1.00 | 1.87 / 1.01 | 7.04 / 0.99 |
| One factor, all positive | 1.36 / 1.00 | 3.13 / 1.00 | 18.60 / 1.00 |

*Mean of estimate over true variance: classical correction (counts p) / ours (counts d). n = 250, 400 samples per cell.*

The two agree only when almost every stock is held, as for independent stocks at p/n = 0.2. The worst case is the market that looks most like a real equity market — every stock loads positively on the market factor — where the optimizer holds only 17% to 35% of the stocks. There the classical correction is further from the truth than no correction at all.

The 1/√m rate is visible too. In four markets with m from 50 to 400, √m times the mean relative error stays between 0.57 and 1.27.

These are averages. In a single sample the corrected value still scatters: its standard deviation across samples grows from 0.10 at small d/m to 0.49 at d/m = 0.92. The bound in the theorem is a bound on the expected error, not a prediction interval.

# 5 More stocks than days

With n days of data, demeaned returns span only m = n − 1 directions. With more stocks than that, some portfolios do not move at all in the sample, and an unconstrained optimizer reports zero risk. Constraints may or may not rule those portfolios out. When they do not, the in-sample variance is exactly zero, the true variance is not, and no correction can help.

Whether this happens is decided by one number: the statistical dimension δ of the constraint cone after whitening by the covariance, a measure of how large the constraint set looks to random noise ([Amelunxen et al., 2014](https://arxiv.org/abs/1303.6672)). If δ is above m, zero in-sample risk is almost certain; if δ is below m, it almost never happens; the switch takes place within a window of order √δ.

![Probability of zero in-sample risk against m over the statistical dimension](img/figure-003.svg)

***Figure 3** δ ranges from 17 to 199 across the five markets, but all five curves cross 1/2 at m/δ between 0.96 and 1.00.*

Three markets make it concrete:

- **Independent stocks, long-only.** δ = p/2, so the portfolio breaks at p = 2m. This is the threshold that [Kondor, Papp and Caccioli (2017)](https://arxiv.org/abs/1612.07067) found with the replica method; their equations also imply our correction for this case, though they did not state it. Up to the threshold the long-only portfolio's true variance stays within a factor π of the best possible, while the unconstrained portfolio blows up at p = m.
- **AR(1) correlations.** δ ≈ 0.30p, so the threshold is at p ≈ 3.3m.
- **One factor, all loadings positive.** δ grows only like log p. The market factor cannot be hedged without short positions, so the long-only portfolio essentially never breaks.

When loadings take both signs, the factor can be hedged inside the long-only set and δ grows in proportion to p again. The NASDAQ data show exactly this. With quarterly windows the long-only portfolio reported zero risk in 3%, 19% and 28% of windows for 500, 1,000 and 1,455 stocks — all of them windows ending by early March 2020, in the calm period when many stocks had market loadings near zero or below it.

Below the threshold the correction keeps working. The convex Gaussian min–max theorem (see [Thrampoulidis, Abbasi and Hassibi, 2018](https://arxiv.org/abs/1601.06233)) gives the limits of both variances, without assuming that the unconstrained optimum satisfies the constraints. In that limit the correction is exact, and it carries over to the observed d at a rate of (log m / m)<sup>1/4</sup>.

# 6 Tracking-error budgets

An active manager maximizes an alpha signal subject to a cap on tracking error against a benchmark. Practitioners know that realized tracking error comes out above the ex-ante number — by 1.3 to 1.4 times in backtests with commercial risk models ([Saxena and Stubbs, 2012](https://scinapse.io/papers/2068160108)) — and add a buffer to the ex-ante estimate ([De Nard, Ledoit and Wolf, 2025](https://econ.uzh.ch/dam/jcr:8f7e20cf-e9a7-4fde-87cb-aef82de26404/jpm_2025.pdf)).

The same correction says how large the buffer should be: choose the in-sample cap so that TE<sub>in</sub> / (1 − d/m) equals the target, where d is the face dimension of the tracking-error portfolio. Tracking error is a standard deviation, so the factor is not squared.

| Days n | Stocks p | Target | d/m | Cap = target | Calibrated |
| --- | --- | --- | --- | --- | --- |
| 252 | 50 | 2% | 0.19 | 1.38 | 1.12 |
| 252 | 100 | 2% | 0.37 | 1.68 | 1.11 |
| 252 | 200 | 2% | 0.67 | 2.08 | 1.09 |
| 252 | 200 | 6% | 0.37 | 1.35 | 1.00 |
| 126 | 100 | 2% | 0.72 | 2.32 | 1.06 |

*NASDAQ, fewer stocks than days: median over monthly windows of realized tracking error over target. Equal-weight benchmark, momentum signal; d/m at the calibrated cap.*

A constant buffer could not do this. Across settings the naive cap overshoots by a factor between 1.27 and 2.32, because d/m varies; the calibrated cap stays between 1.00 and 1.12.

With more stocks than days, something new appears: a floor. As the in-sample budget shrinks to zero, the portfolio converges to one that replicates the benchmark exactly in sample. Under long-only constraints that portfolio holds exactly n stocks — it did in all 220 simulated cases. Its realized tracking error is positive, so shrinking the budget cannot push realized tracking error below that floor. In-sample numbers cannot see the floor either: at the floor the in-sample tracking error is 0 and d = m, so TE<sub>in</sub> / (1 − d/m) is 0/0. A practical warning sign is d/m > 0.85 at the calibrated cap; on NASDAQ it flagged 90% to 98% of the windows whose target was below the floor.

![Realized tracking error over target on NASDAQ, against target over floor](img/figure-004.svg)

***Figure 4** The calibrated portfolio lands near max(target, floor); capping in-sample tracking error at the target overshoots everywhere.*

Where the floor lies below the target in most windows, the calibrated portfolio realizes 0.94 to 1.05 times the target, against 1.38 to 1.78 for the naive cap. Below the floor it realizes 1.02 to 1.07 times the floor, which is about as good as it gets. The floors are not small: medians of 5.4% and 7.3% a year with quarterly windows, 3.0% with one-year windows and 500 stocks, and 1.7% among the 500 least volatile stocks — well inside common mandate ranges.

# 7 What it does not fix

- **It does not raise the Sharpe ratio.** The weights are the same; the correction changes what you believe about their risk. Scaling a portfolio to a volatility target with an honest number does not change its Sharpe ratio either. It makes realized volatility land where you aimed.
- **A common daily shock.** If all stocks share a random scale each day (elliptical returns, volatility clustering), the formula comes out 7.5% to 38% too low in our simulations. Refitted leave-one-out — the jackknife — stays between 0.95 and 1.07 there, so report both.
- **Drift.** On NASDAQ the corrected minimum-variance forecast is still 1.1 to 1.5 times too low against an equal-weight reference. A split-sample forecast, which has no optimization bias at all, is also too low, by 1.02 to 1.41 times: the covariance moves between the estimation window and the holding period, and nothing computed inside one window can see that.
- **Factor risk models.** Most institutions do not optimize on the sample covariance. The theory covers the sample covariance, fixed ridge shrinkage and, as a proposal, exponential weights. What plays the role of d for a portfolio built on a factor-model or nodewise estimate is open.
- **The data.** The NASDAQ panel (1,455 stocks, January 2018 to May 2023, from [skfolio-datasets](https://github.com/skfolio/skfolio-datasets)) contains survivors only and is distributed as test data. The empirical part illustrates the method; it does not evaluate these stocks. The theory assumes Gaussian, independent returns.

# 8 How to use it

1. Solve the portfolio problem exactly, so that d is an exact count. Near d/m = 0.9, one miscounted holding moves the corrected estimate by several percent.
2. Report R<sub>in</sub> / (1 − d/m)<sup>2</sup> together with d/m. When d/m is close to 1, the in-sample number carries almost no information.
3. For tracking-error mandates, set the in-sample cap so that TE<sub>in</sub> / (1 − d/m) equals the target, and treat d/m > 0.85 as a sign that the target is at or below the floor.
4. Where you can afford it, also report the refitted leave-one-out estimate and a split-sample forecast, as checks against heavy tails and drift.

# References

1\. Kondor, I., Papp, G., and Caccioli, F. (2017). Analytic solution to variance optimization with no short positions. Journal of Statistical Mechanics. arXiv:1612.07067. [https://arxiv.org/abs/1612.07067](https://arxiv.org/abs/1612.07067)

2\. El Karoui, N. (2010). High-dimensionality effects in the Markowitz problem and other quadratic programs with linear constraints: Risk underestimation. The Annals of Statistics. arXiv:1211.2917. [https://arxiv.org/abs/1211.2917](https://arxiv.org/abs/1211.2917)

3\. Shepard, P. G. (2009). Second Order Risk. arXiv:0908.2455. [https://arxiv.org/abs/0908.2455](https://arxiv.org/abs/0908.2455)

4\. Jagannathan, R., and Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. The Journal of Finance. NBER Working Paper 8922. [https://www.nber.org/papers/w8922](https://www.nber.org/papers/w8922)

5\. Basak, G. K., Jagannathan, R., and Ma, T. (2009). Jackknife Estimator for Tracking Error Variance of Optimal Portfolios. Management Science. [https://doi.org/10.1287/mnsc.1090.1001](https://doi.org/10.1287/mnsc.1090.1001)

6\. Tibshirani, R. J., and Taylor, J. (2012). Degrees of freedom in lasso problems. The Annals of Statistics. [https://doi.org/10.1214/12-AOS1003](https://doi.org/10.1214/12-AOS1003)

7\. Bellec, P. C. (2023). Out-of-sample error estimation for M-estimators with convex penalty. Information and Inference. arXiv:2008.11840. [https://arxiv.org/abs/2008.11840](https://arxiv.org/abs/2008.11840)

8\. Amelunxen, D., Lotz, M., McCoy, M. B., and Tropp, J. A. (2014). Living on the edge: Phase transitions in convex programs with random data. Information and Inference. arXiv:1303.6672. [https://arxiv.org/abs/1303.6672](https://arxiv.org/abs/1303.6672)

9\. Thrampoulidis, C., Abbasi, E., and Hassibi, B. (2018). Precise error analysis of regularized M-estimators in high dimensions. IEEE Transactions on Information Theory. arXiv:1601.06233. [https://arxiv.org/abs/1601.06233](https://arxiv.org/abs/1601.06233)

10\. Saxena, A., and Stubbs, R. A. (2012). An Empirical Case Study of Factor Alignment Problems Using the USER Model. The Journal of Investing. [https://scinapse.io/papers/2068160108](https://scinapse.io/papers/2068160108)

11\. De Nard, G., Ledoit, O., and Wolf, M. (2025). Improved Tracking-Error Management for Active and Passive Investing. The Journal of Portfolio Management. [https://econ.uzh.ch/dam/jcr:8f7e20cf-e9a7-4fde-87cb-aef82de26404/jpm_2025.pdf](https://econ.uzh.ch/dam/jcr:8f7e20cf-e9a7-4fde-87cb-aef82de26404/jpm_2025.pdf)

12\. skfolio-datasets: datasets distributed with the skfolio library (NASDAQ daily prices). [https://github.com/skfolio/skfolio-datasets](https://github.com/skfolio/skfolio-datasets)`,
  zh: String.raw`<div class="editor-note">
<p class="editor-note-label">编者按</p>
<p>三年前我还在用 MATLAB 和 R 做高维带约束的投资组合，代码在 <a href="https://github.com/microfan1/CROWN">CROWN</a>，论文后来发在 <a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2025.2602832">JASA</a>；主页的 Misc 里还留着这段经历。当时翻的是学长打印出来的范剑青《<a href="https://www.taylorfrancis.com/books/mono/10.1201/9780429096280/statistical-foundations-data-science-jianqing-fan-runze-li-cun-hui-zhang-hui-zou">数据科学统计基础</a>》，毕业前我帮忙校对了这本书的中译本，<a href="http://fj.tsxcfw.com/bookshow.asp?id=3230017">译本</a>已经出版。</p>
<p>LLM agents 在数学和代码上已经这么强，这边做研究又不需要多卡 GPU，而我还在学这些高等统计学，于是不免好奇：最强的 AI 能不能写出一篇合格的统计四大顶刊论文，从 idea 到成稿？</p>
<p>顺着高维统计组合这个题目，我把要求写成 prompt，让 Claude Opus 5.5 Max 在 autoresearch 模式下跑，一天之内就产出了一篇论文，并以 blog 的形式放在这里。整个过程我很少干预它的行为和决定：idea、实验、论文、乃至这篇 blog 都是它写的，我只做了粗略的审核和校对，所以不愿意把这份 contribution 算在自己头上。</p>
<p>这项研究看起来像那么一回事，但我很难保证它真的有研究价值和实用价值。更让我在意的是另一件事：如果一天就能产出这样一项研究，是不是一个"卷王"可以一天投一篇统计顶刊，审稿系统很快就会承受不住？当人类专家越来越难应对 AI 生成的研究成果，怎么分辨一篇论文的 validity 和 contribution、并据此决定 accept 还是 reject，是我写这篇 blog 时真正好奇的研究问题。</p>
</div>

用一年的日收益率算一个最小方差组合，优化器会顺便告诉你这个组合的风险有多大。这个数是偏低的，而且常常低得离谱。后面有个例子：500 只股票、250 天数据、禁止卖空，优化器报出来的方差只有组合实际方差的约 1/18。这个例子后面还会回来。

原因和"训练误差总比测试误差好看"是一回事。协方差矩阵是用数据估出来的，优化器又用同一份数据去挑权重，它特别擅长找到那些"在这份样本里风险恰好被低估"的组合。不加约束时，这个差距有多大早就算清楚了。可实际的组合都有约束：禁止卖空、单只上限、行业比例、跟踪误差预算。约束一旦顶住，经典公式就不再适用。

这篇讲的是用什么来代替它。答案一句话：数持仓，不数资产。

# 1 结论

- **数持仓，不数资产。** 有 n 天数据时，把样本内方差除以 (1 − d/m)<sup>2</sup>，其中 m = n − 1，d 是持仓数减一。一般地，d 是优化器落在约束集上那个"面"的维数，优化器算完就能直接读出来，不用另外估，也不用把组合重新拟合一遍。
- **股票比天数少时，这个校正可以严格证明是准的**，对任意协方差、任意线性约束都成立，误差按 1/√m 缩小。在 20 个模拟市场里，校正后平均与真实方差相差不超过 2.6%；不校正则最多低估 150 倍，整整差了两个数量级。
- **按资产数校正会矫枉过正。** 经典校正用股票数 p 而不是 d：p/n = 0.5 时把风险高估 1.3 到 3.1 倍，p/n = 0.8 时最多高估 18.6 倍；p ≥ n 时公式直接失去定义。
- **股票比天数多时，一个数决定组合会不会崩。** 当约束锥的统计维数超过 m，优化器就会找到一个样本内风险为零的组合。低于这条线时，校正依然有效。
- **跟踪误差预算也能这样定。** 纳斯达克数据上，直接把样本内跟踪误差上限设成目标，中位数超标 27% 到 132%；用 d 校准后只超 0% 到 12%。股票比天数多时，还存在一个预算定得再紧也压不下去的"地板"，第 6 节会具体算。

# 2 优化器报的数为什么偏低

需要两件事同时发生。

第一，协方差矩阵是估出来的，而且估不准。500 只股票的协方差矩阵有 125,250 个不同的元素，一年日数据只有 250 × 500 = 125,000 个数。每个元素都带着不小的噪声。

第二，同一份数据既用来挑组合，又用来给组合打分。如果组合是事先定好的，它的样本方差就是真实方差的无偏估计：有时高、有时低，平均下来是对的。偏差来自"挑"。举个最简单的例子：500 只股票的真实方差完全相同，用 250 天数据分别估方差，然后挑出估计值最小的那只。我简单模拟了一下，被挑中那只的估计方差平均只有真实值的 0.75。每个估计本身都没错，只是被挑中的恰好是噪声帮了忙的那一只，换一份数据就是另一只。

最小方差优化器把这件事放大了很多倍。它不只挑波动小的股票，还会利用样本里"看起来很强"的相关性去构造对冲。两只股票样本相关系数 0.8、真实只有 0.3，看起来是一对很好的多空组合；到了样本外，这个对冲根本对冲不住，因为 0.8 里有一大半是噪声。

不加约束时，偏差的大小是精确知道的。样本内方差被压低约 1 − p/n 倍（[El Karoui, 2010](https://arxiv.org/abs/1211.2917)），你实际承担的方差被抬高约 1/(1 − p/n) 倍，两者相差约 (1 − p/n)<sup>−2</sup>。用这个因子去修正风险预测，早就有人提过（[Shepard, 2009](https://arxiv.org/abs/0908.2455)）。p/n = 0.4 时差 2.8 倍；p/n = 0.8 时差 25 倍。

大家也早就知道约束有帮助（[Jagannathan and Ma, 2003](https://www.nber.org/papers/w8922)）。可经典公式背后的推导需要封闭解，有了不等式约束就没有封闭解。于是改用 jackknife（[Basak, Jagannathan and Ma, 2009](https://doi.org/10.1287/mnsc.1090.1001)）：把组合重新拟合 n 次，再看这些重拟合结果的离散程度。

# 3 数面，不数资产

三只股票、禁止卖空时，所有允许的组合构成一个三角形。优化器无论落在哪里，都在三角形的某个"面"上：内部（三只都持有）、某条边（持有两只）或某个顶点（只持有一只）。在这个面上，问题看起来就是没有约束的：没持有的股票权重被钉在 0，其余的可以自由滑动。自由方向的个数 d = 持仓数 − 1，三只股票时就是 2。

![三只股票的禁止卖空组合构成一个三角形](img/figure-001-zh.svg)

***图 1** 优化器落在一条边上：持有两只股票，只有一个自由方向，所以 d = 1。*

于是很自然地，把经典公式里的 p − 1 个自由参数换成 d：

**R<sub>out</sub> ≈ R<sub>in</sub> / (1 − d/m)<sup>2</sup>，其中 m = n − 1。**

R<sub>in</sub> 是优化器报出的样本内方差，R<sub>out</sub> 是组合真实的方差。这个规则可以推广到任意线性约束：如果每只股票的权重有上下限，d 就是严格夹在上下限之间的权重个数减一；每多一个顶格的约束（比如某个行业比例正好顶到上限），就再少一个自由方向。

回到开头的例子：500 只股票、AR(1) 相关、250 天、禁止卖空。优化器持有约 192 只，d/m ≈ 0.77。样本内方差只有真实值的 1/18.6，除以 (1 − 0.77)<sup>2</sup> 之后，平均与真实值相差不到 0.1%。按资产数算，p/n = 2，问题看起来没救了；按自由方向算，并非如此。

一个显然的疑问：d 本身是数据挑出来的。前面刚说过"挑"会制造偏差，凭什么用一个挑出来的数去做校正？lasso 有一模一样的结构：哪些系数非零是数据决定的，但 lasso 的自由度恰好就等于非零系数的个数（[Tibshirani and Taylor, 2012](https://doi.org/10.1214/12-AOS1003)）。带约束的组合本质上是一个最小二乘问题，d 正是它的自由度：在解的附近，数据稍微扰动，落点所在的面不会变；对高斯收益率，这一点以概率 1 成立。

此前缺的，是在组合真正所处的设定下证明这件事：收益率是随机的，而不是固定设计；约束是硬约束，而不是惩罚项。[Bellec (2023)](https://arxiv.org/abs/2008.11840) 对取有限值的凸惩罚证明了 (1 − df/n)<sup>−2</sup> 这个估计，约束恰好被排除在外。我们把约束看成越来越陡的惩罚的极限，得到两个结果：

- **有限样本的误差界。** p < n 时，校正后估计的相对误差，期望不超过 C/√m，其中 C 只依赖 p/n：不依赖协方差矩阵，不依赖约束，也不管约束在总体上是否顶住。
- **一个精确恒等式。** 它解释了校正为什么有效：R<sub>in</sub> 的偏差是一阶的，相对大小约 2d/m；校正后的偏差是二阶的。

# 4 准不准？

图 2 是 20 个模拟市场：独立股票、AR(1) 相关，以及载荷全正或有正有负的单因子、三因子市场。校正前，样本内与真实方差之比几乎正好落在 (1 − d/m)<sup>2</sup> 这条曲线上，不论是哪个市场：偏差有多大，只取决于 d/m。校正后，所有点都贴在 1 上，包括 d/m = 0.92、样本内方差低估了 150 倍的那个点，这正是校正想要的效果。

![20 个模拟市场中校正的准确度](img/figure-002-zh.svg)

***图 2** 空心圆：样本内 ÷ 真实方差。实心：校正后 ÷ 真实方差。所有市场，校正前后。*

用 p 代替 d 并不是无伤大雅的近似：

| 市场 | p/n = 0.2 | p/n = 0.5 | p/n = 0.8 |
| --- | --- | --- | --- |
| 独立股票 | 1.01 / 0.99 | 1.35 / 1.02 | 4.26 / 1.02 |
| AR(1) | 1.10 / 1.00 | 1.85 / 1.00 | 7.88 / 1.02 |
| 单因子，载荷有正有负 | 1.13 / 1.00 | 1.87 / 1.01 | 7.04 / 0.99 |
| 单因子，载荷全正 | 1.36 / 1.00 | 3.13 / 1.00 | 18.60 / 1.00 |

*估计值 ÷ 真实方差的均值：经典校正（数 p）/ 本文（数 d）。n = 250，每个格子 400 次模拟；两个比值都越接近 1 越好。*

两者只在几乎所有股票都被持有时才一致，比如独立股票、p/n = 0.2 那一列。最糟的恰恰是最像真实股市的情形：所有股票对市场因子的载荷都为正，优化器只持有 17% 到 35% 的股票。在这个市场里，经典校正离真相比完全不校正还远，高估的幅度也是最大的。

1/√m 的速率也看得出来。在四个市场里，m 从 50 到 400，√m 乘以平均相对误差一直稳定在 0.57 到 1.27 之间，不随 m 漂移。

这些都是平均值。单次样本里，校正值仍有波动：它在不同样本间的标准差，从 d/m 小时的 0.10 增长到 d/m = 0.92 时的 0.49。定理给的是期望误差的界，不是预测区间。

# 5 股票比天数多

n 天的数据去掉均值后，只能张成 m = n − 1 个方向。股票数超过这个数时，有些组合在样本里完全不动，不加约束的优化器就会报告"风险为零"。约束可能排除这些组合，也可能排除不了。排除不了时，样本内方差恰好是零，真实方差却不是，任何校正都救不回来。

会不会发生，由一个数决定：协方差白化之后约束锥的统计维数 δ，它衡量约束集在随机噪声"眼里"有多大（[Amelunxen et al., 2014](https://arxiv.org/abs/1303.6672)）。δ 高于 m 时，几乎一定出现零风险；δ 低于 m 时，几乎不会出现；切换发生在宽度约 √δ 的窗口里。

![样本内风险为零的概率与 m/δ 的关系](img/figure-003-zh.svg)

***图 3** 五个市场的 δ 从 17 到 199，相差十几倍，但五条曲线都在 m/δ = 0.96 到 1.00 之间穿过 1/2。*

三个市场让它更具体：

- **独立股票、禁止卖空。** δ = p/2，组合在 p = 2m 时崩溃。这正是 [Kondor, Papp and Caccioli (2017)](https://arxiv.org/abs/1612.07067) 用 replica 方法找到的阈值；他们的方程其实也隐含了这种情形下的校正公式，只是没有写出来。直到阈值之前，禁止卖空组合的真实方差都不超过理想值的 π 倍，而不加约束的组合在 p = m 时就发散了，禁止卖空则要到 p = 2m 才崩。
- **AR(1) 相关。** δ ≈ 0.30p，阈值在 p ≈ 3.3m。
- **单因子、载荷全正。** δ 只按 log p 增长。不卖空就对冲不掉市场因子，所以禁止卖空的组合几乎永远不会崩，p 再大也一样。

载荷有正有负时，市场因子可以在禁止卖空的范围内被对冲，δ 又重新和 p 同阶。纳斯达克数据恰好显示了这一点：用季度窗口，500、1,000、1,455 只股票时，分别有 3%、19%、28% 的窗口出现了零风险组合，而且全部是 2020 年 3 月初之前结束的窗口，也就是那段平静期，当时很多股票的市场载荷接近零甚至为负，约束几乎不起作用。

阈值以下，校正依然有效。借助凸高斯极小极大定理（参见 [Thrampoulidis, Abbasi and Hassibi, 2018](https://arxiv.org/abs/1601.06233)），可以算出两种方差的极限，而且不需要假设不加约束的最优组合满足约束。在这个极限里，校正是精确的；它以 (log m / m)<sup>1/4</sup> 的速率传递到实际观测的 d 上。

# 6 跟踪误差预算

主动管理的基金经理在跟踪误差上限的约束下，最大化 alpha 信号。业内都知道，事后实际跟踪误差会高于事前估计：用商业风险模型回测，高出 1.3 到 1.4 倍（[Saxena and Stubbs, 2012](https://scinapse.io/papers/2068160108)），所以经理们会给事前估计留一个缓冲，问题是留多少（[De Nard, Ledoit and Wolf, 2025](https://econ.uzh.ch/dam/jcr:8f7e20cf-e9a7-4fde-87cb-aef82de26404/jpm_2025.pdf)）。

同一个校正告诉你缓冲该留多大：选样本内上限，使 TE<sub>in</sub> / (1 − d/m) 等于目标，其中 d 是跟踪误差组合所在面的维数。跟踪误差是标准差，所以这里不平方，只有方差才平方。

| 天数 n | 股票 p | 目标 | d/m | 上限 = 目标 | 校准后 |
| --- | --- | --- | --- | --- | --- |
| 252 | 50 | 2% | 0.19 | 1.38 | 1.12 |
| 252 | 100 | 2% | 0.37 | 1.68 | 1.11 |
| 252 | 200 | 2% | 0.67 | 2.08 | 1.09 |
| 252 | 200 | 6% | 0.37 | 1.35 | 1.00 |
| 126 | 100 | 2% | 0.72 | 2.32 | 1.06 |

*纳斯达克，股票比天数少：月度窗口上"实际跟踪误差 ÷ 目标"的中位数。基准为等权组合，信号为动量；d/m 取校准后的值。*

固定的缓冲做不到这一点。不同设定下，直接设上限会超标 1.27 到 2.32 倍，因为 d/m 在变；校准后则稳定在 1.00 到 1.12 倍，几乎贴着目标。

股票比天数多时，会出现一个新东西：地板。把样本内预算一路收紧到零，组合会收敛到一个在样本内完全复制基准的组合；在禁止卖空约束下，它恰好持有 n 只股票，220 个模拟样本无一例外。它的真实跟踪误差是正的，所以预算收得再紧，实际跟踪误差也压不到地板以下。样本内的数字还看不见这个地板：在地板处，样本内跟踪误差是 0，d = m，TE<sub>in</sub> / (1 − d/m) 是 0/0。实用的警报是：校准后的 d/m > 0.85。在纳斯达克数据上，它识别出了 90% 到 98% 目标低于地板的窗口。

![纳斯达克数据上，实际跟踪误差 ÷ 目标与目标 ÷ 地板的关系](img/figure-004-zh.svg)

***图 4** 校准后的组合落在 max(目标, 地板) 附近；直接把样本内上限设成目标，处处超标。*

地板在多数窗口低于目标时，校准后的实际跟踪误差是目标的 0.94 到 1.05 倍，直接设上限则是 1.38 到 1.78 倍。目标低于地板时，校准后是地板的 1.02 到 1.07 倍，基本已经是能做到的最好。地板也不小：季度窗口的中位数是每年 5.4% 和 7.3%，一年窗口、500 只股票时是 3.0%，波动最低的 500 只股票里是 1.7%，都落在常见的跟踪误差约定范围内。

# 7 它解决不了什么

- **它不会提高夏普比率。** 权重没变，变的只是你对风险的认识。用诚实的风险数字把组合缩放到目标波动率，也不会改变夏普比率，只是让实际波动落在你瞄准的地方。
- **全市场共同的冲击。** 如果所有股票每天共享一个随机的波动尺度（椭圆分布、波动率聚集），这个公式在我们的模拟里会偏低 7.5% 到 38%。重新拟合的留一法（也就是 jackknife）在同样的情形下稳定在 0.95 到 1.07，所以建议两个都报告。
- **漂移。** 纳斯达克上，以等权组合为参照，校正后的最小方差预测仍然偏低 1.1 到 1.5 倍。完全没有优化偏差的 split-sample 预测同样偏低 1.02 到 1.41 倍：协方差在估计窗口和持有期之间变了，单个窗口里算出来的任何东西都看不见这一点。
- **因子风险模型。** 大多数机构并不直接用样本协方差去优化。理论覆盖的是样本协方差、固定强度的岭收缩，以及作为提议的指数加权。如果组合是用因子模型或节点回归的估计构造的，什么量扮演 d 的角色，目前还是开放问题。
- **数据本身。** 纳斯达克面板（1,455 只股票，2018 年 1 月到 2023 年 5 月，来自 [skfolio-datasets](https://github.com/skfolio/skfolio-datasets)）只包含存续股票，数据方声明仅供测试。实证部分是在演示方法，不是在评价这些股票。理论假设收益率服从独立的高斯分布。

# 8 怎么用

1. 精确求解组合问题，让 d 是一个准确的计数。在 d/m = 0.9 附近，数错一个持仓，校正值就会差好几个百分点，d/m 越大越敏感。
2. 报告 R<sub>in</sub> / (1 − d/m)<sup>2</sup>，同时报告 d/m。d/m 接近 1 时，样本内的数字几乎不包含信息。
3. 有跟踪误差约定时，选样本内上限使 TE<sub>in</sub> / (1 − d/m) 等于目标；校准后 d/m > 0.85，就说明目标可能已经低于地板。
4. 算得起的话，再报告重新拟合的留一法估计和 split-sample 预测，用来检查厚尾和漂移。

# 参考文献

1\. Kondor, I., Papp, G., and Caccioli, F. (2017). Analytic solution to variance optimization with no short positions. Journal of Statistical Mechanics. arXiv:1612.07067. [https://arxiv.org/abs/1612.07067](https://arxiv.org/abs/1612.07067)

2\. El Karoui, N. (2010). High-dimensionality effects in the Markowitz problem and other quadratic programs with linear constraints: Risk underestimation. The Annals of Statistics. arXiv:1211.2917. [https://arxiv.org/abs/1211.2917](https://arxiv.org/abs/1211.2917)

3\. Shepard, P. G. (2009). Second Order Risk. arXiv:0908.2455. [https://arxiv.org/abs/0908.2455](https://arxiv.org/abs/0908.2455)

4\. Jagannathan, R., and Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. The Journal of Finance. NBER Working Paper 8922. [https://www.nber.org/papers/w8922](https://www.nber.org/papers/w8922)

5\. Basak, G. K., Jagannathan, R., and Ma, T. (2009). Jackknife Estimator for Tracking Error Variance of Optimal Portfolios. Management Science. [https://doi.org/10.1287/mnsc.1090.1001](https://doi.org/10.1287/mnsc.1090.1001)

6\. Tibshirani, R. J., and Taylor, J. (2012). Degrees of freedom in lasso problems. The Annals of Statistics. [https://doi.org/10.1214/12-AOS1003](https://doi.org/10.1214/12-AOS1003)

7\. Bellec, P. C. (2023). Out-of-sample error estimation for M-estimators with convex penalty. Information and Inference. arXiv:2008.11840. [https://arxiv.org/abs/2008.11840](https://arxiv.org/abs/2008.11840)

8\. Amelunxen, D., Lotz, M., McCoy, M. B., and Tropp, J. A. (2014). Living on the edge: Phase transitions in convex programs with random data. Information and Inference. arXiv:1303.6672. [https://arxiv.org/abs/1303.6672](https://arxiv.org/abs/1303.6672)

9\. Thrampoulidis, C., Abbasi, E., and Hassibi, B. (2018). Precise error analysis of regularized M-estimators in high dimensions. IEEE Transactions on Information Theory. arXiv:1601.06233. [https://arxiv.org/abs/1601.06233](https://arxiv.org/abs/1601.06233)

10\. Saxena, A., and Stubbs, R. A. (2012). An Empirical Case Study of Factor Alignment Problems Using the USER Model. The Journal of Investing. [https://scinapse.io/papers/2068160108](https://scinapse.io/papers/2068160108)

11\. De Nard, G., Ledoit, O., and Wolf, M. (2025). Improved Tracking-Error Management for Active and Passive Investing. The Journal of Portfolio Management. [https://econ.uzh.ch/dam/jcr:8f7e20cf-e9a7-4fde-87cb-aef82de26404/jpm_2025.pdf](https://econ.uzh.ch/dam/jcr:8f7e20cf-e9a7-4fde-87cb-aef82de26404/jpm_2025.pdf)

12\. skfolio-datasets：skfolio 库附带的数据集（纳斯达克日度价格）。[https://github.com/skfolio/skfolio-datasets](https://github.com/skfolio/skfolio-datasets)`
};
