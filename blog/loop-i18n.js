(function () {
  var EN = {
    't-outline': 'On this page',
    't-eyebrow': 'Research note · October 2026',
    't-title': 'Accounting for Compute in a Looped Model',
    't-dek': 'A looped language model reuses the same block of layers. The parameter count stays fixed while the computation can be made deeper, one pass at a time. Training is expensive: the Parcae recipe runs about eight loops per sample on average. We trained more than sixty small models on a 5090 24G to see how many of those loops are actually doing work, and what the saved compute can buy.',
    't-meta': '86.5M and 144M parameters · at most 2B training tokens · FineWeb-Edu',
    't-sum': 'Takeaway',
    't-sum-1': 'A model trained with the original Parcae recipe collects 94% of its gain by loop 4. After that the hidden state almost stops moving, and the later loops mostly idle.',
    't-sum-2': 'We changed the recipe in two places. Every sample runs exactly four loops, and every loop receives a gradient. From loop 2 on, the loop block attends only to the most recent 64 tokens. On the same training data, this lean recipe matches the original, and it is slightly better on the smaller model. Training is 1.25 to 1.4 times faster, generation is 2 to 3 times faster, and the KV cache is about one third as large.',
    't-sum-3': 'The advantage has edges. It shrinks as training gets longer, and at 2B tokens the original recipe pulls slightly ahead by looping more. At this small scale, a plain Transformer with the same parameter count and the same training time beats every looped variant in this comparison.',
    't-howto': '<b>How to read the numbers.</b> Loss is validation cross-entropy, and lower is better. At this scale, a drop of 0.01 is roughly equivalent to training on 2% to 5% more data. Rerunning a recipe with a new seed moves the loss by 0.001 to 0.01, so we treat any gap inside 0.01 as a tie.',
    'h-bg': 'The model and the recipe',
    'h-plateau': 'After four loops, the original model mostly stops',
    'h-recipe': 'Two changes in the lean recipe',
    'h-why': 'Why four fixed loops match a mean of eight',
    'h-window': 'The loop block does not need to look far',
    'h-ledger': 'The ledger',
    'h-limits': 'Where this stops',
    'h-gap': 'The longer the run, the smaller the gap',
    'h-plain': 'At this scale a plain Transformer is the better buy',
    'h-scale': 'The scale is small',
    'h-prior': 'Prior work',
    'h-details': 'Setup',
    'h-comments': 'Discussion',
    'h-cite': 'Cite this article',
    'h-ref': 'References',
    'cite-help': 'Copy this BibTeX entry for papers or research notes.',
    'cite-btn': 'Copy BibTeX',
    'p-bg-1': 'A plain Transformer uses each layer once. A looped model turns the middle layers into a block and sends the same sequence through that block again. Each extra loop adds depth without adding parameters. Our layout matches <a href="https://arxiv.org/abs/2604.12946" data-citation="1">Parcae</a>: two prelude layers and two coda layers run once, and a two-layer loop block repeats for T loops. Every loop reads the prelude output again, as drawn in Figure 1 for the whole model.',
    'p-bg-2': 'Each extra loop costs another forward pass, and during training it can cost a backward pass too. Parcae is the strongest public training recipe we compared against, so the rest of this note calls it the original recipe. It draws a loop count for each sample, with mean 8. To save memory and compute, only the last four loops receive gradients; the earlier loops are forward-only. At inference the usual choice is 8 to 12 loops.',
    'p-bg-3': 'The question is an accounting one. Is a mean of eight loops necessary, is it worth training only the last four, and does every loop need the full context?',
    'p-plateau': 'Start with what an original-recipe model does at inference. Take the 144M model trained on 1B tokens and evaluate it at 1 through 32 loops (Figure 2). From 1 loop to 4, loss falls from 3.50 to 3.26. From 4 to 8 it falls another 0.016. Past 8 loops the curve is flat. The first four loops collect 94% of the gain between 1 and 8. All seven original-recipe models we trained at this size show the same plateau after the fourth loop.',
    'p-hidden': 'The hidden state tells the same story. Each new loop changes it by about half to two thirds as much as the previous loop, and by loop 12 the change is on the order of bf16 rounding. The model settles at a fixed state. Further loops do not move it. That state is a fixed point. Of the eight loops the original recipe runs on average, the later ones barely affect the output, and they cost as much as the early ones.',
    'p-recipe-1': 'That measurement suggests two changes, which together we call the lean recipe (Figure 1, third row). First, train with exactly four loops, and backpropagate through all four. There is no random loop count and no forward-only prefix. Second, the loop block sees the full context on loop 1 only. Loops 2 through 4 see the most recent 64 tokens. The four layers outside the loop are left unchanged from the original layout.',
    'p-recipe-2': 'On the same training data, four loops of the lean recipe match the original recipe at its best loop count, which is 12. For the 144M model trained on 1B tokens, the lean recipe is 0.005 lower on both seeds, inside the noise. For the 86.5M model trained on 0.5B tokens, it is better on all three seeds, by 0.025 on average. Training is also about 1.4 times faster. The next two sections say why neither change gives up accuracy. The speed accounting comes after that.',
    'p-why-1': 'We measured how much each loop moves the hidden state (Figure 3). On the original recipe, the change halves at loops 2, 3, and 4, and the state settles. On the lean recipe, those three loops keep making large edits. The state does not converge.',
    'p-why-2': 'The difference comes from which loops receive a gradient. The original loop count is random with mean 8, but only the last four loops are trained. Only about one sample in ten draws 4 loops or fewer, and only those samples train the loops that start from the initial state. Every other sample trains loops that are already near the fixed point. The model learns to converge quickly and then stop. With four fixed loops, every step from the initial state to the output has a gradient, so the model learns to use each of them.',
    'p-why-3': 'Two controls support this, each run once. At a matched training time, four loops with a gradient on every loop beat six loops whose first three are forward-only, by 0.023 loss. The extra loops are worth less than training every loop. In the other control, the original random schedule is kept, but only the last loop receives a gradient. The gain from looping, measured as the loss drop from 1 loop to 8, shrinks from 0.23 to 0.11. The model uses the loop about half as much, and the drop is large enough to see in the numbers.',
    'p-why-4': 'Fixing the count has a cost: the model can only run four loops (Figure 4). Fewer or more loops are clearly worse. Eight loops are 0.07 worse than four, and one loop is 0.8 worse. An original-recipe model can run anywhere from 6 to 32 loops with almost the same loss. The lean recipe gives up the usual selling point of a looped model, which is spending more inference loops for a better answer. Figure 2 says the original recipe was already limited on that point: past eight loops there is nothing left to buy. <a href="https://arxiv.org/abs/2610.06833" data-citation="2">Huang et al. (2026)</a> report both of these facts for models ranging from 100M to 1.6B parameters.',
    'p-win-1': 'The second change comes from an ablation. On the 86.5M model, we removed one piece of compute at a time from the loop block of the fixed-4 model, and Figure 5 shows which of those pieces can be removed without a worse loss.',
    'p-win-2': 'Only the shorter attention window is free. Loops 2 through 4 can see the last 64 tokens or the last 256 without a worse loss, and even restricting loop 1 to 64 tokens does not hurt. Every other cut is clearly worse: half width by 0.029, MLP only by 0.047, reusing loop 1\'s KV by 0.031, and stopping at two loops by 0.026.',
    'p-win-3': 'So each loop still needs its full width and its own attention. It does not need distant tokens. The prelude and the coda, four layers of full attention, carry the long-range information. The 64-token window reproduced on the other two seeds and on the 144M model. Keeping full attention on loop 1, as the lean recipe does, is the conservative choice.',
    'p-win-4': 'There is a cost. Tokens that copy content from more than 500 positions earlier have about 0.02 higher loss, and they are about 3% of tokens. The other tokens have slightly lower loss. Which of the two recipes wins on downstream tasks still needs a larger-scale experiment.',
    'p-led-1': 'Together, the two changes raise training throughput on the 144M model from 35k to 49k tokens per second, a factor of 1.42. Part of that is an inefficiency in the current original-recipe implementation. It draws a loop count per sample, so every sample in a batch waits for the longest one. Padding burns 15% to 20% of the compute. Drawing one count for the whole batch would make the original recipe about 15% faster, and the lean advantage would fall to 1.25 times.',
    'p-led-2': 'Faster training can be spent on a lower loss. We gave the original recipe less data so that wall-clock time matched, and both runs finished the learning-rate decay (Figure 6). At 5.7 hours the lean loss is 0.11 lower. At 11.3 hours it is 0.06 lower. On the 86.5M model, 1.8 hours gives a gap of 0.18. All six paired repeats agree on the direction. This comparison uses the original implementation as it stands. Fixing the padding inefficiency would close about a third to a half of the gap between these two training recipes.',
    'p-led-3': 'Generation is where the gap is larger. A looped model stores a separate KV cache for each loop, so eight loops means eight copies. The lean recipe stores four loops, and three of them store only the last 64 tokens. Timed on the 144M layout (Figure 7), the lean recipe generates a token 2.3 to 3.1 times faster than the original recipe at 8 loops, with about one third of the cache. At a context of 16k tokens it is about as fast as a plain 6-layer Transformer of the same size, and the caches match, because both have only six layers that see the full context. At 2,000 tokens the plain Transformer is still about 20% faster.',
    'p-gap-1': 'At matched wall-clock time, the gap falls from 0.18 on the smaller model, to 0.11 at 5.7 hours on the larger one, to 0.06 at 11.3 hours. Matching the amount of data instead of the time gives the same trend. At 144M parameters and 2B tokens, the original recipe at 12 loops is 0.005 to 0.009 better than the lean recipe, on both seeds. It spends three times the compute to get there. If it is also limited to 4 loops, it is still 0.013 to 0.014 worse.',
    'p-gap-2': 'More training seems to make extra loops more valuable, which matches the scaling reported in the Parcae paper. Whether fixing the count above four loops keeps the advantage is still open: on the smaller model, eight fixed loops beat the original recipe by 0.034, and a longer run still needs a fuller experimental analysis.',
    'p-plain': 'On the 86.5M model we also trained a 6-layer Transformer with the same parameter count and no loop. At about two hours it has a lower loss than every looped variant, 0.03 to 0.04 below the lean recipe (the green line in the upper panel of Figure 6). At this size the loop itself is not yet worth it. The lean recipe closes part of the gap and does not reverse it. The same comparison on a larger model still needs experimental analysis.',
    'p-scale': 'Everything here is at most 144M parameters and 2B tokens, on one dataset and one 5090 24G. Most of the parameters sit in the embedding table. The compute-heavy part is 19.4M parameters in the smaller model and 43.7M in the larger one. What a larger model or a longer training run would do still needs further experimental analysis.',
    'p-prior': 'Most of the phenomena here have been seen in other settings.',
    'li-1': 'Drawing a random loop count and training only the last few loops comes from <a href="https://arxiv.org/abs/2502.05171" data-citation="3">Huginn</a> (Geiping et al., 2025). Parcae keeps that choice. Both Parcae and <a href="https://arxiv.org/abs/2610.06833" data-citation="2">Huang et al. (2026)</a> report that loss stops falling once inference uses more loops than the training average, and the two papers agree on that point.',
    'li-2': 'Fixed-count training is common. <a href="https://arxiv.org/abs/2510.25741" data-citation="4">Ouro</a> and <a href="https://arxiv.org/abs/2605.20670" data-citation="5">LT2</a> both fix the count at 4. Huang et al. compare the two schedules from 100M to 1.6B parameters: at a matched mean, a fixed count beats a random one, but the model gets worse if inference uses a different count. The appendix of <a href="https://arxiv.org/abs/2609.19107" data-citation="6">Loop growth</a> also compares random and fixed counts.',
    'li-3': '<a href="https://arxiv.org/abs/2604.21106" data-citation="7">Iso-depth</a> compares, at matched compute, training only the last few loops against training every loop. Training only the last loops wins there, because the saved compute buys more data. We compare a different split of the same compute: more loops with gradients on only some of them, against fewer loops with a gradient on every one.',
    'li-4': 'Local attention on later loops is used by <a href="https://arxiv.org/abs/2605.20670" data-citation="5">LT2</a>, <a href="https://arxiv.org/abs/2510.24824" data-citation="8">PLT</a>, and <a href="https://arxiv.org/abs/2610.02383" data-citation="9">LPT</a>. PLT and LPT let later loops see 64 tokens while also reading the global KV from the first loop. Sharing the first loop\'s KV directly is the <a href="https://arxiv.org/abs/2507.10524" data-citation="10">MoR</a> choice. PLT reports that sharing KV alone loses accuracy, and our own runs show the same loss.',
    'p-add': 'What we add is three measurements in one setup: a joint account of training time, generation speed, and loss, including implementation overhead; a loop-by-loop look at the hidden state that explains why a fixed count does not lose; and an ablation of which pieces of compute inside the loop block can be removed without hurting the loss.',
    'dt-data': 'Data',
    'dd-data': 'FineWeb-Edu, read in order. The validation set and the test set are held-out shards that never enter training.',
    'dt-layout': 'Layout',
    'dd-layout': 'Two prelude layers, a two-layer loop block, and two coda layers. Each loop reads the prelude output, which is the Parcae injection. The vocabulary contains 32,768 tokens, shared by every layer.',
    'dt-size': 'Two sizes',
    'dd-size': '86.5M: width 512, 4 heads, MLP 2048, and 19.4M parameters outside the embedding. 144M: width 768, 6 heads, MLP 3072, and 43.7M parameters outside the embedding on the larger model.',
    'dt-train': 'Training',
    'dd-train': 'Sequence length 2048, about 520k tokens per step. The optimizer is Muon, with learning rate 0.008, constant for the first half and then linear to zero. That is Parcae\'s default. Every recipe uses this setting; nothing was tuned per recipe.',
    'dt-orig': 'Original',
    'dd-orig': 'The loop count of each sample is drawn from a Poisson distribution with mean 8. Backpropagation covers at most the last four loops.',
    'dt-local': 'Local attention',
    'dd-local': 'A causal sliding window of 64 tokens, implemented with PyTorch FlexAttention.',
    'dt-rep': 'Repeats',
    'dd-rep': 'The main comparison uses 3 seeds at 86.5M and 2 seeds at 144M. Within a seed, every recipe shares initialization and data order. Each ablation was trained as one single training run, with no extra seed.',
    'dt-eval': 'Evaluation',
    'dd-eval': 'Every model is scored at 1, 2, 3, 4, 6, 8, 12, 16, 24, and 32 loops. Every number is a finished run through the learning-rate decay, not a midpoint checkpoint.',
    'dt-hw': 'Hardware',
    'dd-hw': 'One 5090 24G, bf16, torch.compile.',
    'foot-src': 'The numbers come from the evaluation record written at the end of each run. The raw timing and per-loop diagnostics are in the project log.',
    'foot-back': '← All notes',
    'show-data': 'Show the data',
    'show-compare': 'Show the matched-data comparison',
    'ft-2': 'Original recipe: validation loss against the number of inference loops',
    'ft-3': 'How much each loop moves the hidden state',
    'ft-4': 'Validation loss against inference loops, both recipes',
    'ft-5': 'Loss change after removing one piece of the loop block',
    'ft-6': 'Who has the lower loss at the same training time',
    'ft-7a': 'Time to generate one token (ms)',
    'ft-7b': 'KV cache (MB)',
    'cap-1': '<b>Figure 1</b> The model, and the three training recipes compared here. The original-recipe row is drawn for a sample drawn for one sample that happened to draw 9 loops.',
    'cap-2': '<b>Figure 2</b> Original recipe, 144M parameters, 1B training tokens, mean of two seeds. The horizontal axis is the inference loop count on a log scale.',
    'cap-3': '<b>Figure 3</b> Relative change is the size of the difference across a loop, divided by the size of the state entering that loop. Both main models have 144M parameters, were trained on 2B tokens, and are averaged over 32 validation segments. The fixed-4 full-attention model, trained on 1B tokens, reads 0.70, 0.54, 0.46, the same pattern.',
    'cap-4': '<b>Figure 4</b> 144M parameters, 1B tokens, mean of two seeds. The lean recipe is best at 4 loops, 0.005 below the original recipe\'s best score. Every other loop count is clearly worse.',
    'cap-5': '<b>Figure 5</b> The baseline is four fixed loops with full attention. Negative means better. 86.5M parameters, 0.5B tokens, one run each. The gray band is the seed range.',
    'cap-6': '<b>Figure 6</b> Each point is a finished training run. The original-recipe points use different amounts of data. Where several seeds exist, the point is their mean. The upper panel also includes a 6-layer Transformer with the same parameter count and no recurrence at all.',
    'cap-7': '<b>Figure 7</b> Measured on the 144M layout, 16k tokens of context, one sequence at a time. The fixed-4 row changes only the loop count, so the two savings can be read separately.',
    'leg-old': 'Original',
    'leg-lean': 'Lean',
    'leg-lean4': 'Lean (fixed 4)',
    'leg-gpt': 'Plain Transformer (6 layers, no loop)',
    'leg-short': 'Shorter attention only',
    'leg-other': 'Every other cut',
    'th-loops': 'Inference loops',
    'th-s43': 'Seed 43',
    'th-s44': 'Seed 44',
    'th-s44b': 'Seed 44',
    'th-mean': 'Mean',
    'th-model': 'Model',
    'th-data': 'Data',
    'th-seed': 'Seed',
    'th-orig-best': 'Original (best loop count)',
    'th-fix': 'Fixed 4',
    'th-lean': 'Lean',
    'th-l2': 'Loop 2',
    'th-l3': 'Loop 3',
    'th-l4': 'Loop 4',
    'th-o43': 'Original · seed 43',
    'th-l43': 'Lean · seed 43',
    'th-change': 'Change to the loop block',
    'th-loss': 'Validation loss',
    'th-delta': 'Versus baseline',
    'th-time': 'Train time',
    'th-orig-data': 'Original (data)',
    'th-lean-data': 'Lean (data)',
    'th-gap': 'Gap',
    'th-layout': 'Layout',
    'th-kv': 'KV cache',
    'th-2k1': '2k tokens<br>batch 1',
    'th-2k64': '2k tokens<br>batch 64',
    'th-16k1': '16k tokens<br>batch 1',
    'th-16k8': '16k tokens<br>batch 8',
    'm-86': '86.5M',
    'm-144': '144M',
    'd-05': '0.5B tokens',
    'd-1': '1B tokens',
    'd-2': '2B tokens',
    'note-compare': 'Validation loss. The original recipe is scored at its best count from 1 to 32 loops, which is 12 in every case. The other two recipes are scored at 4 loops. Recipes in the same row share a seed, so they share initialization and data order. The test set agrees with the validation set on the direction.',
    'row-orig-2b': 'Original (2B tokens)',
    'row-lean-2b': 'Lean (2B tokens)',
    'row-fix-1b': 'Fixed 4, full attention (1B tokens)',
    'cut-0': 'Baseline: fixed 4, full attention',
    'cut-64': 'Loops 2–4 see the last 64 tokens',
    'cut-all64': 'All 4 loops see the last 64 tokens',
    'cut-256': 'Loops 2–4 see the last 256 tokens',
    'cut-half': 'Loops 2–4 at half width (heads and MLP)',
    'cut-mlp': 'Loops 2–4 drop attention, MLP only',
    'cut-kv': 'Loops 2–4 reuse the KV from loop 1',
    'cut-2': 'Only 2 loops',
    'cut-ref': 'Reference: original recipe',
    'cut-gpt': 'Reference: plain Transformer, 6 layers, no loop',
    'h-199': '1.99 h',
    'h-180': '1.80 h',
    'h-170': '1.70 h',
    'h-181': '1.81 h',
    'h-176': '1.76 h',
    'h-160': '1.60 h',
    'h-193': '1.93 h',
    'h-156': '1.56 h',
    'h-259': '2.59 h',
    'h-133': '1.33 h',
    't-18': 'about 1.8 h',
    't-57': 'about 5.7 h',
    't-113': 'about 11.3 h',
    'cell-a': '3.6979 (345M tokens)',
    'cell-b': '3.5319 (0.5B tokens)',
    'cell-c': '3.3521 (704M tokens)',
    'cell-d': '3.2413 (1B tokens)',
    'cell-e': '3.1654 (1.41B tokens)',
    'cell-f': '3.1024 (2B tokens)',
    'note-time': 'Other points. The original 144M recipe takes 8.07 hours for 1B tokens (loss 3.2452) and 15.98 hours for 2B tokens (loss 3.0932). The original 86.5M recipe takes 2.61 hours for 0.5B tokens (loss 3.5505). A plain Transformer at 86.5M, seed 43, takes 1.33 hours for 0.5B tokens (loss 3.5540) and 1.99 hours for 0.75B tokens (loss 3.4673).',
    'lay-8': 'Original (8 loops)',
    'lay-64': 'All 4 loops see only 64 tokens',
    'lay-gpt': 'Plain Transformer (6 layers)',
    'note-lat': 'bf16, a static KV cache, and a CUDA graph around the whole step. Local-attention loops use a 64-slot ring cache. The time depends on the layout, not on the weight values, so the weights are random. The cache sizes below are for a batch of one sequence at every length.',
    'svg-aria': 'Model layout: a 2-layer prelude, a 2-layer loop block repeated for T loops, and a 2-layer coda; plus the original, fixed-4, and lean training recipes',
    'svg-in': 'Input',
    'svg-pre': 'Prelude',
    'svg-once': 'once',
    'svg-read': 'reread',
    'svg-loop': 'Loop block',
    'svg-t': '× T',
    'svg-share': 'same weights, reused',
    'svg-coda': 'Coda',
    'svg-next': 'Next token',
    'svg-orig': 'Original',
    'svg-rand': 'random, mean 8',
    'svg-last4': 'gradient on last 4',
    'svg-fix': 'Fixed 4',
    'svg-fix1': '4 loops, gradient on each',
    'svg-fix2': 'also 4 at inference',
    'svg-lean': 'Lean',
    'svg-lean1': 'fixed 4; loops 2–4',
    'svg-win': 'last 64 tokens',
    'svg-grad': 'with gradient',
    'svg-fwd': 'forward only',
    'svg-sq': 'one square = one loop'
  };

  var bib = '@misc{sheng2026loop,\n  author = {Sheng, Huanxin},\n  title = {Accounting for Compute in a Looped Model},\n  year = {2026},\n  month = oct,\n  url = {https://brucesheng1202.github.io/blog/loop-compute-ledger.html},\n  note = {Blog post}\n}';

  var lang = new URLSearchParams(location.search).get('lang') === 'zh' ? 'zh' : 'en';
  var frame = 0;

  function applyText() {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      var key = el.getAttribute('data-i18n');
      var htmlMode = el.hasAttribute('data-html');
      var isSvgRoot = el.namespaceURI === 'http://www.w3.org/2000/svg' && el.localName === 'svg';
      if (!el.hasAttribute('data-zh')) {
        el.setAttribute('data-zh', isSvgRoot ? (el.getAttribute('aria-label') || '') : (htmlMode ? el.innerHTML : el.textContent));
      }
      var next = lang === 'en' ? EN[key] : el.getAttribute('data-zh');
      if (next == null) return;
      if (isSvgRoot) el.setAttribute('aria-label', next);
      else if (htmlMode) el.innerHTML = next;
      else el.textContent = next;
    });
    document.title = lang === 'en'
      ? 'Accounting for Compute in a Looped Model — Huanxin Sheng'
      : '循环模型的算力账 — Huanxin Sheng';
    document.querySelectorAll('.lang button').forEach(function (button) {
      var on = button.getAttribute('data-lang') === lang;
      button.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    var outline = document.querySelector('.outline');
    if (outline) outline.setAttribute('aria-label', lang === 'en' ? 'Outline' : '目录');
    var rail = document.getElementById('citation-rail');
    if (rail) rail.setAttribute('aria-label', lang === 'en' ? 'References' : '参考文献');
  }

  function layoutNotes() {
    var rail = document.getElementById('citation-rail');
    var page = document.querySelector('.page');
    var notes = [].slice.call(document.querySelectorAll('.citation-note'));
    if (!rail || !page || !notes.length) return;
    rail.style.minHeight = page.offsetHeight + 'px';
    if (getComputedStyle(rail).display === 'none') return;
    var railTop = rail.getBoundingClientRect().top + window.scrollY;
    var nextTop = 0;
    notes.forEach(function (note) {
      var source = document.getElementById(note.dataset.sourceId);
      if (!source) return;
      var desired = source.getBoundingClientRect().top + window.scrollY - railTop - 2;
      var top = Math.max(desired, nextTop);
      note.style.top = Math.round(top) + 'px';
      nextTop = top + note.offsetHeight + 14;
    });
  }

  function scheduleLayout() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(layoutNotes);
  }

  function buildRail() {
    var notes = document.getElementById('citation-notes');
    if (!notes) return;
    notes.innerHTML = '';
    var seen = {};
    [].slice.call(document.querySelectorAll('.page a[data-citation]')).forEach(function (link) {
      if (link.closest('.bibliography')) return;
      var n = link.getAttribute('data-citation');
      var ref = document.getElementById('ref-' + n);
      if (!ref) return;
      if (!seen[n]) {
        seen[n] = true;
        link.id = 'citation-source-' + n;
        var note = document.createElement('div');
        note.className = 'citation-note';
        note.dataset.sourceId = link.id;
        note.innerHTML = '<span class="citation-number" aria-hidden="true">' + n + '</span><div class="citation-copy"></div>';
        note.querySelector('.citation-copy').innerHTML = ref.innerHTML;
        note.querySelectorAll('a[href]').forEach(function (source) {
          source.textContent = lang === 'zh' ? '查看来源 ↗' : 'Open source ↗';
          source.target = '_blank';
          source.rel = 'noreferrer';
        });
        notes.appendChild(note);
      }
      var note = notes.querySelector('[data-source-id="citation-source-' + n + '"]');
      var setActive = function (on) {
        [].slice.call(document.querySelectorAll('.page a[data-citation="' + n + '"]')).forEach(function (item) {
          if (!item.closest('.bibliography')) item.classList.toggle('citation-active', on);
        });
        if (note) note.classList.toggle('active', on);
      };
      link.onmouseenter = function () { setActive(true); };
      link.onmouseleave = function () { setActive(false); };
      link.onfocus = function () { setActive(true); };
      link.onblur = function () { setActive(false); };
      if (note && !note.dataset.bound) {
        note.dataset.bound = '1';
        note.onmouseenter = function () { setActive(true); };
        note.onmouseleave = function () { setActive(false); };
      }
    });
    scheduleLayout();
  }

  function spy() {
    var links = [].slice.call(document.querySelectorAll('.outline a'));
    if (!links.length) return;
    var y = window.scrollY + 110;
    var current = links[0];
    links.forEach(function (a) {
      var el = document.getElementById((a.getAttribute('href') || '').slice(1));
      if (!el) return;
      if (el.getBoundingClientRect().top + window.scrollY <= y) current = a;
    });
    links.forEach(function (a) { a.classList.toggle('active', a === current); });
  }

  function mountComments() {
    var thread = document.getElementById('comments-thread');
    if (!thread || location.protocol === 'file:') return;
    var want = lang === 'zh' ? 'zh-CN' : 'en';
    if (thread.dataset.ready === want && thread.querySelector('iframe.giscus-frame')) return;
    thread.dataset.ready = want;
    thread.replaceChildren();
    var script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.dataset.repo = 'BruceSheng1202/BruceSheng1202.github.io';
    script.dataset.repoId = 'R_kgDOMH0GSg';
    script.dataset.category = 'Blog Comments';
    script.dataset.categoryId = 'DIC_kwDOMH0GSs4DBPJB';
    script.dataset.mapping = 'specific';
    script.dataset.term = 'loop-compute-ledger';
    script.dataset.strict = '1';
    script.dataset.reactionsEnabled = '1';
    script.dataset.emitMetadata = '0';
    script.dataset.inputPosition = 'top';
    script.dataset.theme = 'light';
    script.dataset.lang = want;
    script.crossOrigin = 'anonymous';
    script.async = true;
    thread.appendChild(script);
  }

  function render() {
    applyText();
    if (window.renderLoopCharts) window.renderLoopCharts(lang);
    buildRail();
    spy();
    document.documentElement.classList.remove('i18n-wait');
    mountComments();
  }

  document.querySelectorAll('.lang button').forEach(function (button) {
    button.addEventListener('click', function () {
      var next = button.getAttribute('data-lang') === 'zh' ? 'zh' : 'en';
      if (next === lang) return;
      lang = next;
      var url = new URL(location.href);
      url.searchParams.set('lang', lang);
      history.replaceState(null, '', url);
      render();
    });
  });

  var code = document.getElementById('cite-code');
  if (code) code.textContent = bib;
  var copy = document.getElementById('cite-copy');
  var status = document.getElementById('cite-status');
  if (copy) {
    copy.addEventListener('click', async function () {
      var copied = false;
      try {
        if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error('unavailable');
        await navigator.clipboard.writeText(bib);
        copied = true;
      } catch (err) {}
      if (!copied) {
        var area = document.createElement('textarea');
        area.value = bib;
        area.setAttribute('readonly', '');
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        try { copied = document.execCommand('copy'); } finally { area.remove(); }
      }
      if (!copied) {
        status.textContent = lang === 'zh' ? '自动复制失败，请手动选择上方条目。' : 'Automatic copy failed. Please select the entry above.';
        return;
      }
      copy.textContent = lang === 'zh' ? '已复制' : 'Copied';
      status.textContent = lang === 'zh' ? 'BibTeX 已复制到剪贴板。' : 'BibTeX copied to the clipboard.';
      window.setTimeout(function () {
        copy.textContent = lang === 'zh' ? '复制 BibTeX' : 'Copy BibTeX';
        status.textContent = '';
      }, 2400);
    });
  }

  window.addEventListener('scroll', function () { spy(); scheduleLayout(); }, { passive: true });
  window.addEventListener('resize', scheduleLayout);
  render();
})();
