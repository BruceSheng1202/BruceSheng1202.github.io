window.POST_CONTENT = {
  en: String.raw`# 1 Overview

This is placeholder body text. Write the English version here, the Chinese version in zh.md, then run the build script to regenerate content.js.

Headings start with a hash and a number, the way the other posts do it: <code># 1 Overview</code> for a section, <code>## 1.1 A subsection</code> for a child.

Images work the usual way. Put the file next to the page under img/ and reference it as img/figure-001.png; the paragraph right below an image, written in italics, becomes its caption.

| Model | Loss | Notes |
| --- | --- | --- |
| Baseline | 3.53 | four loops |
| Ours | 3.51 | four loops, last 64 tokens |

# 2 What to keep

- Numbers in the text, links to sources in the reference list.
- One sentence per idea; the reader is skimming.

# References

1. Author. Title of the paper. arXiv:2601.00001. https://arxiv.org/abs/2601.00001`,
  zh: String.raw`# 1 概览

这是正文的占位内容。英文写在 en.md，中文写在 zh.md，然后运行构建脚本重新生成 content.js。

标题用井号加编号，和另外几篇保持一致：<code># 1 概览</code> 是一级章节，<code>## 1.1 小节</code> 是子章节。

图片照常使用：文件放在同级的 img/ 目录下，正文里写 img/figure-001.png；图片下面紧跟一段斜体文字，会自动变成图注。

| 模型 | Loss | 说明 |
| --- | --- | --- |
| 基线 | 3.53 | 循环 4 轮 |
| 本文 | 3.51 | 循环 4 轮，后 3 轮看 64 token |

# 2 要保留的东西

- 正文里给数字，来源放参考文献。
- 一句话讲一件事，读者是在扫读。

# 参考文献

1. 作者. 论文标题. arXiv:2601.00001. https://arxiv.org/abs/2601.00001`
};
