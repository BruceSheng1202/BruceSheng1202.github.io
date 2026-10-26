# 新文章模板

复制这个目录，改成你的 slug，填内容就能发。`_template` 以 `_` 开头，GitHub Pages 不会发布它。

## 五步

1. 复制并改名（slug 用短横线，例如 `sparse-reward-judge`）
   ```
   cp -r blog/_template blog/<slug>
   rm blog/<slug>/README.md
   ```
2. 改 `index.html` 里的头部信息：`<title>`、`description`、eyebrow（日期）、`h1`、`dek`、`meta` 的日期与阅读时长，以及 `zh` 那一份 `data-zh`。
3. 改 `POST_CONFIG`：`term` 必须等于 slug（giscus 用它找 Discussion），`bibtex` 里的标题、年份、url 一并改掉。
4. 写 `en.md` 和 `zh.md`，然后生成内容：
   ```
   node blog/tools/build-content.mjs blog/<slug>
   ```
   脚本会产出 `content.js`，**不要手改它**。`en.md` / `zh.md` 里不能出现反引号和 `${`，行内代码用 `<code>…</code>`。
5. 三处登记，缺一个就会漏流量：
   - `blog/index.html` 加一条 `<article class="blog-entry">`，日期写 `<time datetime="YYYY-MM-DD">Month D, YYYY</time>`
   - 首页 `index.html` 的 News 加一条：`[Mon YYYY] A new blog, <a href="...">标题</a>.`，只写标题
   - `README.md` 的 Latest Blog 加一条

## 写作约定

- 章节标题带编号：`# 1 Overview`、`## 1.1 A subsection`，目录和锚点自动生成。
- 图片放 `<slug>/img/`，正文用 `img/figure-001.png`；图片下面紧跟一段斜体文字，会变成图注。
- 参考文献放在 `# References` / `# 参考文献` 标题下，每条一个链接；正文引用同一 URL 时，右侧会自动出现对应注释。
- 数据、基准都要给链接，数字要能被读者复核。

## 本地预览

```
python3 -m http.server 8000
# 打开 http://localhost:8000/blog/<slug>/
```

`?lang=zh` 直接看中文；切换语言会记住选择。

## 说明

`opd-loss-design` 和 `jevany-robotics` 用的是这套流程（各自内联了一份渲染脚本）；`loop-compute-ledger` 更早，是静态 HTML + i18n 字典，页面本身仍然有效，但**新文章不要照它写**。
