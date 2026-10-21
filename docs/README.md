# MWAF-Net research companion

This folder is a buildless, offline-capable academic archive for the supplied MWAF-Net supplement. Open `index.html` directly or serve this directory with any static HTTP server. Search, filtering, figures and local mathematics do not use a CDN or network API.

## Contents and preservation

- All active source supplement prose, nine tables and twelve numbered figures are retained.
- Three items from the original main manuscript are also retained in full: pooled ZCHSound Table IV (`#main-m4`), MSWC structure Figure 2 (`#main-m2`), and the eleven-step forward-workflow Algorithm 1 (`#main-a1`). The algorithm includes its input, output and complete abbreviations note. These are additional to the nine supplement tables and twelve supplement figures.
- The prominent `#numbering-map` maps the original v2.0 main-manuscript references to the 14-page submission candidate. Original supplement Table S1 is now main Table VII, Table S6 is now Table VIII, and Fig. S7 is now Fig. 5. Their archive S-number links remain unchanged.
- Original numbering is determined by active LaTeX caption order, separately for tables and figures; archived links are `#table-s1` through `#table-s9` and `#figure-s1` through `#figure-s12`.
- Table cells are parsed from the original LaTeX, including bold and underlined markings. TSV downloads provide the displayed values. `source-manifest.json` records table values, captions and source hashes for checking.
- Fourteen original vector figure PDFs are copied byte for byte, with PNG previews rendered from those originals. Figure S6 and S8 each contain two PDF panels. The archive changes screen layout only.
- Original captions, protocols and terminology remain in the content. Cross references to the "main" manuscript retain the original supplement wording; consult the revised manuscript for any updated main-manuscript numbering. Any source wording or methodological limitation is retained rather than corrected silently.
- Complete source ZIP: `downloads/supplement-source.zip`. Source and per-item downloads are also provided individually. Reviewer feedback, decision letters and author photos are excluded.
- This archive does not claim journal acceptance, publication, endorsement or compliance with a particular journal's submission rules. It is a companion archive, and the manuscript must remain scientifically self-contained.

## GitHub Pages deployment for lieasy/MWAF-Net

The intended GitHub Pages URL would be `https://lieasy.github.io/MWAF-Net/` after the owner enables and deploys GitHub Pages. This package itself does not establish that the URL is live.

### Option A — publish the `docs` directory

1. In the repository root, create or reuse `docs/` after checking for existing project documentation.
2. Copy **the contents of this directory**, including `.nojekyll`, into `docs/`. Avoid overwriting unrelated files without reviewing the diff.
3. Commit and push the changes to the repository's publishing branch.
4. Open repository **Settings → Pages → Build and deployment**, select **Deploy from a branch**, select that branch and `/docs`, then save.
5. Check the Pages deployment outcome and visit the resulting URL. Use the fixed anchors in the manuscript only after the public deployment is verified.

### Option B — use a GitHub Actions workflow

Place the site contents in `docs/` as above. Set **Settings → Pages → Source** to **GitHub Actions**. Add the following as `.github/workflows/pages.yml` (adapt `main` if the repository uses another branch):

```yaml
name: Deploy research companion
on:
  push:
    branches: [main]
    paths: ['docs/**', '.github/workflows/pages.yml']
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v6
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v4
        with:
          path: docs
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

No JavaScript build step is required. All links to local assets are relative so GitHub's repository subpath and `file://` both work. Do not put peer-review materials or personal data in the public repository.

## Version preservation and citations

For a stable scholarly record, preserve this snapshot in a named Git tag/release and, where appropriate, a repository archive with a persistent identifier. Future edits should identify the manuscript version and avoid changing existing S-number anchors. Links support access to additional material; they do not replace information the journal requires in the article or count as an exemption from its page limits.

## Third-party code

KaTeX 0.16.22 is bundled locally under `assets/katex/` with its MIT license. No analytics, advertising, external fonts or remote JavaScript are included.
