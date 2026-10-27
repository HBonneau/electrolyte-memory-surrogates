# Upload to GitHub

Create an empty repository named `electrolyte-memory-surrogates` in your GitHub account. Choose its visibility. The prepared folder already includes its README and MIT license.

From this project folder:

```bash
git init -b main
git add .
git status --short
git commit -m "Add electrolyte relaxation surrogate study"
```

Copy the repository's clone URL from GitHub. Replace `YOUR_REPOSITORY_URL` below with that URL:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Use your own Git author configuration. `data/`, `artifacts/`, and the recorded `results/` are intentionally included. Environments, caches, new training runs, and reproduced results are ignored. After upload, check the README figure and documentation links.

The supplied ZIP is a source-project archive; unpack it before using these commands. Repository creation and upload have not been performed by this packaging step.
