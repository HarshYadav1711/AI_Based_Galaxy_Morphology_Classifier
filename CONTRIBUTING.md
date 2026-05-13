# Contributing

Thank you for helping improve this project. The goal is to keep the codebase **readable**, **local-first**, and **easy to run** on a laptop.

## Quick start for contributors

```bash
git clone <your-fork-url>
cd AI-Based-Galaxy-Morphology-Classifier   # or your clone directory name
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements/dev.txt
pip install -e .
pre-commit install   # optional but recommended
```

## Quality checks (must pass in CI)

```bash
ruff check src tests scripts app
black --check src tests scripts app
pytest -q
```

Auto-format before committing:

```bash
black src tests scripts app
```

## Pull requests

- Prefer **small, focused** PRs with a clear description (use the PR template).
- Avoid adding **required cloud services**, **paid APIs**, or heavy optional stacks unless discussed in an issue first.
- Update **README** or **docs/** when user-facing commands or defaults change.

## CI badge (after you publish the repo)

Replace `OWNER` and `REPO` in your README with your GitHub username and repository name so the workflow badge resolves:

```markdown
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions)
```

## Code of conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
