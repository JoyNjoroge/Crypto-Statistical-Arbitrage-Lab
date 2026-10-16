# Contributing

```bash
make setup      # venv, dev dependencies, pre-commit hook
make check      # ruff + mypy + pytest, the same gate CI applies
```

Before opening a pull request:

- Keep the research rules in the README intact. A change that lets the formation window see holdout data, fills a signal at its own close, or invents prices for missing candles will not be merged however much it improves the numbers.
- Every new accounting path needs a test that checks the identity `net = gross + funding - borrow - execution`.
- Tests must not touch the network. Mock the HTTP session (see `tests/test_download.py`).
- If you change default parameters, say why in `configs/research_baseline.txt` and the changelog. Do not tune them on the holdout.
- Real-market results go in `docs/RESEARCH_LOG.md` with the exact command, data range and package version that produced them.
