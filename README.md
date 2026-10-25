<p align="center"><img src="assets/banner.svg" alt="AI Engineering Lab — RAG, LLM, EEG and ML, built from scratch" width="100%"></p>

# ai-engineering-lab

[![CI](https://github.com/nidhalom/ai-engineering-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/nidhalom/ai-engineering-lab/actions/workflows/ci.yml)

Small, tested Python projects on EEG/brain-computer interfaces, RAG and LLM tooling, and classic machine learning, each written from scratch and each reporting its results honestly (including the ones that did not beat a baseline).

## Projects

All numbers below are copied from each project's own README. Most evaluations are on small or synthetic data, and each README says so in detail.

### RAG / LLM

| Project | What it is | Key tech | Result stated in its README |
|---|---|---|---|
| [bm25-from-scratch](rag-llm/bm25-from-scratch) | BM25 / BM25+ in pure Python with an accent-folding tokenizer and inverted index | Python stdlib | 16 docs, 16 queries: hit@1 0.938, MRR 0.969 (same as scikit-learn TF-IDF; "not a benchmark") |

## Run any project

Each project is self-contained. Open its folder and follow its README; the pattern is:

```bash
cd rag-llm/bm25-from-scratch          # any project folder
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[test]"              # or: pip install -r requirements.txt pytest
python -m pytest -q
```

`eeg-bci/eegnet-pytorch` needs PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) and its tests take about a minute. Projects that can call Claude (`rag-eval-kit`, `llm-json-extractor`) run fully offline unless `ANTHROPIC_API_KEY` is set.

CI (`.github/workflows/ci.yml`) runs one matrix job per project, each installing its own dependencies and running its own tests.

## Why a lab

These projects started as separate repositories. They share one idea: build the thing small, test it, and report what actually happened. Keeping them in one repository makes them easier to browse and compare, and the original commit history of each project was preserved when they were merged.

## License

MIT, see [LICENSE](LICENSE).
