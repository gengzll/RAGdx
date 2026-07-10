"""Sample a public QA dataset into ragdx's input formats.

Produces, under ``--out``:

* ``corpus.jsonl``    — ``{"text", "source"}`` per line (the retrieval pool;
  feed as the CLI corpus argument)
* ``questions.jsonl`` — ``{"question", "ground_truth"}`` per line (CLI
  ``--questions``)
* ``questions.csv``   — same rows as CSV (upload as the studio's GT file)

Usage::

    # China: route HF downloads through the mirror first
    #   set HF_ENDPOINT=https://hf-mirror.com   (Windows cmd)
    #   export HF_ENDPOINT=https://hf-mirror.com (bash)

    python scripts/prepare_dataset.py mini-wiki --n 30
    python scripts/prepare_dataset.py squad     --n 30
    python scripts/prepare_dataset.py hotpotqa  --n 20

    # then:
    ragdx experiment datasets/squad/corpus.jsonl --has-gt \
        --questions datasets/squad/questions.jsonl --mode with_gt

Dataset notes
-------------
mini-wiki  rag-datasets/rag-mini-wikipedia — ~3.2k passages + ~918 QA,
           download only a few MB. Corpus = the full passage pool
           (retrieval is meaningful out of the box).
squad      squad (v1.1) validation split (~10.6k rows, ~5 MB slice).
           Corpus = the sampled questions' paragraphs + 3x random
           distractor paragraphs so retrieval isn't trivial.
hotpotqa   hotpot_qa 'distractor' validation split (~7.4k rows,
           ~45 MB slice). Each question ships 10 paragraphs
           (2 gold + 8 distractors); corpus = all paragraphs of the
           sampled rows. Multi-hop — stresses retrieval hardest.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def _write_outputs(out_dir: Path, corpus: list[dict], questions: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for row in questions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "questions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "ground_truth"])
        w.writeheader()
        w.writerows(questions)
    print(f"  corpus.jsonl    : {len(corpus)} passages")
    print(f"  questions.jsonl : {len(questions)} questions (+ questions.csv)")
    print(f"  -> {out_dir}")


def prepare_mini_wiki(n: int, out_dir: Path, seed: int, from_dir: Path | None = None) -> None:
    if from_dir is not None:
        # Local parquet pair (e.g. cloned from a GitHub mirror):
        # <dir>/passages.parquet + <dir>/test.parquet.
        import pandas as pd

        passages = pd.read_parquet(from_dir / "passages.parquet").to_dict("records")
        qa = pd.read_parquet(from_dir / "test.parquet").to_dict("records")
    else:
        from datasets import load_dataset

        passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
        qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")

    corpus = [
        {"text": str(row["passage"]).strip(), "source": f"mini-wiki-{i}"}
        for i, row in enumerate(passages)
        if str(row.get("passage") or "").strip()
    ]
    rng = random.Random(seed)
    rows = rng.sample(list(qa), min(n, len(qa)))
    questions = [
        {"question": str(r["question"]).strip(), "ground_truth": str(r["answer"]).strip()}
        for r in rows
        if str(r.get("question") or "").strip() and str(r.get("answer") or "").strip()
    ]
    _write_outputs(out_dir, corpus, questions)


def _load_squad_local(path: Path) -> list[dict]:
    """Flatten the official SQuAD JSON (dev-v1.1.json from
    github.com/rajpurkar/SQuAD-explorer) into HF-style rows."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for article in raw.get("data", []):
        title = article.get("title", "squad")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers = [a.get("text", "") for a in qa.get("answers", [])]
                rows.append({
                    "question": qa.get("question", ""),
                    "answers": {"text": answers},
                    "context": context,
                    "title": title,
                })
    return rows


def prepare_squad(
    n: int, out_dir: Path, seed: int, distractor_x: int = 3,
    from_file: Path | None = None,
) -> None:
    if from_file is not None:
        ds: Any = _load_squad_local(from_file)
    else:
        from datasets import load_dataset

        ds = load_dataset("squad", split="validation")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    sampled = [ds[i] for i in idxs]

    questions = [
        {
            "question": str(r["question"]).strip(),
            "ground_truth": str((r["answers"]["text"] or [""])[0]).strip(),
        }
        for r in sampled
    ]
    # Corpus: the sampled questions' own paragraphs + random distractor
    # paragraphs (deduped) so retrieval has something to get wrong.
    seen: set[str] = set()
    corpus: list[dict] = []

    def _add(context: str, title: str) -> None:
        text = str(context).strip()
        if text and text not in seen:
            seen.add(text)
            corpus.append({"text": text, "source": title})

    for r in sampled:
        _add(r["context"], str(r.get("title") or "squad"))
    n_distractors = distractor_x * len(sampled)
    for i in rng.sample(range(len(ds)), min(n_distractors * 4, len(ds))):
        if len(corpus) >= len(sampled) + n_distractors:
            break
        r = ds[i]
        _add(r["context"], str(r.get("title") or "squad"))
    _write_outputs(out_dir, corpus, questions)


def prepare_hotpotqa(n: int, out_dir: Path, seed: int, from_file: Path | None = None) -> None:
    rng = random.Random(seed)
    if from_file is not None and from_file.suffix == ".parquet":
        # HF-converted parquet (validation-00000-of-00001.parquet):
        # context is already {"title": [...], "sentences": [[...], ...]}.
        import pandas as pd

        rows = pd.read_parquet(from_file).to_dict("records")
        sampled = rng.sample(rows, min(n, len(rows)))
    elif from_file is not None:
        # Official JSON (hotpot_dev_distractor_v1.json): a list of records
        # whose "context" is [[title, [sentence, ...]], ...].
        raw = json.loads(from_file.read_text(encoding="utf-8"))
        sampled_raw = rng.sample(raw, min(n, len(raw)))
        sampled = [
            {
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "context": {
                    "title": [c[0] for c in r.get("context", [])],
                    "sentences": [c[1] for c in r.get("context", [])],
                },
            }
            for r in sampled_raw
        ]
    else:
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        idxs = rng.sample(range(len(ds)), min(n, len(ds)))
        sampled = [ds[i] for i in idxs]

    questions = [
        {"question": str(r["question"]).strip(), "ground_truth": str(r["answer"]).strip()}
        for r in sampled
    ]
    # Each row carries 10 titled paragraphs (2 gold + 8 distractors).
    seen: set[str] = set()
    corpus: list[dict] = []
    for r in sampled:
        context = r["context"]
        for title, sentences in zip(context["title"], context["sentences"], strict=False):
            text = "".join(sentences).strip()
            if text and text not in seen:
                seen.add(text)
                corpus.append({"text": text, "source": str(title)})
    _write_outputs(out_dir, corpus, questions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=["mini-wiki", "squad", "hotpotqa"])
    ap.add_argument("--n", type=int, default=30, help="questions to sample")
    ap.add_argument("--out", type=str, default="", help="output dir (default datasets/<name>)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--from-file", type=str, default="",
        help="Use locally downloaded data instead of HuggingFace (no "
        "network needed). squad: dev-v1.1.json "
        "(github.com/rajpurkar/SQuAD-explorer); hotpotqa: "
        "hotpot_dev_distractor_v1.json OR the HF-converted "
        "validation-*.parquet; mini-wiki: the DIRECTORY containing "
        "passages.parquet + test.parquet (e.g. a clone of "
        "github.com/gengzll/rag_dataset wikipedia-mini/).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("datasets") / args.dataset
    from_file = Path(args.from_file) if args.from_file else None
    print(f"Preparing {args.dataset} (n={args.n}) ...")
    if args.dataset == "mini-wiki":
        prepare_mini_wiki(args.n, out_dir, args.seed, from_dir=from_file)
    elif args.dataset == "squad":
        prepare_squad(args.n, out_dir, args.seed, from_file=from_file)
    else:
        prepare_hotpotqa(args.n, out_dir, args.seed, from_file=from_file)
    print("\nRun it:")
    print(
        f"  ragdx experiment {out_dir / 'corpus.jsonl'} --has-gt "
        f"--questions {out_dir / 'questions.jsonl'} --mode with_gt"
    )


if __name__ == "__main__":
    main()
