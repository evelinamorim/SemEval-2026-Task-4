# LIAAD INESC TEC at SemEval-2026 Task 4

This repository contains the system developed by LIAAD INESC TEC for [SemEval-2026 Task 4: Narrative Similarity](https://www.codabench.org/competitions/10273/).

## Track A

Our system targets **Track A** only. In this track, each instance is a triplet composed of an anchor story, a story A, and a story B. The system predicts whether story A (`true`) or story B (`false`) is more similar to the anchor.

## `drs_text_baseline.py`

This is the main script implementing our approach. It evaluates five methods:

| Method | Type | Description |
|--------|------|-------------|
| `text_only` | Unsupervised | Cosine similarity of Sentence-BERT embeddings |
| `drs_cosine` | Unsupervised | Cosine similarity of DRS feature vectors |
| `drs_event_type` | Unsupervised | Event type distribution similarity from DRS |
| `hybrid_simple` | Unsupervised | Weighted average of DRS + text similarity scores |
| `hybrid_logistic` / `hybrid_rf` | Supervised | ML classifier trained on DRS + text features (5-fold CV) |

### Hybrid Simple (submitted method)

The `hybrid_simple` method combines DRS structural features with neural text similarity without requiring any training. For each triplet, it computes:

1. **Text similarity** — cosine similarity of Sentence-BERT (`all-mpnet-base-v2`) embeddings of the full story texts.
2. **DRS structural similarity** — an aggregate over DRS-derived features including event type distributions, tense and aspect ratios, temporal relation patterns, semantic relation ratios, event n-gram sequences, and participant role links, all normalized by event count.

The final score is a weighted combination of both (55% text + 45% DRS). The story with the higher combined similarity to the anchor is predicted as closer. No training is required.

DRS annotations are generated automatically using the [text2story](https://github.com/LIAAD/text2story) Python toolkit.

### Usage

```bash
pip install sentence-transformers
python drs_text_baseline.py <drs_dir> <jsonl_path>
```

**Example:**
```bash
python drs_text_baseline.py data/drs/ data/dev_track_a.jsonl
```

## Citation

If you use this code, please cite our system paper:

```
@inproceedings{amorim2026liaad,
  title     = {LIAAD INESCTEC at SemEval-2026 Task 4: Unsupervised Narrative Similarity via Discourse Representation Structures and Sentence Embeddings},
  author    = {Amorim, Evelin and Jorge, Al{\'i}pio and Silvano, Purifica{\c{c}}{\~a}o},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026}
}
```
