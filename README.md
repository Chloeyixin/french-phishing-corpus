# French Phishing Message Corpus

An annotated corpus of 1,591 French messages (phishing, spam, ham) together with the code used in
*Detecting French Phishing Messages under Label Scarcity: A Corpus and Label-Efficiency Benchmark*.

The repository serves two purposes: (i) release the annotated corpus, and (ii) allow every table and
figure in the paper to be regenerated from the released data.

## Corpus

| Channel | Label | Count |
|---|---|---|
| Email | phishing | 168 |
| SMS | phishing | 132 |
| Vishing (transcribed) | phishing | 13 |
| Webpage (form text) | phishing | 4 |
| Physical (letter/notice) | phishing | 2 |
| SMS | spam | 612 |
| SMS | ham | 660 |
| **Total** | | **1,591** |

Phishing messages account for 20% of the corpus; the phishing/non-phishing imbalance is intentional
and is not resampled in any experiment. SMS spam and ham were drawn from a CC-BY-4.0 public SMS
collection; all phishing messages were collected and annotated by the author. Personal identifiers
(phone numbers, email addresses, URLs, account numbers) were removed during cleaning.

Corpus file: `法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx` (sheet `corpus`).

## Repository layout

```
data/            corpus workbook (see file name above)
scripts/         all experiment code (see mapping below)
results/         JSON outputs, one file per experiment
figures/         figure sources
```

## Script-to-result mapping

| Paper element | Script | Output |
|---|---|---|
| Table 3 (unsupervised / seeded K-means) | `scripts/hybrid_kmeans_v3.py` | `results/cv_stratified5.json` |
| Table 5 rows 1–8 (TF-IDF baselines) | `scripts/supervised_baseline.py` | `results/cv_stratified5.json` |
| Table 5, CamemBERT row | `scripts/train_camembert.py` | `results/camembert_results.json` |
| Table 5, group-aware row | `scripts/train_camembert_grouped.py` | `results/camembert_grouped_results.json` |
| Table 6, Figure 4 (label efficiency) | `scripts/label_efficiency.py` | `results/label_efficiency.json` |
| Section 6.5, SMS channel subset | `scripts/sms_channel_svm.py` | `results/sms_channel_svm.json` |
| Section 6.5, near-duplicate audit and grouped CV | `scripts/neardup_audit.py` | `results/neardup_audit.json`, `results/cv_grouped5.json` |
| Figure 3 (three supervision levels) | `scripts/hybrid_kmeans.py` | `results/cv_stratified5.json` |

## Reported results

Five-fold cross-validation, phishing as the positive class. Mean ± standard deviation.

| Model | P | R | F1 | Acc |
|---|---|---|---|---|
| K-means, no supervision | 0.197 | 0.966 | 0.327 | — |
| Seeded K-means, fixed seed | 0.742 | 0.605 | 0.667 | — |
| Multinomial NB, TF-IDF | — | — | 0.800 | — |
| k-NN, TF-IDF | — | — | 0.552 | — |
| SVM, TF-IDF | 0.971 | 0.824 | 0.860 | 0.960 |
| Logistic regression, TF-IDF | 0.967 | 0.837 | 0.860 | 0.962 |
| CamemBERT, partial fine-tuning | 0.942 | 0.959 | 0.950 ± 0.015 | 0.980 |
| CamemBERT, group-aware CV | 0.951 | 0.965 | 0.958 ± 0.010 | 0.983 |

Per-fold F1 for CamemBERT: 0.953 / 0.969 / 0.924 / 0.952 / 0.953 (standard CV) and
0.947 / 0.960 / 0.952 / 0.977 / 0.955 (group-aware CV). The group-aware variant constrains messages
belonging to the same near-duplicate cluster to the same fold.

Label efficiency (Table 6): seeded K-means reaches F1 0.687 / 0.617 / 0.777 / 0.843 / 0.862 / 0.882 at
5 / 10 / 20 / 40 / 80 / 160 labelled messages per class, against 0.492 / 0.554 / 0.688 / 0.787 / 0.827 /
0.874 for the supervised SVM. Semi-supervised learning is ahead below roughly 80 labels per class, and
40 labels per class is the point where the curves stop being unstable.

## Environment

Python 3.11, scikit-learn 1.4.0, torch 2.15.0 (CPU), transformers 5.23.0, sentencepiece, openpyxl 3.1.5,
matplotlib 3.11.0. All runs were executed on a 2-core CPU machine with 4 GB of RAM; no GPU was used.

### CamemBERT configuration

`camembert-base`, max sequence length 96, batch size 8, AdamW with lr 3e-5, weight decay 0.01, 3 epochs,
10% warmup, gradient clipping 1.0, class-weighted cross-entropy. The embeddings and transformer layers
0–7 are frozen, leaving roughly 28 M trainable parameters. This is a memory-driven choice: full
fine-tuning on the same machine exceeded the available RAM. The reported scores should therefore be
read as a conservative lower bound for a French encoder on this task, not as the best achievable
transformer result.

## Reproducing

```
pip install -r requirements.txt
python scripts/supervised_baseline.py        # TF-IDF baselines, Table 5
python scripts/train_camembert.py            # CamemBERT, Table 5
python scripts/train_camembert_grouped.py    # group-aware CV, Table 5
python scripts/label_efficiency.py           # Table 6, Figure 4
python scripts/neardup_audit.py              # near-duplicate audit, Section 6.5
```

Each script writes its JSON to `results/`. Random seeds are fixed and recorded in the output files.

## Ethics and licensing

All messages were collected from publicly accessible sources or from the author's own devices, with
personal identifiers removed before annotation. No message contains a live phishing URL, and no
credential, phone number, or account number is reproduced. The annotated corpus is released for
research use; the SMS spam and ham portion remains under CC-BY-4.0 and is redistributed here with
attribution.

## Citation

```
@article{yi2026frenchphishing,
  title   = {Detecting French Phishing Messages under Label Scarcity: A Corpus and Label-Efficiency Benchmark},
  author  = {Yi, Xin},
  year    = {2026},
  note    = {Manuscript submitted to Computers \& Security}
}
```
