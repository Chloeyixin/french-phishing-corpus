# French Phishing Corpus + Multi-Level Detection Baselines / 法语钓鱼语料库与多层级检测基线

[English](#english) | [中文](#中文)

---

## English

**A French Phishing Corpus and Multi-Level Detection Baselines** — data and reproducible scripts accompanying the manuscript submitted to *Language Resources and Evaluation*.

### About
- **Corpus size**: 1,591 French messages
  - **319 phishing samples** collected from official French sources (Cybermalveillance.gouv.fr, bank and telecommunication operator security pages, French administrative institutions), each with a verbatim French original, scam type, impersonated entity, rationale, and traceable source URL.
  - **1,272 SMS messages** (612 spam / 660 ham) from the SpamShield multilingual dataset (CC-BY-4.0).
- **Baselines reported** (unified evaluation protocol, phishing = positive class):
  - Unsupervised K-means (TF-IDF, K=2): Precision 0.197, Recall 0.966, **F1 = 0.327**
  - Semi-supervised seeded K-means (10 phish + 10 ham seeds): Precision 0.742, Recall 0.605, **F1 = 0.667**
  - Phishing-signal weighted ranking: Top-20 hit rate 100%, Top-81 hit rate 91.4% (random baseline 20%)
  - Supervised SVM / Logistic Regression (plain TF-IDF, 5-fold CV): **F1 = 0.860**, Accuracy 0.960

### Data
| File | Description |
|---|---|
| `data/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx` | Full corpus, two sheets (phishing / SMS); **input used by all scripts** |
| `data/法语钓鱼语料_319条.csv` | Phishing subset, 9 columns (ID, French text, type, impersonated entity, technique, rationale, label, source, remarks) |
| `data/法语垃圾短信_1272条.csv` | SMS subset, 5 columns |

### Reproduce
```bash
pip install pandas scikit-learn openpyxl numpy
cd scripts
python3 hybrid_kmeans.py        # Table: unsupervised baselines
python3 hybrid_kmeans_feat.py   # feature-discriminativeness analysis
python3 hybrid_kmeans_v2.py     # seeded K-means (K=2 / K=3)
python3 score_ranking.py        # Top-N ranking
python3 supervised_baseline.py  # supervised 5-fold CV comparison
```
All scripts read `data/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx` (relative path) and print the metrics reported in the paper. Random seeds are fixed (42) for reproducibility.

### License
- Phishing subset (collected from official public sources): released for research use; please cite this manuscript and the original sources when reusing.
- SMS subset: SpamShield dataset, **CC-BY-4.0** (https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets).
- Scripts: MIT.

### Citation (to be completed after acceptance)
Xin Yi. A French Phishing Corpus and Multi-Level Detection Baselines: Unsupervised, Semi-Supervised, and Supervised Evaluation. *Language Resources and Evaluation* (submitted).

---

## 中文

**法语钓鱼语料库与多层级检测基线** — 配套论文（投稿《Language Resources and Evaluation》）的数据与可复现脚本。

### 简介
- **语料规模**：1,591 条法语消息
  - **319 条钓鱼样本**，全部来自法国官方/权威来源（Cybermalveillance.gouv.fr、各银行与运营商安全公告、法国行政机构），每条含法语原文、诈骗类型、伪装对象、判定依据与可追溯来源。
  - **1,272 条短信**（垃圾 612 / 正常 660），来自 SpamShield 多语数据集（CC-BY-4.0）。
- **报告基线**（统一评估口径，钓鱼为正类）：
  - 无监督 K-means（TF-IDF, K=2）：精确率 0.197，召回率 0.966，**F1 = 0.327**
  - 半监督 seeded K-means（10 钓鱼 + 10 正常种子）：**F1 = 0.667**
  - 钓鱼信号加权排序：Top-20 命中率 100%，Top-81 命中率 91.4%（随机基线 20%）
  - 监督 SVM / 逻辑回归（纯 TF-IDF，5 折交叉验证）：**F1 = 0.860**，准确率 0.960

### 复现
```bash
pip install pandas scikit-learn openpyxl numpy
cd scripts
python3 hybrid_kmeans.py        # 无监督基线
python3 hybrid_kmeans_feat.py   # 特征区分度分析
python3 hybrid_kmeans_v2.py     # seeded K-means（K=2 / K=3）
python3 score_ranking.py        # Top-N 排序
python3 supervised_baseline.py  # 监督 5 折对比
```
所有脚本读取 `data/` 下同名 xlsx（相对路径），随机种子固定为 42。

### 许可
- 钓鱼子集（采集自官方公开来源）：供研究使用，引用本论文及原始来源即可。
- 短信子集：SpamShield 数据集，**CC-BY-4.0**。
- 脚本：MIT。
