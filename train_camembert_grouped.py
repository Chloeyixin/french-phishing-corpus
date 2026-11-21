# -*- coding: utf-8 -*-
"""CamemBERT, group-aware 5-fold CV (near-duplicate clusters kept inside one fold)."""
import os, json, re, time, gc
import numpy as np, openpyxl
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

XLSX = '/home/user/Doubao/chats/38442055174458370/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'
OUT = '/home/user/Doubao/chats/38455438158350082/camembert_grouped_results.json'
MAXLEN, EPOCHS, BS, LR, FREEZE_UPTO = 96, 3, 8, 3e-5, 8
torch.set_num_threads(2)

STOP = set("""le la les un une des et ou mais donc or ni car à de du au aux en dans sur sous avec sans par pour que qui quoi dont où ne pas plus moins très tout tous toute toutes ce cet cette ces il elle ils elles on nous vous je tu mon ma mes ton ta tes son sa ses notre votre leur nos vos leur se si y en a est sont était étaient être avoir avait avaient fait faites faire peut peuvent doit doivent comme aussi ainsi alors lorsque quand après avant entre contre chez vers parmi depuis pendant tant selon""".split())


def clean(t):
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\b\d[\d\s\-.()/]*\d', ' ', t.lower())
    t = re.sub(r'[^\w\sàâäéèêëîïôöùûüçœ]', ' ', t, flags=re.UNICODE)
    return ' '.join(w for w in t.split() if w not in STOP and len(w) > 2)


wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
rows = [r for r in wb['钓鱼邮件语料'].iter_rows(values_only=True)][1:]
sms = [r for r in wb['垃圾短信数据'].iter_rows(values_only=True)][1:]
wb.close()
docs = [str(r[1]) for r in rows if r[1]] + [str(r[1]) for r in sms if len(r) > 1 and r[1]]
y = np.array([1] * len([r for r in rows if r[1]]) + [0] * len([r for r in sms if len(r) > 1 and r[1]]))

# near-duplicate clusters (cosine >= 0.9 on the same TF-IDF space)
V = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
X = V.fit_transform([clean(t) for t in docs])
S = cosine_similarity(X); np.fill_diagonal(S, 0)
pairs = np.argwhere(np.triu(S, 1) >= 0.9)
par = list(range(len(docs)))


def f(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


for a, b in pairs:
    ra, rb = f(a), f(b)
    if ra != rb:
        par[ra] = rb
groups = np.array([f(i) for i in range(len(docs))])
print('corpus', len(docs), 'pos', int(y.sum()), 'near-dup pairs', len(pairs), 'clusters', len(set(groups)), flush=True)

tok = AutoTokenizer.from_pretrained('camembert-base')


class DS(Dataset):
    def __init__(self, texts, labels):
        enc = tok(texts, truncation=True, max_length=MAXLEN, padding=True)
        self.enc = {k: torch.tensor(v) for k, v in enc.items()}
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, torch.tensor(self.labels[i])


def run_fold(tr_idx, te_idx, fold):
    torch.manual_seed(42); np.random.seed(42)
    model = AutoModelForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    for p_ in model.roberta.embeddings.parameters():
        p_.requires_grad = False
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < FREEZE_UPTO:
            for p_ in layer.parameters():
                p_.requires_grad = False
    dl = DataLoader(DS([docs[i] for i in tr_idx], y[tr_idx]), batch_size=BS, shuffle=True)
    tl = DataLoader(DS([docs[i] for i in te_idx], y[te_idx]), batch_size=32)
    n_pos, n_neg = int(y[tr_idx].sum()), int(len(tr_idx) - y[tr_idx].sum())
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([len(tr_idx) / (2 * n_neg), len(tr_idx) / (2 * n_pos)], dtype=torch.float))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = len(dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time(); tot = 0.0
        for batch, lab in dl:
            loss = lossf(model(**batch).logits, lab)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        print(f'  gfold{fold} ep{ep+1} loss {tot/len(dl):.4f} {time.time()-t0:.0f}s', flush=True)
    model.eval(); preds = []
    with torch.no_grad():
        for batch, lab in tl:
            preds.append(model(**batch).logits.argmax(-1).numpy())
    pred = np.concatenate(preds)
    pr, rc, f1v, _ = precision_recall_fscore_support(y[te_idx], pred, average='binary', pos_label=1)
    acc = accuracy_score(y[te_idx], pred)
    print(f'  gfold{fold}: P={pr:.3f} R={rc:.3f} F1={f1v:.3f} Acc={acc:.3f} (pos rate pred {pred.mean():.3f})', flush=True)
    del model; gc.collect()
    return dict(precision=pr, recall=rc, f1=f1v, accuracy=acc, pred_pos_rate=float(pred.mean()))


res = {'folds': [], 'config': dict(maxlen=MAXLEN, epochs=EPOCHS, bs=BS, lr=LR, freeze_upto=FREEZE_UPTO, split='StratifiedGroupKFold(groups=near-dup clusters, cos>=0.9)')}
for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(docs, y, groups=groups)):
    res['folds'].append(run_fold(tr, te, k + 1))
    json.dump(res, open(OUT, 'w'), indent=1)

agg = {m: float(np.mean([f[m] for f in res['folds']])) for m in ('precision', 'recall', 'f1', 'accuracy')}
agg['f1_std'] = float(np.std([f['f1'] for f in res['folds']]))
res['aggregate'] = agg
json.dump(res, open(OUT, 'w'), indent=1)
print('AGGREGATE', json.dumps(agg, indent=1), flush=True)
