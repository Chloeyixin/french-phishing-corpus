# -*- coding: utf-8 -*-
"""CamemBERT fine-tuning baseline: 5-fold stratified CV, class-weighted loss, same protocol."""
import os, json, re, time, math, gc
import numpy as np, openpyxl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, Adafactor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

XLSX = '/home/user/Doubao/chats/38442055174458370/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'
OUT = '/home/user/Doubao/chats/38455438158350082/camembert_results.json'
MAXLEN, EPOCHS, BS, LR = 96, 3, 8, 3e-5
FREEZE_UPTO = 8
torch.set_num_threads(2)

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ph, sm = [], []
ws = wb['钓鱼邮件语料']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]:
        continue
    ph.append(str(row[1]))
ws2 = wb['垃圾短信数据']
for i, row in enumerate(ws2.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]:
        continue
    sm.append(str(row[1]))
wb.close()
docs = ph + sm
y = np.array([1] * len(ph) + [0] * len(sm))
print('corpus', len(docs), 'phishing', int(y.sum()), flush=True)

tok = AutoTokenizer.from_pretrained('camembert-base')


class DS(Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(texts, truncation=True, max_length=MAXLEN, padding=True)
        self.labels = labels
        self.enc = {k: torch.tensor(v) for k, v in self.enc.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, torch.tensor(self.labels[i])


def run_fold(tr_idx, te_idx, fold):
    torch.manual_seed(42)
    np.random.seed(42)
    model = AutoModelForSequenceClassification.from_pretrained('camembert-base', num_labels=2)
    train_ds = DS([docs[i] for i in tr_idx], y[tr_idx])
    test_ds = DS([docs[i] for i in te_idx], y[te_idx])
    dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    tl = DataLoader(test_ds, batch_size=32)
    n_pos, n_neg = int(y[tr_idx].sum()), int(len(tr_idx) - y[tr_idx].sum())
    w = torch.tensor([len(tr_idx) / (2 * n_neg), len(tr_idx) / (2 * n_pos)], dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w)
    for n, p_ in model.roberta.embeddings.named_parameters():
        p_.requires_grad = False
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < FREEZE_UPTO:
            for p_ in layer.parameters():
                p_.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = len(dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        tot = 0.0
        for batch, lab in dl:
            out = model(**batch)
            loss = lossf(out.logits, lab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        rss = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])//1024
        print(f'  fold{fold} ep{ep+1} loss {tot/len(dl):.4f} {time.time()-t0:.0f}s rss={rss}MB', flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch, lab in tl:
            preds.append(model(**batch).logits.argmax(-1).numpy())
    pred = np.concatenate(preds)
    p, r, f1, _ = precision_recall_fscore_support(y[te_idx], pred, average='binary', pos_label=1)
    acc = accuracy_score(y[te_idx], pred)
    print(f'  fold{fold}: P={p:.3f} R={r:.3f} F1={f1:.3f} Acc={acc:.3f}', flush=True)
    return dict(precision=p, recall=r, f1=f1, accuracy=acc)


res = {'folds': [], 'config': dict(maxlen=MAXLEN, epochs=EPOCHS, bs=BS, lr=LR, model='camembert-base', freeze_upto=FREEZE_UPTO, opt='AdamW')}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k, (tr, te) in enumerate(skf.split(docs, y)):
    res['folds'].append(run_fold(tr, te, k + 1))
    gc.collect()
    json.dump(res, open(OUT, 'w'), indent=1)

agg = {m: float(np.mean([f[m] for f in res['folds']])) for m in ('precision', 'recall', 'f1', 'accuracy')}
agg['f1_std'] = float(np.std([f['f1'] for f in res['folds']]))
res['aggregate'] = agg
json.dump(res, open(OUT, 'w'), indent=1)
print('AGGREGATE', json.dumps(agg, indent=1), flush=True)
