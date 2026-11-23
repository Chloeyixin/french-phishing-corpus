# -*- coding: utf-8 -*-
"""Label-efficiency sweep: seeded K-means vs supervised SVM at fixed label budgets."""
import json, re, random
import numpy as np, openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, f1_score

XLSX = '/home/user/Doubao/chats/38442055174458370/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'
STOP = set("""le la les un une des et ou mais donc or ni car à de du au aux en dans sur sous avec sans par pour que qui quoi dont où ne pas plus moins très tout tous toute toutes ce cet cette ces il elle ils elles on nous vous je tu mon ma mes ton ta tes son sa ses notre votre leur nos vos leur se si y en a est sont était étaient être avoir avait avaient fait faites faire peut peuvent doit doivent comme aussi ainsi alors lorsque quand après avant entre contre chez vers parmi depuis pendant tant selon""".split())

def clean(t):
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\b\d[\d\s\-.()/]*\d', ' ', t.lower())
    t = re.sub(r'[^\w\sàâäéèêëîïôöùûüçœ]', ' ', t, flags=re.UNICODE)
    return ' '.join(w for w in t.split() if w not in STOP and len(w) > 2)

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ph, sm = [], []
ws = wb['钓鱼邮件语料']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    ph.append(str(row[1]))
ws2 = wb['垃圾短信数据']
for i, row in enumerate(ws2.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    lab = int(row[2] if row[2] is not None else 1)
    sm.append({'text': str(row[1]), 'label': lab})
wb.close()

docs = ph + [s['text'] for s in sm]
y = np.array([1] * len(ph) + [0] * len(sm))
n_ph = len(ph)
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
T = vec.fit_transform([clean(d) for d in docs])
print('docs', T.shape, 'phishing', n_ph)

ph_idx = np.where(y == 1)[0]
ng_idx = np.where(y == 0)[0]
BUDGETS = [5, 10, 20, 40, 80, 160]
REPEATS = 10
rng = np.random.default_rng(42)

def seeded(n, repeats=REPEATS):
    f1s = []
    for r in range(repeats):
        seeds_p = rng.choice(ph_idx, n, replace=False)
        seeds_n = rng.choice(ng_idx, n, replace=False)
        c_p = np.asarray(T[seeds_p].mean(axis=0)).ravel()
        c_n = np.asarray(T[seeds_n].mean(axis=0)).ravel()
        init = np.vstack([c_p, c_n])
        pred = KMeans(n_clusters=2, init=init, n_init=1, random_state=42).fit_predict(T)
        # map cluster with higher phishing share to positive
        share = [float((pred[y == 1] == c).mean()) if (pred == c).sum() else 0.0 for c in (0, 1)]
        pc = int(np.argmax(share))
        yb = (pred == pc).astype(int)
        f1s.append(f1_score(y, yb, pos_label=1))
    return float(np.mean(f1s)), float(np.std(f1s))

def supervised(n, repeats=5):
    f1s = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for r in range(repeats):
        for tr, te in skf.split(T, y):
            ptr = np.array([i for i in tr if y[i] == 1])
            ntr = np.array([i for i in tr if y[i] == 0])
            if len(ptr) < n or len(ntr) < n: continue
            sp = rng.choice(ptr, n, replace=False)
            sn = rng.choice(ntr, n, replace=False)
            idx = np.concatenate([sp, sn])
            clf = LinearSVC(class_weight='balanced', random_state=42)
            clf.fit(T[idx], y[idx])
            f1s.append(f1_score(y[te], clf.predict(T[te]), pos_label=1))
    return float(np.mean(f1s)), float(np.std(f1s))

res = {'seeded': {}, 'supervised': {}}
for n in BUDGETS:
    m, s = seeded(n)
    res['seeded'][n] = [round(m, 3), round(s, 3)]
    print(f'seeded n={n:3d} F1={m:.3f} +-{s:.3f}')
for n in BUDGETS:
    m, s = supervised(n)
    res['supervised'][n] = [round(m, 3), round(s, 3)]
    print(f'supervised n={n:3d} F1={m:.3f} +-{s:.3f}')

json.dump(res, open('/home/user/Doubao/chats/38455438158350082/label_efficiency.json', 'w', encoding='utf-8'), indent=1)
