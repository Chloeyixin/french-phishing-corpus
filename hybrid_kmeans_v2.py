# -*- coding: utf-8 -*-
"""变体实验：A=高区分度手工特征 K-means；B=Seeded K-means（官方钓鱼初始化质心）"""
import json, re
import numpy as np
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             precision_recall_fscore_support)
from scipy.sparse import hstack, csr_matrix, vstack

XLSX = 'data/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'
STOP = set("""le la les un une des et ou mais donc or ni car à de du au aux en dans sur sous avec sans par pour que qui quoi dont où ne pas plus moins très tout tous toute toutes ce cet cette ces il elle ils elles on nous vous je tu mon ma mes ton ta tes son sa ses notre votre leur nos vos leur se si y en a est sont était étaient être avoir avait avaient fait faites faire peut peuvent doit doivent comme aussi ainsi alors lorsque quand après avant entre contre chez vers parmi depuis pendant tant selon""".split())
URL_RE = re.compile(r'https?://|www\.|\b[a-z0-9-]+\.(com|net|org|fr|eu|io|ly|xyz|top|club|site|info)\b', re.I)
SHORT_RE = re.compile(r'\b(bit\.ly|t\.co|goo\.gl|tinyurl|shorturl|ow\.ly|is\.gd|buff\.ly|cutt\.ly)\b|https?://[^\s]{1,12}\b', re.I)
MONEY_RE = re.compile(r'[€$£]|\b\d[\d\s.,]{2,}\s?(euros?|€|dollars?)\b', re.I)
BANK = re.compile(r'\b(banque|compte|relevé|identifiant|identif|code|mot de passe|carte|crédit|virement|transaction|sécurit|paypal|amazon|dhl|la poste|imp[ôô]ts|police|gendarmerie|arcep|free|orange|sosh|bouygues)\b', re.I)
THREAT = re.compile(r'\b(p[ée]nalit[ée]|amende|poursuites|pirat[ée]|ransomware|webcam|chantage|dossier|proc[èe]s|prison|virus|infection)\b', re.I)
GREETING = re.compile(r'\b(bonjour|madame|monsieur|ch[èe]re?|salut|bonsoir|cordialement)\b', re.I)

def clean(t):
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\b\d[\d\s\-.()/]*\d', ' ', t.lower())
    t = re.sub(r'[^\w\sàâäéèêëîïôöùûüçœ]', ' ', t, flags=re.UNICODE)
    return ' '.join(w for w in t.split() if w not in STOP and len(w) > 2)

def feats(t):
    tl = t.lower()
    return [1 if URL_RE.search(tl) else 0,
            1 if SHORT_RE.search(tl) else 0,
            1 if MONEY_RE.search(tl) else 0,
            1 if BANK.search(tl) else 0,
            1 if THREAT.search(tl) else 0,
            len(t)]

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ph, sm = [], []
ws = wb['钓鱼邮件语料']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    ph.append(str(row[1]))
ws2 = wb['垃圾短信数据']
for i, row in enumerate(ws2.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    sm.append({'text': str(row[1]), 'label': int(row[2] if row[2] is not None else 1)})
wb.close()

docs = ph + [s['text'] for s in sm]
true = np.array([1]*len(ph) + [0]*len(sm))
n_ph = len(ph)

FNAMES = ['has_url','short_url','money','bank','threat','len']
F = np.array([feats(t) for t in docs], dtype=float)
F[:, -1] = F[:, -1] / 400.0

vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
T = vec.fit_transform([clean(d) for d in docs])

def evals(pred, name, store):
    ph_share = [float((pred[true==1]==c).mean()) if (pred==c).sum() else 0.0 for c in range(np.max(pred)+1)]
    pc = int(np.argmax(ph_share))
    yb = (pred == pc).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(true, yb, average='binary', pos_label=1)
    store[name] = dict(precision=round(prec,4), recall=round(rec,4), f1=round(f1,4),
                       ari=round(adjusted_rand_score(true,pred),4),
                       nmi=round(normalized_mutual_info_score(true,pred),4),
                       ph_dist=[int((pred[true==1]==c).sum()) for c in range(np.max(pred)+1)])
    print(f'  {name:32s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} ARI={adjusted_rand_score(true,pred):.3f} 钓鱼分布={store[name]["ph_dist"]}')

R = {}
# A：只用高区分度手工特征（6维）K-means
print('[A] 手工特征(6维) K-means')
Xa = csr_matrix(F)
for k in [2,3]:
    evals(KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xa), f'A_featonly_k{k}', R)

# B：TF-IDF + 特征 拼接，seeded 初始化：10条钓鱼 + 10条正常短信作质心
print('[B] TF-IDF+特征, seeded(10钓鱼+10正常) K=2')
Xb = hstack([T, csr_matrix(F)]).tocsr()
seed_idx = list(range(10)) + [n_ph + 600 + i for i in range(10)]  # 前10钓鱼 + 短信区后10条(正常区)
c0 = np.asarray(Xb[seed_idx[:10]].mean(axis=0)).ravel(); c1 = np.asarray(Xb[seed_idx[10:]].mean(axis=0)).ravel()
init_cent = np.vstack([c0, c1])
km = KMeans(n_clusters=2, init=init_cent, n_init=1, random_state=42)
evals(km.fit_predict(Xb), 'B_seeded_k2', R)

# C：纯TF-IDF seeded 对照（同种子，证明是种子作用还是特征作用）
print('[C] 纯TF-IDF, seeded(10钓鱼+10正常) K=2')
c0c = np.asarray(T[seed_idx[:10]].mean(axis=0)).ravel(); c1c = np.asarray(T[seed_idx[10:]].mean(axis=0)).ravel()
init_cent_c = np.vstack([c0c, c1c])
kmc = KMeans(n_clusters=2, init=init_cent_c, n_init=1, random_state=42)
evals(kmc.fit_predict(T), 'C_tfidf_seeded_k2', R)

json.dump(R, open('hybrid_result_v2.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved hybrid_result_v2.json')
