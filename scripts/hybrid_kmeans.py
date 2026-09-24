# -*- coding: utf-8 -*-
"""混合语料 K-means 实验：81 钓鱼 + 1272 垃圾短信，能否自动分出钓鱼簇"""
import json, re, string
import numpy as np
import pandas as pd
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             precision_recall_fscore_support, accuracy_score)

XLSX = 'data/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'

STOP = set("""le la les un une des et ou mais donc or ni car à de du au aux en dans sur sous avec sans par pour que qui quoi dont où ne pas plus moins très tout tous toute toutes ce cet cette ces il elle ils elles on nous vous je tu mon ma mes ton ta tes son sa ses notre votre leur nos vos leurs se si y en a est sont était étaient être avoir avait avaient fait faites faire peut peuvent doit doivent comme aussi ainsi alors lorsque quand après avant entre contre chez vers parmi depuis pendant tant selon""".split())

def clean(t):
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\b\d[\d\s\-.()/]*\d', ' ', t.lower())
    t = re.sub(r'[^\w\sàâäéèêëîïôöùûüçœ]', ' ', t, flags=re.UNICODE)
    toks = [w for w in t.split() if w not in STOP and len(w) > 2]
    return ' '.join(toks)

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ph = []
ws = wb['钓鱼邮件语料']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0: continue
    if row[0] and row[1]:
        ph.append({'id': str(row[0]), 'text': str(row[1]), 'type': str(row[2] or '')})
sm = []
ws2 = wb['垃圾短信数据']
for i, row in enumerate(ws2.iter_rows(values_only=True)):
    if i == 0: continue
    if row[0] and row[1]:
        sm.append({'id': str(row[0]), 'text': str(row[1]), 'label': int(row[2] if row[2] is not None else 1), 'cat': str(row[3] or '')})
wb.close()

docs  = [p['text'] for p in ph] + [s['text'] for s in sm]
true  = np.array([1]*len(ph) + [0]*len(sm))
kind  = np.array(['钓鱼']*len(ph) + ['垃圾' if s['label']==1 else '正常' for s in sm])
ids   = [p['id'] for p in ph] + [s['id'] for s in sm]
clean_docs = [clean(d) for d in docs]

vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
X = vec.fit_transform(clean_docs)
feat = vec.get_feature_names_out()

results = {}
for k in [2,3,4,5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred = km.fit_predict(X)
    # 簇 -> 钓鱼占比
    ph_share = [float((pred[true==1]==c).mean()) if (pred==c).sum() else 0.0 for c in range(k)]
    # 主实验：K=2 时把钓鱼占比高的簇当"预测钓鱼簇"
    if k == 2:
        ph_cluster = int(np.argmax(ph_share))
        y_pred_bin = (pred == ph_cluster).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(true, y_pred_bin, average='binary', pos_label=1)
        acc = accuracy_score(true, y_pred_bin)
        ari = adjusted_rand_score(true, pred)
        nmi = normalized_mutual_info_score(true, pred)
        results['k2'] = dict(ph_cluster=ph_cluster, precision=round(prec,4), recall=round(rec,4),
                             f1=round(f1,4), accuracy=round(acc,4), ari=round(ari,4), nmi=round(nmi,4),
                             ph_share=[round(x,4) for x in ph_share],
                             ph_wrong=[ids[i] for i in range(len(true)) if true[i]==1 and pred[i]!=ph_cluster])
        # 钓鱼样本在各簇的分布
        results['k2_ph_dist'] = [int((pred[true==1]==c).sum()) for c in range(k)]
    cluster_comp = []
    for c in range(k):
        idx = np.where(pred==c)[0]
        n_ph = int((true[idx]==1).sum())
        n_spam = int(((kind == '垃圾') & (pred==c)).sum())
        n_norm = int(((kind == '正常') & (pred==c)).sum())
        # 簇内 top 词（均值 TF-IDF）
        sub = X[idx].mean(axis=0).A1
        top = [str(feat[i]) for i in np.argsort(sub)[-8:][::-1]]
        cluster_comp.append(dict(cluster=int(c), size=int(len(idx)),
                                 n_phishing=int(n_ph), n_spam=n_spam, n_normal=n_norm,
                                 top_words=[str(t) for t in top]))
    results[f'k{k}_comp'] = cluster_comp

# 降维可视化（固定随机种子）
pca = PCA(n_components=2, random_state=42)
coords_pca = pca.fit_transform(X.toarray()).tolist()
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
pred2 = km2.fit_predict(X)
ph_cluster2 = int(np.argmax([float((pred2[true==1]==c).mean()) if (pred2==c).sum() else 0.0 for c in range(2)]))
results['viz'] = dict(coords_pca=coords_pca, true=[int(t) for t in true], kind=[str(x) for x in kind],
                      pred2=[int(p) for p in pred2], ph_cluster=ph_cluster2)
results['n'] = dict(total=len(docs), phishing=len(ph), sms=len(sm),
                    spam=sum(1 for s in sm if s['label']==1), normal=sum(1 for s in sm if s['label']==0))

json.dump(results, open('hybrid_result.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(results['k2'], ensure_ascii=False))
print('样本总数', len(docs), '钓鱼', len(ph), '垃圾短信', len(sm))
print('钓鱼样本在 K=2 两簇分布:', results['k2_ph_dist'])
for c in results['k2_comp']:
    print(f"簇{c['cluster']}: 规模{c['size']} 钓鱼{c['n_phishing']} 垃圾短信{c['n_spam']} 正常{c['n_normal']} top词={c['top_words'][:5]}")
