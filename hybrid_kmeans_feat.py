# -*- coding: utf-8 -*-
"""增强特征版：TF-IDF + 手工钓鱼信号特征 → K-means，与纯TF-IDF基线对比"""
import json, re
import numpy as np
import pandas as pd
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             precision_recall_fscore_support, accuracy_score)
from scipy.sparse import hstack, csr_matrix

XLSX = 'data/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'
STOP = set("""le la les un une des et ou mais donc or ni car à de du au aux en dans sur sous avec sans par pour que qui quoi dont où ne pas plus moins très tout tous toute toutes ce cet cette ces il elle ils elles on nous vous je tu mon ma mes ton ta tes son sa ses notre votre leur nos vos leur se si y en a est sont était étaient être avoir avait avaient fait faites faire peut peuvent doit doivent comme aussi ainsi alors lorsque quand après avant entre contre chez vers parmi depuis pendant tant selon""".split())

URL_RE = re.compile(r'https?://|www\.|\b[a-z0-9-]+\.(com|net|org|fr|eu|io|ly|xyz|top|club|site|info)\b', re.I)
SHORT_RE = re.compile(r'\b(bit\.ly|t\.co|goo\.gl|tinyurl|shorturl|ow\.ly|is\.gd|buff\.ly|cutt\.ly)\b|https?://[^\s]{1,12}\b', re.I)
MONEY_RE = re.compile(r'[€$£]|\b\d[\d\s.,]{2,}\s?(euros?|€|dollars?)\b', re.I)
URGENCY = re.compile(r'\b(urgent|immédiat|immédiatement|maintenant|vite|dernier|dernière|expire|expiré|action requise|avant\s+\d|délai)\b', re.I)
PRIZE = re.compile(r'\b(gagn[ée]|prix|cadeau|lot|concours|gratuit|gratuite|récompense|tirage)\b', re.I)
BANK = re.compile(r'\b(banque|compte|relevé|identifiant|identif|code|mot de passe|carte|crédit|virement|transaction|sécurit|paypal|amazon|dhl|la poste|imp[ôô]ts|police|gendarmerie|arcep|free|orange|sosh|bouygues)\b', re.I)
VERIFY = re.compile(r'\b(v[ée]rif|s[ée]curis|confirm|bloqu[ée]|suspendu|verrouill[ée]|connexion|session|mot de passe|d[ée]bloqu)\b', re.I)
THREAT = re.compile(r'\b(p[ée]nalit[ée]|amende|poursuites|pirat[ée]|ransomware|webcam|chantage|dossier|proc[èe]s|prison|virus|infection)\b', re.I)
CALLBACK = re.compile(r'\b(appelez|rappelez|composez|08\s?\d{2}|0899|118\s?\d{3}|num[ée]ro)\b', re.I)
ATTACH = re.compile(r'\b(pi[èe]ce jointe|t[ée]l[ée]charger|document|facture|fichier|pdf|attachment)\b', re.I)
GREETING = re.compile(r'\b(bonjour|madame|monsieur|ch[èe]re?|salut|bonsoir|cordialement)\b', re.I)
PHONE = re.compile(r'\b(0[1-9])([\s.]\d{2}){4}\b|\b\+?\d{2}\s?\d{1,2}[\s.]\d{2}[\s.]\d{2}[\s.]\d{2}\b')

def clean(t):
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\b\d[\d\s\-.()/]*\d', ' ', t.lower())
    t = re.sub(r'[^\w\sàâäéèêëîïôöùûüçœ]', ' ', t, flags=re.UNICODE)
    return ' '.join(w for w in t.split() if w not in STOP and len(w) > 2)

def feats(t):
    tl = t.lower()
    return [1 if URL_RE.search(tl) else 0,
            1 if SHORT_RE.search(tl) else 0,
            1 if MONEY_RE.search(tl) else 0,
            1 if URGENCY.search(tl) else 0,
            1 if PRIZE.search(tl) else 0,
            1 if BANK.search(tl) else 0,
            1 if VERIFY.search(tl) else 0,
            1 if THREAT.search(tl) else 0,
            1 if CALLBACK.search(tl) else 0,
            1 if ATTACH.search(tl) else 0,
            0 if GREETING.search(tl) else 1,  # 缺称呼=1
            1 if PHONE.search(tl) else 0,
            len(t)]

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ph, sm = [], []
ws = wb['钓鱼邮件语料']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    ph.append({'id': str(row[0]), 'text': str(row[1])})
ws2 = wb['垃圾短信数据']
for i, row in enumerate(ws2.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    sm.append({'id': str(row[0]), 'text': str(row[1]), 'label': int(row[2] if row[2] is not None else 1)})
wb.close()

docs  = [p['text'] for p in ph] + [s['text'] for s in sm]
true  = np.array([1]*len(ph) + [0]*len(sm))
kind  = np.array(['钓鱼']*len(ph) + ['垃圾' if s['label']==1 else '正常' for s in sm])
ids   = [p['id'] for p in ph] + [s['id'] for s in sm]

FNAMES = ['has_url','short_url','money','urgency','prize','bank','verify','threat','callback','attach','no_greeting','phone','len']
F = np.array([feats(t) for t in docs], dtype=float)
F[:, -1] = F[:, -1] / 400.0  # 长度归一化到 0~1 量级（400字封顶）

vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
T = vec.fit_transform([clean(d) for d in docs])
X = hstack([T, csr_matrix(F)]).tocsr()
feat_names = list(vec.get_feature_names_out()) + FNAMES

def run(k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred = km.fit_predict(X)
    ph_share = [float((pred[true==1]==c).mean()) if (pred==c).sum() else 0.0 for c in range(k)]
    comp = []
    for c in range(k):
        idx = np.where(pred==c)[0]
        sub = X[idx].mean(axis=0).A1
        top = [str(feat_names[i]) for i in np.argsort(sub)[-10:][::-1]]
        comp.append(dict(cluster=int(c), size=int(len(idx)),
                         n_phishing=int((true[idx]==1).sum()),
                         n_spam=int(((kind=='垃圾') & (pred==c)).sum()),
                         n_normal=int(((kind=='正常') & (pred==c)).sum()),
                         top_words=top[:8]))
    return pred, ph_share, comp

res = {}
pred2, ph_share2, comp2 = run(2)
ph_cluster = int(np.argmax(ph_share2))
y_pred_bin = (pred2 == ph_cluster).astype(int)
prec, rec, f1, _ = precision_recall_fscore_support(true, y_pred_bin, average='binary', pos_label=1)
res['k2'] = dict(ph_cluster=ph_cluster, precision=round(prec,4), recall=round(rec,4), f1=round(f1,4),
                 accuracy=round(accuracy_score(true, y_pred_bin),4),
                 ari=round(adjusted_rand_score(true, pred2),4),
                 nmi=round(normalized_mutual_info_score(true, pred2),4),
                 ph_share=[round(x,4) for x in ph_share2],
                 ph_wrong=[ids[i] for i in range(len(true)) if true[i]==1 and pred2[i]!=ph_cluster],
                 k2_ph_dist=[int((pred2[true==1]==c).sum()) for c in range(2)])
res['k2_comp'] = comp2
for k in [3,4,5]:
    pred, _, comp = run(k)
    res[f'k{k}_comp'] = comp

# 各手工特征的区分度（钓鱼 vs 非钓鱼 的占比差）
def rate(mask): return F[mask].mean(axis=0)
res['feat_profile'] = {f: {'phishing': round(float(rate(true==1)[i]),3), 'non_phishing': round(float(rate(true==0)[i]),3)}
                       for i, f in enumerate(FNAMES)}

json.dump(res, open('hybrid_result_feat.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('=== 增强特征 K=2 ===')
print(json.dumps({k:v for k,v in res['k2'].items() if k!='ph_wrong'}, ensure_ascii=False))
print('钓鱼分布:', res['k2']['k2_ph_dist'])
for c in comp2:
    print(f"簇{c['cluster']}: 规模{c['size']} 钓鱼{c['n_phishing']} 垃圾{c['n_spam']} 正常{c['n_normal']} top={c['top_words'][:6]}")
print('=== 特征区分度（钓鱼占比 vs 非钓鱼占比）===')
for f, p in res['feat_profile'].items():
    print(f"  {f:12s} 钓鱼{p['phishing']:.3f} / 非钓{p['non_phishing']:.3f} 差{p['phishing']-p['non_phishing']:+.3f}")
