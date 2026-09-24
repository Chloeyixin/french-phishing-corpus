# -*- coding: utf-8 -*-
"""监督基线对比：NB / SVM / LR / KNN × TF-IDF(+增强特征)，5折分层CV，与无监督同口径"""
import json, re
import numpy as np
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

XLSX = 'data/法语语料库_最终版_钓鱼邮件319条+垃圾短信1272条.xlsx'
STOP = set("""le la les un une des et ou mais donc or ni car à de du au aux en dans sur sous avec sans par pour que qui quoi dont où ne pas plus moins très tout tous toute toutes ce cet cette ces il elle ils elles on nous vous je tu mon ma mes ton ta tes son sa ses notre votre leur nos vos leur se si y en a est sont était étaient être avoir avait avaient fait faites faire peut peuvent doit doivent comme aussi ainsi alors lorsque quand après avant entre contre chez vers parmi depuis pendant tant selon""".split())
URL_RE = re.compile(r'https?://|www\.|\b[a-z0-9-]+\.(com|net|org|fr|eu|io|ly|xyz|top|club|site|info)\b', re.I)
SHORT_RE = re.compile(r'\b(bit\.ly|t\.co|goo\.gl|tinyurl|shorturl|ow\.ly|is\.gd|buff\.ly|cutt\.ly)\b|https?://[^\s]{1,12}\b', re.I)
MONEY_RE = re.compile(r'[€$£]|\b\d[\d\s.,]{2,}\s?(euros?|€|dollars?)\b', re.I)
BANK = re.compile(r'\b(banque|compte|relevé|identifiant|identif|code|mot de passe|carte|crédit|virement|transaction|sécurit|paypal|amazon|dhl|la poste|imp[ôô]ts|police|gendarmerie|arcep|free|orange|sosh|bouygues)\b', re.I)
THREAT = re.compile(r'\b(p[ée]nalit[ée]|amende|poursuites|pirat[ée]|ransomware|webcam|chantage|dossier|proc[èe]s|prison|virus|infection)\b', re.I)

def clean(t):
    t = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|\b\d[\d\s\-.()/]*\d', ' ', t.lower())
    t = re.sub(r'[^\w\sàâäéèêëîïôöùûüçœ]', ' ', t, flags=re.UNICODE)
    return ' '.join(w for w in t.split() if w not in STOP and len(w) > 2)

def feats(t):
    tl = t.lower()
    return [1 if URL_RE.search(tl) else 0, 1 if SHORT_RE.search(tl) else 0,
            1 if MONEY_RE.search(tl) else 0, 1 if BANK.search(tl) else 0,
            1 if THREAT.search(tl) else 0, len(t)]

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ph, sm = [], []
ws = wb['钓鱼邮件语料']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    ph.append(str(row[1]))
ws2 = wb['垃圾短信数据']
for i, row in enumerate(ws2.iter_rows(values_only=True)):
    if i == 0 or not row[0] or not row[1]: continue
    sm.append(str(row[1]))
wb.close()

docs = ph + sm
y = np.array([1]*len(ph) + [0]*len(sm))
F = np.array([feats(t) for t in docs], dtype=float); F[:, -1] /= 400.0
vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
T = vec.fit_transform([clean(t) for t in docs])
X_tfidf = T
X_mix = hstack([T, csr_matrix(F)]).tocsr()

MODELS = {
    '朴素贝叶斯(MNB)': MultinomialNB(),
    'SVM(LinearSVC)': LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),
    '逻辑回归(LR)': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5),
}
FEATS = {'纯TF-IDF': X_tfidf, 'TF-IDF+钓鱼特征': X_mix}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

out = {'supervised': [], 'unsupervised_ref': {'baseline_k2_f1': 0.1241, 'feat_k3_f1': 0.32, 'seeded_k3_f1': 0.3861}}
for fname, X in FEATS.items():
    for mname, model in MODELS.items():
        p, r, f1, acc = [], [], [], []
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            pr, rc, f, _ = precision_recall_fscore_support(y[te], pred, average='binary', pos_label=1, zero_division=0)
            p.append(pr); r.append(rc); f1.append(f); acc.append(accuracy_score(y[te], pred))
        out['supervised'].append({
            'features': fname, 'model': mname,
            'precision': round(float(np.mean(p)),4), 'recall': round(float(np.mean(r)),4),
            'f1': round(float(np.mean(f)),4), 'accuracy': round(float(np.mean(acc)),4),
            'p_std': round(float(np.std(p)),4), 'r_std': round(float(np.std(r)),4), 'f1_std': round(float(np.std(f)),4)})
        print(f"{fname:16s} | {mname:18s} | P={np.mean(p):.3f} R={np.mean(r):.3f} F1={np.mean(f):.3f} Acc={np.mean(acc):.3f}")

# 最佳组合的混淆矩阵（一折示例，用于论文）
best = max(out['supervised'], key=lambda x: x['f1'])
X = FEATS[best['features']]
model = MODELS[best['model']]
for tr, te in skf.split(X, y):
    model.fit(X[tr], y[tr]); pred = model.predict(X[te])
    out['best_confusion'] = {'model': best['model'], 'features': best['features'],
                             'matrix': confusion_matrix(y[te], pred).tolist()}
    break
json.dump(out, open('supervised_result.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved supervised_result.json')
