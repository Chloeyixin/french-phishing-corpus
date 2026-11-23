# -*- coding: utf-8 -*-
"""Figure 4: label-efficiency curves."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

res = json.load(open('/home/user/Doubao/chats/38455438158350082/label_efficiency.json', encoding='utf-8'))
B = [5, 10, 20, 40, 80, 160]
se = [res['seeded'][str(b)] for b in B]
su = [res['supervised'][str(b)] for b in B]
se_m = np.array([v[0] for v in se]); se_s = np.array([v[1] for v in se])
su_m = np.array([v[0] for v in su]); su_s = np.array([v[1] for v in su])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.linewidth': 0.8})
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=300)

ax.errorbar(B, se_m, yerr=se_s, marker='o', ms=5.5, lw=1.6, capsize=3,
            color='#1f4e79', label='Seeded K-means (semi-supervised)')
ax.errorbar(B, su_m, yerr=su_s, marker='s', ms=5.0, lw=1.6, capsize=3,
            color='#b1461f', label='Linear SVM (supervised)')
ax.axhline(0.860, ls='--', lw=1.1, color='#555555')
ax.text(4.7, 0.872, 'full supervision: F1 = 0.860', ha='left', va='bottom', fontsize=8.2, color='#555555')
ax.axhline(0.327, ls=':', lw=1.1, color='#999999')
ax.text(4.7, 0.285, 'no supervision: F1 = 0.327', ha='left', va='bottom', fontsize=8.2, color='#777777')

ax.set_xscale('log')
ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
ax.set_xlim(4.2, 190)
ax.set_ylim(0.25, 0.95)
ax.set_xlabel('Labeled messages per class')
ax.set_ylabel('F1 (phishing class, mean ± SD)')
ax.grid(True, ls=':', lw=0.7, color='#cccccc')
ax.legend(loc='lower right', frameon=False, fontsize=8.6, handlelength=2.6)
fig.tight_layout()
fig.savefig('/home/user/Doubao/chats/38455438158350082/fig_label_efficiency.png',
            bbox_inches='tight', facecolor='white')
print('saved')
