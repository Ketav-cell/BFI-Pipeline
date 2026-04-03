import numpy as np, torch, torch.nn as nn, mne, json, warnings
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.signal import welch
from scipy.stats import entropy
warnings.filterwarnings('ignore')
PATH = Path('physionet.org/files/chbmit/1.0.0')
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

class BiLSTM(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(d, 64, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])

def feats(data, fs):
    sig = np.mean(data, axis=0)
    f, p = welch(sig, fs, nperseg=min(256, len(sig) // 2))
    tot = np.sum(p) + 1e-10
    return np.array([np.sum(p[(f >= l) & (f < h)]) / tot for l, h in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]] + [entropy(p / tot + 1e-10), np.std(sig)], dtype=np.float32)

def load(pdir):
    sf = pdir / f'{pdir.name}-summary.txt'
    if not sf.exists():
        return (None, None)
    sz = {}
    with open(sf) as f:
        for ln in f:
            if 'File Name:' in ln:
                cur = ln.split(':')[1].strip()
                sz[cur] = []
            elif 'Seizure' in ln and 'Start' in ln:
                try:
                    sz[cur].append(int(ln.split(':')[1].split()[0]))
                except:
                    pass
    X, y = ([], [])
    for edf in pdir.glob('*.edf'):
        if edf.name not in sz:
            continue
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
            raw.filter(0.5, 45, verbose=False)
            fs, d = (raw.info['sfreq'], raw.get_data())
            win = int(30 * fs)
            for i in range(0, d.shape[1] - win, win // 2):
                t = i / fs
                lab = any((0 < s - t <= 3600 for s in sz[edf.name]))
                ft = feats(d[:, i:i + win], fs)
                if not np.any(np.isnan(ft)):
                    X.append(ft)
                    y.append(int(lab))
        except:
            pass
    return (np.array(X), np.array(y)) if X else (None, None)
print('Loading...')
data = {}
for p in sorted(PATH.iterdir()):
    if not p.name.startswith('chb'):
        continue
    X, y = load(p)
    if X is not None and len(X) > 12:
        Xs = np.array([X[i:i + 12] for i in range(len(X) - 11)])
        ys = np.array([y[i + 11] for i in range(len(y) - 11)])
        data[p.name] = (Xs, ys)
        print(f'  {p.name}: {ys.sum()} pos')
pts = [p for p, (X, y) in data.items() if y.sum() > 0]
print(f'{len(pts)} patients with seizures\n')
res = []
for tp in pts:
    print(f'{tp}...', end=' ')
    trX = np.concatenate([data[p][0] for p in data if p != tp])
    trY = np.concatenate([data[p][1] for p in data if p != tp])
    teX, teY = data[tp]
    pos, neg = (np.where(trY == 1)[0], np.where(trY == 0)[0])
    idx = np.concatenate([pos, np.random.choice(neg, min(len(neg), len(pos) * 3), replace=False)])
    np.random.shuffle(idx)
    trX, trY = (trX[idx], trY[idx])
    tr = DataLoader(TensorDataset(torch.FloatTensor(trX), torch.FloatTensor(trY)), 32, shuffle=True)
    te = DataLoader(TensorDataset(torch.FloatTensor(teX), torch.FloatTensor(teY)), 32)
    m = BiLSTM(trX.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    for _ in range(30):
        m.train()
        for X, y in tr:
            opt.zero_grad()
            nn.BCELoss()(m(X.to(DEVICE)).squeeze(), y.to(DEVICE)).backward()
            opt.step()
    m.eval()
    pr, lb = ([], [])
    with torch.no_grad():
        for X, y in te:
            pr.extend(m(X.to(DEVICE)).squeeze().cpu().tolist())
            lb.extend(y.tolist())
    if len(set(lb)) > 1:
        auc = roc_auc_score(lb, pr)
        res.append(auc)
        print(f'AUC: {auc:.3f}')
    else:
        print('skip')
print(f'\nAUROC: {np.mean(res):.3f} +/- {np.std(res):.3f}')
Path('results').mkdir(exist_ok=True)
json.dump({'auroc': np.mean(res), 'std': np.std(res), 'n': len(res)}, open('results/bilstm_results.json', 'w'))