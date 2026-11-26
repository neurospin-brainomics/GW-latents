import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import umap
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

random.seed(42)
FASTA_DIR  = '/home/tl288429/fasta'
N_SUBJECTS = 40
N_GW       = 200   # réduit pour que le GW soit rapide
N_VIZ      = 200   # même points pour cohérence transport / visu
N_PCA      = 64
N_FRAMES   = 12    # étapes de la déformation

# ── Données ───────────────────────────────────────────────────────────────────

def load_fasta(fasta_dir, chunk_size=6000):
    sequences, genes, subjects = [], [], []
    for fname in sorted(os.listdir(fasta_dir)):
        if not fname.endswith('.fasta'):
            continue
        subject = fname.removesuffix('.fasta')
        for record in SeqIO.parse(os.path.join(fasta_dir, fname), 'fasta'):
            seq = ''.join(
                b if b in 'ATCG' else random.choice('ATCG')
                for b in str(record.seq).upper()
            )
            parts = record.description.split('|')
            gene = parts[2] if len(parts) > 2 else record.id
            for start in range(0, len(seq), chunk_size):
                chunk = seq[start:start + chunk_size]
                if len(chunk) >= 100:
                    sequences.append(chunk)
                    genes.append(gene)
                    subjects.append(subject)
    return sequences, np.array(genes), np.array(subjects)

sequences, genes, subjects = load_fasta(FASTA_DIR)
selected = sorted(np.unique(subjects))[:N_SUBJECTS]
mask = np.isin(subjects, selected)
sequences = [sequences[i] for i in np.where(mask)[0]]
genes     = genes[mask]
subjects  = subjects[mask]
print(f"{len(sequences)} séquences — {len(np.unique(genes))} gènes")

# ── Modèle ────────────────────────────────────────────────────────────────────

print("Chargement NT-500M...")
tokenizer = AutoTokenizer.from_pretrained(
    'InstaDeepAI/nucleotide-transformer-500m-1000g', trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    'InstaDeepAI/nucleotide-transformer-500m-1000g', trust_remote_code=True)
model = model.cuda().eval()

def embed(seqs, layer=-1):
    embeddings = []
    for i, seq in enumerate(seqs):
        tokens = tokenizer(seq, return_tensors='pt', truncation=True,
                           max_length=1000, return_attention_mask=True)
        tokens = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens, output_hidden_states=True).hidden_states[layer]
        attn = tokens['attention_mask'].unsqueeze(-1).float()
        emb = (hidden * attn).sum(1) / attn.sum(1).clamp(min=1)
        embeddings.append(emb.cpu())
        if i % 500 == 0:
            print(f'  {i}/{len(seqs)}')
    return torch.cat(embeddings).numpy()

print("Couche finale...")
E_final = embed(sequences, layer=-1)
print("Couche 6...")
E_mid   = embed(sequences, layer=6)

# ── Sous-échantillonnage stratifié ────────────────────────────────────────────

def stratified_sample(labels, n):
    rng = np.random.default_rng(42)
    unique = np.unique(labels)
    per_class = max(1, n // len(unique))
    idx = np.concatenate([
        rng.choice(np.where(labels == l)[0],
                   size=min(per_class, (labels == l).sum()),
                   replace=False)
        for l in unique
    ])
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))

idx = stratified_sample(genes, N_VIZ)
E_f = E_final[idx]
E_m = E_mid[idx]
g   = genes[idx]

# ── PCA ───────────────────────────────────────────────────────────────────────

pca_f = PCA(n_components=N_PCA, random_state=42).fit(E_final)
pca_m = PCA(n_components=N_PCA, random_state=42).fit(E_mid)
E_f_pca = pca_f.transform(E_f)
E_m_pca = pca_m.transform(E_m)

# ── Gromov-Wasserstein ────────────────────────────────────────────────────────

print("Gromov-Wasserstein...")
n = len(idx)
a = b = np.ones(n) / n

def normed_dist(X):
    C = euclidean_distances(X).astype(np.float64)
    return C / C.max()

C_f = normed_dist(E_f_pca)
C_m = normed_dist(E_m_pca)

gw_dist, log = ot.gromov.entropic_gromov_wasserstein2(
    C_f, C_m, a, b, epsilon=0.05, loss_fun='square_loss', log=True)
T = log['T']
print(f"Distance GW : {gw_dist:.6f}")

# Correspondances optimales : pour chaque point de E_f, son match dans E_m
matches = T.argmax(axis=1)

# ── Réductions 2D et 3D ───────────────────────────────────────────────────────

print("t-SNE 2D...")
# t-SNE joint sur les deux espaces pour une projection cohérente
all_pca = np.vstack([E_f_pca, E_m_pca])
all_2d  = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(all_pca)
coords_f_2d = all_2d[:n]
coords_m_2d = all_2d[n:]

print("t-SNE 3D...")
all_3d  = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000).fit_transform(all_pca)
coords_f_3d = all_3d[:n]
coords_m_3d = all_3d[n:]

# ── Couleurs ──────────────────────────────────────────────────────────────────

unique_genes = sorted(set(g))
cmap = matplotlib.colormaps['tab20'].resampled(len(unique_genes))
gene2color = {gene: cmap(i) for i, gene in enumerate(unique_genes)}
colors = np.array([gene2color[gene] for gene in g])

# ── GIF 2D ───────────────────────────────────────────────────────────────────

print("Génération GIF 2D...")

fig2d, ax2d = plt.subplots(figsize=(9, 7))

def frame_2d(t):
    ax2d.clear()
    alpha = t / (N_FRAMES - 1)  # 0 → 1

    # Interpolation linéaire entre coords_f_2d et coords_m_2d[matches]
    current = (1 - alpha) * coords_f_2d + alpha * coords_m_2d[matches]

    ax2d.scatter(current[:, 0], current[:, 1],
                 c=colors, s=25, alpha=0.8, linewidths=0)

    # Flèches de transport sur quelques points représentatifs
    sample = np.linspace(0, n-1, 30, dtype=int)
    for i in sample:
        src = coords_f_2d[i]
        dst = coords_m_2d[matches[i]]
        mid = (1 - alpha) * src + alpha * dst
        direction = dst - src
        ax2d.annotate('', xy=mid + 0.01 * direction,
                      xytext=mid - 0.01 * direction,
                      arrowprops=dict(arrowstyle='->', color='gray',
                                      alpha=0.3, lw=0.8))

    ax2d.set_title(
        f"Déformation GW — t-SNE 2D\n"
        f"α = {alpha:.2f}  (0 = couche finale, 1 = couche 6)\n"
        f"GW distance = {gw_dist:.4f}",
        fontsize=11, fontweight='bold')
    ax2d.set_xlabel('t-SNE 1'); ax2d.set_ylabel('t-SNE 2')
    ax2d.set_xlim(all_2d[:, 0].min() - 2, all_2d[:, 0].max() + 2)
    ax2d.set_ylim(all_2d[:, 1].min() - 2, all_2d[:, 1].max() + 2)

anim2d = animation.FuncAnimation(fig2d, frame_2d, frames=N_FRAMES, interval=400)
out2d = '/home/tl288429/gw_deformation_2d.gif'
anim2d.save(out2d, writer='pillow', dpi=120)
plt.close(fig2d)
print(f"GIF 2D → {out2d}")

# ── GIF 3D ───────────────────────────────────────────────────────────────────

print("Génération GIF 3D...")

fig3d = plt.figure(figsize=(10, 8))
ax3d  = fig3d.add_subplot(111, projection='3d')

def frame_3d(t):
    ax3d.clear()
    alpha = t / (N_FRAMES - 1)
    current = (1 - alpha) * coords_f_3d + alpha * coords_m_3d[matches]

    ax3d.scatter(current[:, 0], current[:, 1], current[:, 2],
                 c=colors, s=20, alpha=0.7, linewidths=0)

    # Rotation douce de la caméra
    ax3d.view_init(elev=20, azim=alpha * 360)

    ax3d.set_title(
        f"Déformation GW — t-SNE 3D\nα = {alpha:.2f}",
        fontsize=11, fontweight='bold')
    ax3d.set_xlabel('t-SNE 1'); ax3d.set_ylabel('t-SNE 2'); ax3d.set_zlabel('t-SNE 3')

anim3d = animation.FuncAnimation(fig3d, frame_3d, frames=N_FRAMES, interval=400)
out3d = '/home/tl288429/gw_deformation_3d.gif'
anim3d.save(out3d, writer='pillow', dpi=120)
plt.close(fig3d)
print(f"GIF 3D → {out3d}")

print("\nDone.")