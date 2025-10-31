# 文件名：bace_tsne_local_bert_robust.py
# 用法：
#   python bace_tsne_local_bert_robust.py --csv_path ../property_data/bace/raw/bace.csv --max_n 1200 --perplexity 35
# 依赖：
#   rdkit-pypi scikit-learn matplotlib transformers tqdm numpy pandas umap-learn

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


# ---------------------------
# 参数
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="BACE t-SNE (local CSV): graph-only vs text-only vs graph+text (plus separated view)")
    default_csv = os.path.join(os.path.dirname(__file__), "..", "property_data", "bace", "raw", "bace.csv")
    ap.add_argument("--csv_path", type=str, default=default_csv, help="本地 bace.csv 路径")
    ap.add_argument("--max_n", type=int, default=1200, help="最多使用的样本数")
    ap.add_argument("--perplexity", type=float, default=35.0, help="t-SNE perplexity")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--save_path", type=str, default="bace_tsne_plus.png", help="输出图片文件名")
    ap.add_argument("--bert_model", type=str, default="distilbert-base-uncased", help="文本编码器")
    ap.add_argument("--clip_abs", type=float, default=1e6, help="清洗时的绝对裁剪上限")
    return ap.parse_args()


# ---------------------------
# 工具
# ---------------------------
def sanitize(X, clip_abs=1e6):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs is not None and clip_abs > 0:
        X = np.clip(X, -clip_abs, clip_abs)
    return X


def standardize(X, eps=1e-9, clip_abs=1e6):
    X = sanitize(X, clip_abs=clip_abs)
    mu = np.nanmean(X, axis=0, keepdims=True)
    Xc = X - mu
    sd = np.nanstd(Xc, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return Xc / sd


def report_nonfinite(name, X):
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    print(f"[CHECK] {name}: shape={X.shape}, NaN={int(n_nan)}, Inf={int(n_inf)}")


# ---------------------------
# 读取数据
# ---------------------------
def load_bace_local(csv_path, max_n=None):
    df_raw = pd.read_csv(csv_path)
    smiles_col_candidates = ["smiles", "SMILES", "mol", "molecule"]
    label_col_candidates = ["label", "Label", "labels", "y", "target", "targets", "Class", "class"]

    smiles_col = next((c for c in smiles_col_candidates if c in df_raw.columns), None)
    label_col = next((c for c in label_col_candidates if c in df_raw.columns), None)

    if smiles_col is None or label_col is None:
        raise KeyError(f"无法在 {csv_path} 找到必要列。")

    df = df_raw[[smiles_col, label_col]].rename(columns={smiles_col: "smiles", label_col: "label"})
    df["label"] = df["label"].map(lambda x: 1 if str(x).lower() in ["1", "active", "pos", "positive", "true", "t"] else 0)
    df = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)

    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=0).reset_index(drop=True)
    return df


# ---------------------------
# 模板生成
# ---------------------------
BACE_TEMPLATES = [
    "BACE1 is an aspartic-acid protease related to AD. Can this molecule bind to BACE1?",
    "The assay tests whether the molecule can bind to the BACE1 protein. Is this molecule effective to the assay?",
    "Predict whether this molecule shows activity in the BACE1 binding assay. Answer Yes or No.",
    "BACE1 cleaves APP and is a target in Alzheimer's disease. Is the molecule active in the BACE1 assay?",
    "Determine if this molecule inhibits beta-secretase 1 (BACE1). Yes or No?",
]
def build_instructions(n, seed=0):
    random.seed(seed)
    return [random.choice(BACE_TEMPLATES) for _ in range(n)]


# ---------------------------
# SMILES & 特征
# ---------------------------
def filter_valid_smiles(smiles_list, labels, instructions):
    from rdkit import Chem
    mask = [Chem.MolFromSmiles(s) is not None for s in smiles_list]
    mask = np.array(mask, dtype=bool)
    return (
        [s for s, ok in zip(smiles_list, mask) if ok],
        labels[mask],
        [t for t, ok in zip(instructions, mask) if ok],
    )


def morgan_fp(smiles_list, radius=2, nbits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        fps.append(arr.astype(np.float32))
    return np.stack(fps)


def encode_text_bert(texts, model_name="distilbert-base-uncased", batch_size=32, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text (BERT)"):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            h = enc(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outs.append(pooled.cpu().numpy())
    return np.vstack(outs)


# ---------------------------
# 降维
# ---------------------------
def pca_reduce(X, out_dim=128, random_state=0):
    from sklearn.decomposition import PCA
    if X.shape[1] <= out_dim:
        return X
    pca = PCA(n_components=out_dim, random_state=random_state)
    return pca.fit_transform(X)


def run_tsne(X, perplexity=35.0, random_state=0):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    return tsne.fit_transform(X)


# ---------------------------
# 新增：完全分开的图（UMAP + supervision）
# ---------------------------
def run_umap_supervised(X, labels, random_state=0):
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        spread=2.0,
        target_metric="categorical",
        target_weight=0.9,
        random_state=random_state
    )
    Z = reducer.fit_transform(X, y=labels)
    return Z


# ---------------------------
# 绘图
# ---------------------------
def plot_four(Z_graph, Z_text, Z_fusion, Z_umap, labels, save_path="bace_tsne_plus.png"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 16))

    plots = [
        ("BACE • Graph-only (Morgan)", Z_graph),
        ("BACE • Text-only (DistilBERT)", Z_text),
        ("BACE • Graph+Text (Late fusion)", Z_fusion),
        ("BACE • Graph+Text (Supervised UMAP, fully separated)", Z_umap)
    ]

    for i, (title, Z) in enumerate(plots, 1):
        ax = fig.add_subplot(4, 1, i)
        ax.scatter(Z[:, 0], Z[:, 1], s=10, c=labels)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved to {save_path}")
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------
# 主流程
# ---------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    df = load_bace_local(args.csv_path, max_n=args.max_n)
    smiles = df["smiles"].tolist()
    labels = df["label"].astype(int).to_numpy()
    instructions = build_instructions(len(df), seed=args.seed)

    smiles, labels, instructions = filter_valid_smiles(smiles, labels, instructions)

    # Graph-only
    print("==> Computing Morgan fingerprints (graph-only)...")
    X_graph = morgan_fp(smiles, radius=2, nbits=2048)
    X_graph_std = standardize(X_graph, clip_abs=args.clip_abs)
    Z_graph = run_tsne(pca_reduce(X_graph_std, 128), perplexity=args.perplexity, random_state=args.seed)

    # Text-only
    print("==> Encoding instructions (text-only)...")
    X_text = encode_text_bert(instructions, model_name=args.bert_model)
    X_text_std = standardize(X_text)
    Z_text = run_tsne(pca_reduce(X_text_std, 128), perplexity=args.perplexity, random_state=args.seed)

    # Late fusion
    print("==> Late fusion (graph + text)...")
    Xg = pca_reduce(X_graph_std, 128)
    Xt = pca_reduce(X_text_std, 128)
    X_fusion = standardize(np.concatenate([Xg, Xt], axis=1))
    Z_fusion = run_tsne(X_fusion, perplexity=args.perplexity, random_state=args.seed)

    # New: supervised UMAP for clear separation
    print("==> Building supervised UMAP (clearly separated)...")
    Z_umap = run_umap_supervised(X_fusion, labels, random_state=args.seed)

    # 绘制四个图
    plot_four(Z_graph, Z_text, Z_fusion, Z_umap, labels, save_path=args.save_path)


if __name__ == "__main__":
    main()
