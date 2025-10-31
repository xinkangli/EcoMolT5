# 文件名：bace_tsne_local_bert_robust.py
# 用法：
#   python bace_tsne_local_bert_robust.py --csv_path ../property_data/bace/raw/bace.csv --max_n 1200 --perplexity 35
# 依赖：rdkit-pypi scikit-learn matplotlib transformers tqdm numpy pandas

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
    ap = argparse.ArgumentParser(description="BACE t-SNE (local CSV): graph-only vs text-only vs graph+text (BERT, robust)")
    default_csv = os.path.join(os.path.dirname(__file__), "..", "property_data", "bace", "raw", "bace.csv")
    ap.add_argument("--csv_path", type=str, default=default_csv, help="本地 bace.csv 路径（相对 zuotu/ 或绝对路径）")
    ap.add_argument("--max_n", type=int, default=1200, help="最多使用的样本数（None=全部）")
    ap.add_argument("--perplexity", type=float, default=35.0, help="t-SNE perplexity")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--save_path", type=str, default="bace_tsne.png", help="输出图片文件名")
    ap.add_argument("--bert_model", type=str, default="distilbert-base-uncased", help="文本编码器（不需要 sentencepiece）")
    ap.add_argument("--clip_abs", type=float, default=1e6, help="清洗时对极端值的绝对裁剪上限")
    return ap.parse_args()

# ---------------------------
# 工具：清洗 / 标准化 / 检查
# ---------------------------
def sanitize(X, clip_abs=1e6):
    """把 NaN/Inf 替换为 0，并可选裁剪极端值"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs is not None and clip_abs > 0:
        X = np.clip(X, -clip_abs, clip_abs)
    return X

def standardize(X, eps=1e-9, clip_abs=1e6):
    """更稳健的标准化：避免除 0，并对输入先清洗"""
    X = sanitize(X, clip_abs=clip_abs)
    mu = np.nanmean(X, axis=0, keepdims=True)
    Xc = X - mu
    sd = np.nanstd(Xc, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)  # 避免除以极小数
    return Xc / sd

def report_nonfinite(name, X):
    n_nan  = np.isnan(X).sum()
    n_inf  = np.isinf(X).sum()
    print(f"[CHECK] {name}: shape={X.shape}, NaN={int(n_nan)}, Inf={int(n_inf)}")

# ---------------------------
# 读取本地 CSV
# ---------------------------
def load_bace_local(csv_path, max_n=None):
    df_raw = pd.read_csv(csv_path)
    smiles_col_candidates = ["smiles", "SMILES", "mol", "molecule"]
    label_col_candidates  = ["label", "Label", "labels", "y", "target", "targets", "Class", "class"]

    smiles_col = next((c for c in smiles_col_candidates if c in df_raw.columns), None)
    label_col  = next((c for c in label_col_candidates if c in df_raw.columns), None)

    if smiles_col is None or label_col is None:
        raise KeyError(
            f"无法在 {csv_path} 找到所需列；可用列：{list(df_raw.columns)}\n"
            f"需要包含 SMILES 列之一：{smiles_col_candidates}，以及标签列之一：{label_col_candidates}"
        )

    df = df_raw[[smiles_col, label_col]].rename(columns={smiles_col: "smiles", label_col: "label"})

    def coerce_label(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            v = v[0] if len(v) else None
        try:
            vv = float(v)
            if np.isnan(vv):
                return None
            return int(round(vv))
        except Exception:
            s = str(v).strip().lower()
            if s in ["active", "act", "pos", "positive", "1", "true", "t"]:
                return 1
            if s in ["inactive", "inact", "neg", "negative", "0", "false", "f"]:
                return 0
            return None

    df["label"] = df["label"].map(coerce_label)
    df = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)

    if (max_n is not None) and (len(df) > max_n):
        df = df.sample(n=max_n, random_state=0).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df

# ---------------------------
# 指令模板（避免全相同导致退化）
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
# 过滤无效 SMILES
# ---------------------------
def filter_valid_smiles(smiles_list, labels, instructions):
    from rdkit import Chem
    mask = []
    for s in smiles_list:
        ok = Chem.MolFromSmiles(s) is not None
        mask.append(ok)
    mask = np.array(mask, dtype=bool)
    smiles_f = [s for s, ok in zip(smiles_list, mask) if ok]
    labels_f = labels[mask]
    instr_f  = [t for t, ok in zip(instructions, mask) if ok]
    n_drop = int((~mask).sum())
    if n_drop > 0:
        print(f"[INFO] Dropped {n_drop} invalid SMILES (kept {len(smiles_f)}).")
    return smiles_f, labels_f, instr_f

# ---------------------------
# 图特征：Morgan 指纹
# ---------------------------
def morgan_fp(smiles_list, radius=2, nbits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
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

# ---------------------------
# 文本特征：DistilBERT（不依赖 sentencepiece）
# ---------------------------
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
            h = enc(**inputs).last_hidden_state           # [B, L, H]
            mask = inputs.attention_mask.unsqueeze(-1)     # [B, L, 1]
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean-pool
            outs.append(pooled.cpu().numpy())
    return np.vstack(outs)

# ---------------------------
# 降维 & 可视化
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

def plot_three(Z_graph, Z_text, Z_fusion, labels, save_path="bace_tsne.png"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 12))

    ax = fig.add_subplot(3,1,1)
    ax.scatter(Z_graph[:,0], Z_graph[:,1], s=10, c=labels)
    ax.set_title("BACE • Graph-only (Morgan)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    ax = fig.add_subplot(3,1,2)
    ax.scatter(Z_text[:,0], Z_text[:,1], s=10, c=labels)
    ax.set_title("BACE • Text-only (DistilBERT)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

    ax = fig.add_subplot(3,1,3)
    ax.scatter(Z_fusion[:,0], Z_fusion[:,1], s=10, c=labels)
    ax.set_title("BACE • Graph+Text (Late fusion)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")

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

    print(f"==> Loading local CSV: {args.csv_path}")
    df = load_bace_local(args.csv_path, max_n=args.max_n)
    print(f"[Data] samples={len(df)}, pos={int(df['label'].sum())}, neg={int((1-df['label']).sum())}")

    smiles = df["smiles"].tolist()
    labels = df["label"].astype(int).to_numpy()
    instructions = build_instructions(len(df), seed=args.seed)

    # 过滤无效 SMILES（避免全 0 指纹行）
    smiles, labels, instructions = filter_valid_smiles(smiles, labels, instructions)

    # Graph-only
    print("==> Computing Morgan fingerprints (graph-only)...")
    X_graph = morgan_fp(smiles, radius=2, nbits=2048)
    report_nonfinite("X_graph (raw)", X_graph)
    X_graph = sanitize(X_graph, clip_abs=args.clip_abs)
    X_graph_std = standardize(X_graph, clip_abs=args.clip_abs)
    X_graph_red = pca_reduce(X_graph_std, out_dim=128, random_state=args.seed)
    Z_graph = run_tsne(X_graph_red, perplexity=args.perplexity, random_state=args.seed)

    # Text-only (DistilBERT)
    print("==> Encoding instructions with DistilBERT (text-only)...")
    X_text = encode_text_bert(instructions, model_name=args.bert_model)
    report_nonfinite("X_text (raw)", X_text)
    X_text = sanitize(X_text, clip_abs=args.clip_abs)
    X_text_std = standardize(X_text, clip_abs=args.clip_abs)
    X_text_red = pca_reduce(X_text_std, out_dim=128, random_state=args.seed)
    Z_text = run_tsne(X_text_red, perplexity=args.perplexity, random_state=args.seed)

    # Graph + Text (late fusion)
    print("==> Building late-fusion features (graph+text)...")
    Xg = pca_reduce(standardize(sanitize(X_graph, clip_abs=args.clip_abs), clip_abs=args.clip_abs),
                    out_dim=128, random_state=args.seed)
    Xt = pca_reduce(standardize(sanitize(X_text,  clip_abs=args.clip_abs), clip_abs=args.clip_abs),
                    out_dim=128, random_state=args.seed)
    X_fusion = np.concatenate([Xg, Xt], axis=1)
    report_nonfinite("X_fusion (concat before std)", X_fusion)
    X_fusion = sanitize(X_fusion, clip_abs=args.clip_abs)
    X_fusion_std = standardize(X_fusion, clip_abs=args.clip_abs)
    Z_fusion = run_tsne(X_fusion_std, perplexity=args.perplexity, random_state=args.seed)

    # 可视化
    plot_three(Z_graph, Z_text, Z_fusion, labels, save_path=args.save_path)

if __name__ == "__main__":
    main()
