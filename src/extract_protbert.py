"""
extract_protbert.py
Author: Yuhao Rao
Description: Extract ProtBERT CLS embeddings for peptide sequences.
"""

import os, math, torch, argparse, pandas as pd, numpy as np, tqdm
from transformers import BertTokenizerFast, BertModel

# ---------- 参数 ----------
CSV_PATH  = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_minCount1_R1-4_ER&Score.csv"
OUT_DIR   = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures"
MODEL_ID  = "Rostlab/prot_bert"
BATCH     = 16          # 如显存不足可降到 8 或 4

# ---------- 准备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading ProtBERT ({MODEL_ID}) on {device} …")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID, do_lower_case=False)
model = (
    BertModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    .eval()
    .to(device)
)

# ---------- 读入数据 ----------
df = pd.read_csv(CSV_PATH)
if "peptide_sequence" not in df.columns:
    raise KeyError("列 'peptide_sequence' 未找到！")

seqs = df["peptide_sequence"].astype(str).tolist()
n_seq = len(seqs)
print(f"Total sequences: {n_seq}")

# ---------- 生成空格分隔序列 ----------
def add_spaces(seq: str) -> str:
    return " ".join(seq.strip())

spaced_seqs = [add_spaces(s) for s in seqs]

# ---------- 批量嵌入 ----------
all_vecs = np.empty((n_seq, 1024), dtype=np.float16)  # 预分配

with torch.no_grad():
    for i in tqdm.tqdm(range(0, n_seq, BATCH)):
        batch_seqs = spaced_seqs[i : i + BATCH]
        toks = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        reps = model(**toks).last_hidden_state[:, 0]  # CLS token
        # 转到 CPU / FP16
        all_vecs[i : i + reps.size(0)] = reps.cpu().numpy().astype(np.float16)

# ---------- 保存 ----------
base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
npy_path  = os.path.join(OUT_DIR, f"{base_name}_protBERT.npy")
pq_path   = os.path.join(OUT_DIR, f"{base_name}_protBERT.parquet")

np.save(npy_path, all_vecs)
print(f"NumPy saved to: {npy_path}")

# 保存为 Parquet，方便与 Pandas 联用
emb_df = pd.DataFrame(all_vecs.astype(np.float32))
emb_df.columns = [f"protBERT_{i}" for i in range(1024)]
emb_df.insert(0, "peptide_sequence", df["peptide_sequence"])
emb_df.to_parquet(pq_path, index=False)
print(f"Parquet saved to: {pq_path}")

print("Done!")
