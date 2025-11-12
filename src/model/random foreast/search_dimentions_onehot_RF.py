"""
One-hot CSV without leakage: GroupKFold + RF regression
"""

import re, ast, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

CSV_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\05.Onehot\BB_ER&Score_onehot_encoded.csv"

# ---------- 读取 ----------
df = pd.read_csv(CSV_PATH)

# ---------- 剔除泄漏列 ----------
leak_patterns = [r"R\d+_frequency", r"ER$", r"PC\d+"]   # 正则列表
for patt in leak_patterns:
    df = df.drop(columns=[c for c in df.columns if re.match(patt, c)])

# ---------- 处理 onehot ----------
# 情况 A: 已经有列 0,1,2,...159 -> 直接使用
onehot_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
if len(onehot_cols) > 0:
    feature_cols = onehot_cols
else:
    # 情况 B: 有一列 'onehot' 字符串，需要拆开
    def parse_vec(s):
        arr = np.fromstring(s.strip("[]"), sep=' ', dtype=np.int8)
        return arr
    onehot_arr = np.vstack(df["onehot"].apply(parse_vec).values)
    onehot_df  = pd.DataFrame(onehot_arr,
                              columns=[f"oh_{i}" for i in range(onehot_arr.shape[1])])
    df = pd.concat([df.reset_index(drop=True), onehot_df], axis=1)
    feature_cols = onehot_df.columns
    df = df.drop(columns=["onehot"])   # 删原字符串列

# -------- 标签 & 分组 ----------
X = df[feature_cols].astype(np.int8).values
y = df["score"].values
groups = df["peptide_sequence"].values   # 保证同一肽只落一个 fold

print("样本:", len(y), "  特征维度:", X.shape[1])

# -------- 评估指标 --------
def rmse(y_t, y_p): return np.sqrt(mean_squared_error(y_t, y_p))
scoring = {"RMSE": make_scorer(rmse, greater_is_better=False),
           "R2": "r2"}

# -------- 模型 & GroupKFold --------
rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
gkf = GroupKFold(n_splits=5)

cv_res = cross_validate(rf, X, y, cv=gkf,
                        groups=groups, scoring=scoring, n_jobs=-1)

rmse_vals = -cv_res["test_RMSE"]
r2_vals   =  cv_res["test_R2"]

print("\n===== 5-fold (group) =====")
print("RMSE 每折:", np.round(rmse_vals, 4))
print("R²   每折:", np.round(r2_vals,   4))
print(f"平均 RMSE = {rmse_vals.mean():.4f} ± {rmse_vals.std():.4f}")
print(f"平均 R²   = {r2_vals.mean():.4f} ± {r2_vals.std():.4f}")

# -------- Baseline --------
y_mean = np.full_like(y, y.mean())
baseline_rmse = rmse(y, y_mean)
baseline_r2   = r2_score(y, y_mean)
print("\nBaseline RMSE =", round(baseline_rmse,4),
      "  R² =", round(baseline_r2,4))
