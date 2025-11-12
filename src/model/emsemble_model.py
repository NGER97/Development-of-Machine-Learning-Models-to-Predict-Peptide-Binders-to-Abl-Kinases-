# ================================================================
# Ensemble model selection (R²) for peptide protBERT embeddings
# Yuhao Rao — 2025-06-09
# ================================================================
import os, warnings, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, VotingRegressor, StackingRegressor
)
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# ---------- 路径 ----------
EMBED_PATH = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\03.GenerateFeatures\BB_ER&Score_protBERT.parquet"
RAW_CSV    = r"D:\Me\IMB\Data\Yuhao\NGS\Abl L1\02.DataPreprocessing\BB_add_pseudocount_remove_r5.csv"
MODEL_OUT  = r"D:\Me\IMB\Data\Yuhao\models"

# ---------- 读取数据 ----------
df = (pd.read_parquet(EMBED_PATH)
      if EMBED_PATH.endswith(".parquet") else pd.read_csv(EMBED_PATH))
if "score" not in df.columns:
    df = df.merge(pd.read_csv(RAW_CSV, usecols=["peptide_sequence", "score"]),
                  on="peptide_sequence", how="left")

feat_cols = [c for c in df.columns if c.startswith("protBERT_")]
X = df[feat_cols].astype(np.float32).values
y = df["score"].values.astype(np.float32)

# 留出 15 % 做最终测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# ---------- 定义基模型 ----------
base_models, param_spaces = {}, {}

# 1) 树系集成
base_models["rf"] = RandomForestRegressor(random_state=42, n_jobs=-1)
param_spaces["rf"] = {"rf__n_estimators":[300,500,700],
                      "rf__max_depth":[None,15,25]}

base_models["gbrt"] = GradientBoostingRegressor(random_state=42)
param_spaces["gbrt"] = {"gbrt__n_estimators":[300,500],
                        "gbrt__learning_rate":[0.03,0.05,0.1]}

base_models["etr"] = ExtraTreesRegressor(random_state=42, n_jobs=-1)
param_spaces["etr"] = {"etr__n_estimators":[300,600],
                       "etr__max_depth":[None,15,25]}

# 2) XGBoost / LightGBM / CatBoost（按需安装）
try:
    from xgboost import XGBRegressor
    base_models["xgb"] = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist")
    param_spaces["xgb"] = {"xgb__n_estimators":[400,700],
                           "xgb__max_depth":[4,6,8],
                           "xgb__learning_rate":[0.03,0.05,0.1]}
except ModuleNotFoundError:
    warnings.warn("xgboost 未安装，跳过 XGBRegressor")

try:
    from lightgbm import LGBMRegressor
    base_models["lgb"] = LGBMRegressor(random_state=42, n_jobs=-1)
    param_spaces["lgb"] = {"lgb__n_estimators":[400,700],
                           "lgb__num_leaves":[31,63,127],
                           "lgb__learning_rate":[0.03,0.05,0.1]}
except ModuleNotFoundError:
    warnings.warn("lightgbm 未安装，跳过 LGBMRegressor")

try:
    from catboost import CatBoostRegressor
    base_models["cat"] = CatBoostRegressor(
        verbose=0, random_seed=42, loss_function="RMSE")
    param_spaces["cat"] = {"cat__iterations":[600,800],
                           "cat__depth":[4,6,8],
                           "cat__learning_rate":[0.03,0.05,0.1]}
except ModuleNotFoundError:
    warnings.warn("catboost 未安装，跳过 CatBoostRegressor")

# 3) 线性 / 核方法 / MLP
base_models["enet"] = ElasticNet(random_state=42)
param_spaces["enet"] = {"enet__alpha":[0.01,0.1,1.0],
                        "enet__l1_ratio":[0.1,0.5,0.9]}

base_models["svr"] = SVR()
param_spaces["svr"] = {"svr__C":[1,10,100],
                       "svr__gamma":["scale","auto"]}

base_models["mlp"] = MLPRegressor(random_state=42, max_iter=1000)
param_spaces["mlp"] = {"mlp__hidden_layer_sizes":[(128,128),(256,128)],
                       "mlp__learning_rate_init":[0.001,0.0005]}

# ---------- 超参搜索 ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}

for name, model in base_models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),  # 树模型对尺度不敏感，但统一放进去更简洁
        (name, model)
    ])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_spaces[name],
        n_iter=12,
        scoring="r2",          # 使用 R²
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    print(f"[{name}] CV best R² = {search.best_score_:.4f}")

# ---------- 单模型测试集表现 ----------
def r2(model): return r2_score(y_test, model.predict(X_test))

single_best_name, single_best = max(
    best_models.items(), key=lambda kv: r2(kv[1])
)
print(f"\n单模型最佳  : {single_best_name}  R² = {r2(single_best):.4f}")

# ---------- Stacking & Voting ----------
stack = StackingRegressor(
    estimators=[(n,m) for n,m in best_models.items()],
    final_estimator=GradientBoostingRegressor(random_state=42),
    passthrough=True, n_jobs=-1
).fit(X_train, y_train)

vote = VotingRegressor(
    estimators=[(n,m) for n,m in best_models.items()],
    n_jobs=-1
).fit(X_train, y_train)

candidates = {**{single_best_name: single_best}, "stack": stack, "vote": vote}
best_name, best_model = max(candidates.items(), key=lambda kv: r2(kv[1]))
print(f"\n最终最佳模型: {best_name}  测试集 R² = {r2(best_model):.4f}")

# ---------- 全数据重训练 & 保存 ----------
best_model.fit(X, y)
os.makedirs(MODEL_OUT, exist_ok=True)
model_path = os.path.join(MODEL_OUT, f"{best_name}_regressor.joblib")
joblib.dump(best_model, model_path)
print(f"模型已保存至: {model_path}")
