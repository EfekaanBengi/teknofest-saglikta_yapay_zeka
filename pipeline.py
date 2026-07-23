import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score, matthews_corrcoef, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.stats import ks_2samp
import lightgbm as lgb
import xgboost as xgb
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from config import CHARGE_MAP, HYDRO_MAP

warnings.filterwarnings("ignore")


def secure_mi_scorer(X, y):
    return mutual_info_classif(X, y, random_state=42)


def robust_oof_predict(pipeline, X, y, cv):
    oof_sum = np.zeros(len(y))
    counts = np.zeros(len(y))
    y_arr = np.asarray(y)
    for tr, te in cv.split(X, y_arr):
        pipeline.fit(X.iloc[tr].reset_index(drop=True), y_arr[tr])
        oof_sum[te] += pipeline.predict_proba(X.iloc[te].reset_index(drop=True))[:, 1]
        counts[te] += 1
    return oof_sum / np.maximum(counts, 1)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        self.encoders_ = {}
        for col in self.cat_cols:
            if col in X.columns:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                oe.fit(X[col].fillna("UNKNOWN").astype(str).values.reshape(-1, 1))
                self.encoders_[col] = oe
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, oe in self.encoders_.items():
            if col in X.columns:
                X[col] = oe.transform(X[col].fillna("UNKNOWN").astype(str).values.reshape(-1, 1)).ravel()
        return X


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, imputer, num_cols):
        self.imputer = imputer
        self.num_cols = num_cols

    def fit(self, X, y=None):
        self.valid_cols_ = [c for c in self.num_cols if c in X.columns]
        self.scaler_ = RobustScaler()
        if self.valid_cols_:
            self.scaler_.fit(self.imputer.fit_transform(X[self.valid_cols_]))
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.valid_cols_:
            present = [c for c in self.valid_cols_ if c in X.columns]
            X[present] = self.scaler_.transform(self.imputer.transform(X[present]))
        for col in [c for c in X.columns if c not in self.valid_cols_]:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
        return X


class DynamicVariantPipeline:
    def __init__(self, data_path, panel_name):
        self.data_path = data_path
        self.panel_name = panel_name
        self.df = pd.read_csv(data_path)
        self.output_dir = f"outputs/{panel_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.cat_cols, self.num_cols, self.bin_cols = [], [], []
        self.best_model_name = self.best_pipeline = self.best_oof_probs = self.best_preds = None
        self.best_threshold = 0.5
        self.mi_scores = None
        self.panel_results = {}
        self.all_models_results = []

    def execute(self):
        print(f"\n{'='*55}\n{self.panel_name}\n{'='*55}")
        self.phase_1_eda()
        self.phase_2_feature_engineering()
        self.phase_3_leakage_audit()
        self.phase_5_outliers_profiling()
        self.phase_7_feature_selection()
        self.phase_4_6_8_9_dynamic_engine()
        self.phase_10_shap()
        self.phase_11_error_analysis()
        return self.panel_results

    def _refresh_cols(self):
        self.cat_cols = [c for c in self.df.columns if c.startswith("CAT_")]
        self.bin_cols = [c for c in self.df.columns if c.endswith("_missing") or c == "aa_same"]
        self.num_cols = [c for c in self.df.columns if c not in self.cat_cols + self.bin_cols + ["Label"]]

    def _build_pipe(self, imputer, resampler, model):
        steps = [
            ("cat", CategoricalEncoder(self.cat_cols)),
            ("num", NumericalPreprocessor(imputer, self.num_cols)),
            ("sel", SelectPercentile(score_func=secure_mi_scorer, percentile=30 if self.panel_name in ["PAH", "CFTR"] else 50)),
        ]
        if resampler is not None:
            steps.append(("res", resampler))
        steps.append(("clf", model))
        return ImbPipeline(steps)

    def phase_1_eda(self):
        print("--- Phase 1: EDA ---")
        df = self.df
        print(f"Shape: {df.shape} | Patojenik: {(df['Label']==1).sum()} | Benign: {(df['Label']==0).sum()}")
        prefixes = ["AL_", "CAT_", "EK_", "AA_"]
        print({p: len([c for c in df.columns if c.startswith(p)]) for p in prefixes})

        missing = df.isnull().mean() * 100
        print(missing.sort_values(ascending=False).head(10))

        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
        plt.title(f"{self.panel_name} - Missing Data Map")
        plt.tight_layout(); plt.savefig(f"{self.output_dir}/phase1_missing_map.png"); plt.close()

        drop_cols = missing[missing > 85].index.tolist()
        for c in ["Variant_ID", "CAT_6"]:
            if c in df.columns and c not in drop_cols:
                drop_cols.append(c)
        self.df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        print(f"Silindi: {len(drop_cols)} kolon | Yeni shape: {self.df.shape}")

        sns.countplot(data=self.df, x="Label")
        plt.title(f"{self.panel_name} - Class Dist"); plt.savefig(f"{self.output_dir}/phase1_class_dist.png"); plt.close()

        ek_cols = [c for c in self.df.columns if c.startswith("EK_")]
        if ek_cols:
            self.df[ek_cols].hist(bins=30, figsize=(15, 10))
            plt.suptitle(f"{self.panel_name} - EK Distributions")
            plt.savefig(f"{self.output_dir}/phase1_ek_dist.png"); plt.close()
            sns.heatmap(self.df[ek_cols + ["Label"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
            plt.title("EK Correlation"); plt.savefig(f"{self.output_dir}/phase1_ek_corr.png"); plt.close()

        num_c = self.df.select_dtypes(include=np.number).columns.drop("Label", errors="ignore")
        top20 = self.df[list(num_c) + ["Label"]].corr()["Label"].drop("Label").abs().sort_values(ascending=False).head(20)
        print("Top10 corr:\n", top20.head(10))
        top20.plot(kind="barh", color="#e74c3c")
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phase1_top20_corr.png"); plt.close()

        X_pca = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(self.df[num_c]))
        comps = PCA(n_components=2, random_state=42).fit_transform(X_pca)
        sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=self.df["Label"], alpha=0.7)
        plt.title(f"{self.panel_name} - PCA 2D"); plt.savefig(f"{self.output_dir}/phase1_pca.png"); plt.close()

    def phase_2_feature_engineering(self):
        print("--- Phase 2: Feature Engineering ---")
        self.cat_cols = [c for c in self.df.columns if c.startswith("CAT_")]
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna("UNKNOWN").astype(str)

        if "AA_1" in self.df.columns and "AA_2" in self.df.columns:
            def delta(row, m):
                v1, v2 = m.get(str(row["AA_1"]).upper(), np.nan), m.get(str(row["AA_2"]).upper(), np.nan)
                return abs(v1 - v2) if not (pd.isna(v1) or pd.isna(v2)) else np.nan
            self.df["charge_delta"] = self.df.apply(lambda r: delta(r, CHARGE_MAP), axis=1)
            self.df["hydro_delta"] = self.df.apply(lambda r: delta(r, HYDRO_MAP), axis=1)
            self.df["aa_same"] = (self.df["AA_1"] == self.df["AA_2"]).astype(float)
            self.df.loc[self.df["AA_1"].isna() | self.df["AA_2"].isna(), "aa_same"] = np.nan
            self.df.drop(columns=["AA_1", "AA_2"], inplace=True)

        if "EK_1" in self.df.columns and "EK_2" in self.df.columns:
            self.df["EK_1_x_EK_2"] = self.df["EK_1"] * self.df["EK_2"]
            self.df["EK_1_plus_EK_2_avg"] = (self.df["EK_1"] + self.df["EK_2"]) / 2

        num_c = self.df.select_dtypes(include=np.number).columns.drop("Label", errors="ignore")
        added = 0
        for col in num_c:
            r = self.df[col].isnull().mean()
            if 0.20 <= r <= 0.85:
                self.df[f"{col}_missing"] = self.df[col].isnull().astype(int)
                added += 1
        print(f"MissingIndicator: {added} | Shape: {self.df.shape}")
        self._refresh_cols()

    def phase_3_leakage_audit(self):
        print("--- Phase 3: Leakage Audit ---")
        X = self.df.drop(columns=["Label"])
        y = self.df["Label"]

        X_mi = X.copy()
        for col in self.cat_cols:
            if col in X_mi.columns:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_mi[col] = oe.fit_transform(X_mi[col].fillna("UNKNOWN").astype(str).values.reshape(-1, 1)).ravel()
        X_mi = X_mi.fillna(X_mi.median(numeric_only=True))
        mi = pd.Series(mutual_info_classif(X_mi, y, random_state=42), index=X_mi.columns).sort_values(ascending=False)
        self.mi_scores = mi
        print("Top-20 MI:\n", mi.head(20))

        for col in mi.index:
            # Sızıntı kontrolü: Label'ı doğrudan veren kolonlar (EK_ çekirdek özellikleri hariç)
            if col != "Label" and mi[col] > 0.98 and not col.startswith("EK_"):
                print(f"LEAKAGE: {col} MI={mi[col]:.4f} -> silindi")
                self.df.drop(columns=[col], inplace=True)

        num_only = self.df.select_dtypes(include=np.number).drop(columns=["Label"], errors="ignore")
        Xl, Xr = SimpleImputer(strategy="median").fit_transform(num_only), num_only
        Xl = StandardScaler().fit_transform(Xl)
        Xt, Xte, yt, yte = train_test_split(Xl, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(Xt, yt)
        auc_leaky = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, y, test_size=0.2, random_state=42, stratify=y)
        imp2, sc2 = SimpleImputer(strategy="median"), StandardScaler()
        clf2 = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf2.fit(sc2.fit_transform(imp2.fit_transform(Xtr2)), ytr2)
        auc_strict = roc_auc_score(yte2, clf2.predict_proba(sc2.transform(imp2.transform(Xte2)))[:, 1])
        print(f"Leaky AUC: {auc_leaky:.4f} | Strict AUC: {auc_strict:.4f} | Fark: {auc_leaky-auc_strict:.4f}")
        self._refresh_cols()

    def phase_5_outliers_profiling(self):
        print("--- Phase 5: Outlier Profiling ---")
        ek_cols = [c for c in self.df.columns if c.startswith("EK_")]
        if ek_cols:
            z = (self.df[ek_cols] - self.df[ek_cols].mean()) / self.df[ek_cols].std()
            print("Outliers |Z|>3:\n", (z.abs() > 3).sum()[lambda s: s > 0])
        if "EK_9" in self.df.columns:
            print(f"EK_9 min={self.df['EK_9'].min()} max={self.df['EK_9'].max()}")

        from sklearn.pipeline import Pipeline as SkPipeline
        X, y = self.df.drop(columns=["Label"]), self.df["Label"]
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_num = X.select_dtypes(include=np.number)
        try:
            auc_r = roc_auc_score(y, cross_val_predict(
                SkPipeline([("i", SimpleImputer(strategy="median")), ("s", RobustScaler()),
                            ("c", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))]),
                X_num, y, cv=cv5, method="predict_proba")[:, 1])
            Xw = X_num.copy()
            for col in Xw.columns:
                v = Xw[col].dropna()
                if len(v): Xw[col] = Xw[col].clip(*np.nanpercentile(v, [1, 99]))
            auc_w = roc_auc_score(y, cross_val_predict(
                SkPipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                            ("c", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))]),
                Xw, y, cv=cv5, method="predict_proba")[:, 1])
            print(f"RobustScaler AUC: {auc_r:.4f} | Winsorization AUC: {auc_w:.4f} -> RobustScaler secildi")
        except Exception as e:
            print(f"Scaler karşılaştırma başarısız: {e}")

    def phase_7_feature_selection(self):
        print("--- Phase 7: Feature Selection ---")
        X = self.df.drop(columns=["Label"])
        X_num = X.select_dtypes(include=np.number)
        vt = VarianceThreshold(threshold=0.01)
        vt.fit(X_num.fillna(X_num.median()))
        drop_v = list(set(X_num.columns) - set(X_num.columns[vt.get_support()]))
        if drop_v:
            self.df.drop(columns=drop_v, inplace=True)
            print(f"Düşük varyans silindi: {len(drop_v)}")
        self._refresh_cols()

        if self.mi_scores is not None:
            ratio = 0.30 if self.panel_name in ["PAH", "CFTR"] else 0.50
            valid_mi = self.mi_scores[self.mi_scores.index.isin(self.df.columns)]
            selected_mi = set(valid_mi.head(int(len(valid_mi) * ratio)).index)
            print(f"MI seçimi: {len(selected_mi)} feature (%{int(ratio*100)})")

        X_lgb = X.copy()
        for col in self.cat_cols:
            if col in X_lgb.columns:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_lgb[col] = oe.fit_transform(X_lgb[col].fillna("UNKNOWN").astype(str).values.reshape(-1, 1)).ravel()
        X_lgb = X_lgb.fillna(X_lgb.median(numeric_only=True))
        lgb_m = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)
        lgb_m.fit(X_lgb, self.df["Label"])
        imp = pd.Series(lgb_m.feature_importances_, index=X_lgb.columns).sort_values(ascending=False)
        imp.head(30).plot(kind="bar", figsize=(10, 8))
        plt.title(f"{self.panel_name} - Top30 LGBM Importance")
        plt.tight_layout(); plt.savefig(f"{self.output_dir}/phase7_importance.png"); plt.close()
        if self.mi_scores is not None:
            overlap = selected_mi & set(imp.head(30).index)
            print(f"MI Kesisim LGBM-Top30: {len(overlap)} feature")

    def phase_4_6_8_9_dynamic_engine(self):
        print("--- Phase 4/6/8/9: Dynamic Engine ---")
        X, y = self.df.drop(columns=["Label"]), self.df["Label"]
        self._refresh_cols()
        self.cv = (RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
                   if self.panel_name == "CFTR"
                   else StratifiedKFold(n_splits=10, shuffle=True, random_state=42))

        # --- Imputer seçimi ---
        base_lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        if self.panel_name == "CFTR":
            best_imp = ("Median", SimpleImputer(strategy="median"))
        else:
            best_imp, best_auc_i = ("Median", SimpleImputer(strategy="median")), -1
            for name, imp in [("Median", SimpleImputer(strategy="median")),
                               ("KNN", KNNImputer(n_neighbors=5)),
                               ("Iterative", IterativeImputer(max_iter=5, n_nearest_features=20, random_state=42))]:
                try:
                    auc = roc_auc_score(y, robust_oof_predict(self._build_pipe(clone(imp), None, clone(base_lr)), X, y, self.cv))
                    print(f"  Imputer {name}: {auc:.4f}")
                    if auc > best_auc_i: best_auc_i, best_imp = auc, (name, imp)
                except Exception as e: print(f"  {name} hata: {e}")
            print(f"Seçilen: {best_imp[0]}")

        # --- Class imbalance ---
        base_auc = roc_auc_score(y, robust_oof_predict(
            self._build_pipe(clone(best_imp[1]), None, LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)), X, y, self.cv))
        base_preds = (robust_oof_predict(self._build_pipe(clone(best_imp[1]), None,
            LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)), X, y, self.cv) >= 0.5).astype(int)
        base_score = matthews_corrcoef(y, base_preds) + recall_score(y, base_preds)
        best_res_name, best_res_obj = "None", None

        if self.panel_name != "CFTR":
            for rn, ro in [("SMOTE", SMOTE(random_state=42)), ("BorderlineSMOTE", BorderlineSMOTE(random_state=42))]:
                try:
                    probs = robust_oof_predict(self._build_pipe(clone(best_imp[1]), clone(ro),
                        LogisticRegression(random_state=42, max_iter=1000)), X, y, self.cv)
                    preds = (probs >= 0.5).astype(int)
                    score = matthews_corrcoef(y, preds) + recall_score(y, preds)
                    print(f"  {rn}: AUC={roc_auc_score(y, probs):.4f} MCC+Rec={score:.4f}")
                    if score > base_score: base_score, best_res_name, best_res_obj = score, rn, ro
                except Exception as e: print(f"  {rn} hata: {e}")

        # SMOTE kalite testi
        if best_res_name != "None":
            try:
                X_enc = CategoricalEncoder(self.cat_cols).fit_transform(X)
                X_imp = NumericalPreprocessor(clone(best_imp[1]), self.num_cols).fit_transform(X_enc)
                sel = SelectPercentile(score_func=secure_mi_scorer, percentile=30 if self.panel_name in ["PAH","CFTR"] else 50)
                X_sel = sel.fit_transform(X_imp, y)
                X_res, _ = best_res_obj.fit_resample(X_sel, y)
                X_syn = X_res[len(X_sel):]
                fail = sum(1 for i in range(X_sel.shape[1]) if ks_2samp(X_sel[:, i], X_syn[:, i]).pvalue < 0.05)
                if fail / X_sel.shape[1] > 0.20:
                    print(f"SMOTE kalite dusuk: KS fail ratio: %{fail/X_sel.shape[1]*100:.1f} -> kaldirildi")
                    best_res_name, best_res_obj = "None", None
                else:
                    comps = PCA(n_components=2, random_state=42).fit_transform(X_sel)
                    syn_c = PCA(n_components=2, random_state=42).fit_transform(X_syn)
                    plt.scatter(comps[:, 0], comps[:, 1], alpha=0.5, label="Gerçek")
                    plt.scatter(syn_c[:, 0], syn_c[:, 1], alpha=0.5, label="Sentetik")
                    plt.legend(); plt.title(f"{self.panel_name} - SMOTE PCA")
                    plt.savefig(f"{self.output_dir}/phase6_smote_pca.png"); plt.close()
            except Exception as e:
                print(f"SMOTE kalite testi hata: {e} -> kaldirildi")
                best_res_name, best_res_obj = "None", None

        # --- Model yarışması ---
        models = [
            ("Logistic Regression", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
            ("Random Forest", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
        ]
        if self.panel_name != "CFTR":
            models += [
                ("LightGBM", lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)),
                ("XGBoost", xgb.XGBClassifier(scale_pos_weight=(len(y)-y.sum())/y.sum(), max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_estimators=150, random_state=42, eval_metric="logloss", verbosity=0)),
            ]
        else:
            models.append(("LightGBM", lgb.LGBMClassifier(class_weight="balanced", max_depth=2, reg_lambda=50.0, n_estimators=50, random_state=42, verbose=-1)))

        best_auc = -1
        for mod_name, model in models:
            try:
                pipe = self._build_pipe(clone(best_imp[1]), clone(best_res_obj) if best_res_obj else None, clone(model))
                probs = robust_oof_predict(pipe, X, y, self.cv)
                preds05 = (probs >= 0.5).astype(int)
                auc = roc_auc_score(y, probs)
                pr_auc = average_precision_score(y, probs)

                precs, recs, thrs = precision_recall_curve(y, probs)
                vi = np.where(recs >= 0.90)[0]
                thr = float(thrs[vi[np.argmax(precs[vi])]]) if len(vi) and vi[np.argmax(precs[vi])] < len(thrs) else 0.5
                opt_preds = (probs >= thr).astype(int)

                print(f"  [{mod_name}] AUC={auc:.4f} PR={pr_auc:.4f} MCC={matthews_corrcoef(y,opt_preds):.4f} Rec={recall_score(y,opt_preds):.4f} F1={f1_score(y,opt_preds):.4f}")
                self.all_models_results.append({"Panel": self.panel_name, "Model": mod_name, "AUC": round(auc,4), "PR-AUC": round(pr_auc,4),
                    "MCC": round(matthews_corrcoef(y,opt_preds),4), "Recall": round(recall_score(y,opt_preds),4), "Threshold": round(thr,4)})

                if auc > best_auc:
                    best_auc = auc
                    self.best_model_name = f"{best_imp[0]}_{best_res_name}_{mod_name}"
                    self.best_pipeline = self._build_pipe(clone(best_imp[1]), clone(best_res_obj) if best_res_obj else None, clone(model))
                    self.best_oof_probs = probs
                    self.best_threshold = thr
            except Exception as e:
                print(f"  [{mod_name}] hata: {e}")

        print(f"Kazanan: {self.best_model_name} (AUC={best_auc:.4f} thr={self.best_threshold:.4f})")
        self.best_preds = (self.best_oof_probs >= self.best_threshold).astype(int)

        # --- Phase 9 metrikler ---
        roc_auc = roc_auc_score(y, self.best_oof_probs)
        pr_auc = average_precision_score(y, self.best_oof_probs)
        mcc = matthews_corrcoef(y, self.best_preds)
        rec = recall_score(y, self.best_preds)
        tn, fp, fn, tp = confusion_matrix(y, self.best_preds).ravel()
        f1 = f1_score(y, self.best_preds)
        print(f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} MCC={mcc:.4f} Rec={rec:.4f} Spec={tn/(tn+fp):.4f} F1={f1:.4f}")

        sns.heatmap(np.array([[tn,fp],[fn,tp]]), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign","Pathogenic"], yticklabels=["Benign","Pathogenic"])
        plt.title(f"{self.panel_name} - Confusion Matrix"); plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phase9_cm.png"); plt.close()

        fpr, tpr, _ = roc_curve(y, self.best_oof_probs)
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}"); plt.plot([0,1],[0,1],"k--")
        plt.title(f"{self.panel_name} - ROC"); plt.legend(); plt.savefig(f"{self.output_dir}/phase9_roc.png"); plt.close()

        precs2, recs2, _ = precision_recall_curve(y, self.best_oof_probs)
        plt.plot(recs2, precs2, label=f"PR-AUC={pr_auc:.4f}"); plt.title(f"{self.panel_name} - PR")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
        plt.savefig(f"{self.output_dir}/phase9_pr.png"); plt.close()

        self.panel_results = {"Panel": self.panel_name, "Model": self.best_model_name,
            "ROC-AUC": round(roc_auc,4), "PR-AUC": round(pr_auc,4), "MCC": round(mcc,4),
            "Recall": round(rec,4), "Specificity": round(tn/(tn+fp),4), "F1": round(f1,4), "Threshold": round(self.best_threshold,4)}
        self.best_pipeline.fit(X, y)

    def phase_10_shap(self):
        print("--- Phase 10: SHAP ---")
        X, y = self.df.drop(columns=["Label"]), self.df["Label"]
        X_prep = X.copy()
        curr_cols = list(X_prep.columns)
        for name, step in self.best_pipeline.steps[:-1]:
            if name == "res" or not hasattr(step, "transform"): continue
            if name == "sel":
                mask = step.get_support()
                curr_cols = [curr_cols[i] for i, m in enumerate(mask) if m]
                X_prep = pd.DataFrame(step.transform(X_prep), columns=curr_cols, index=X_prep.index)
            else:
                out = step.transform(X_prep)
                X_prep = pd.DataFrame(out if not isinstance(out, pd.DataFrame) else out.values, columns=curr_cols, index=X_prep.index)

        clean = [str(c).replace("[","_").replace("]","_").replace("<","_") for c in curr_cols]
        X_df = pd.DataFrame(X_prep.values, columns=clean, index=X_prep.index)
        model = self.best_pipeline.named_steps["clf"]
        try:
            if isinstance(model, (RandomForestClassifier, lgb.LGBMClassifier, xgb.XGBClassifier)):
                exp = shap.TreeExplainer(model)
                sv = exp.shap_values(X_df)
                if isinstance(sv, list): sv = sv[1]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3: sv = sv[:,:,1]
            else:
                exp = shap.LinearExplainer(model, X_df)
                sv = exp.shap_values(X_df)
            shap.summary_plot(sv, X_df, show=False)
            plt.tight_layout(); plt.savefig(f"{self.output_dir}/phase10_shap_summary.png"); plt.close()
            y_s = pd.Series(y.values, index=X_df.index)
            fn_idx = X_df[(y_s==1) & (pd.Series(self.best_preds, index=X_df.index)==0)].index
            if len(fn_idx):
                ev = exp.expected_value[1] if isinstance(exp.expected_value, (list, np.ndarray)) and len(exp.expected_value)>1 else exp.expected_value
                pos = X_df.index.get_loc(fn_idx[0])
                shap.force_plot(ev, sv[pos], X_df.iloc[pos], matplotlib=True, show=False)
                plt.gcf().savefig(f"{self.output_dir}/phase10_shap_force_fn.png", bbox_inches="tight"); plt.close()
        except Exception as e:
            print(f"SHAP hata: {e}")

    def phase_11_error_analysis(self):
        print("--- Phase 11: Error Analysis ---")
        X, y = self.df.drop(columns=["Label"]), self.df["Label"]
        df = X.copy()
        df["Label"], df["Prob"], df["Pred"] = y.values, self.best_oof_probs, self.best_preds
        fn = df[(df["Label"]==1) & (df["Pred"]==0)]
        fp = df[(df["Label"]==0) & (df["Pred"]==1)]
        print(f"FN: {len(fn)} | FP: {len(fp)}")
        if not fn.empty or not fp.empty:
            if not fn.empty: sns.histplot(fn["Prob"], bins=10, color="red", alpha=0.5, label=f"FN(n={len(fn)})", kde=True)
            if not fp.empty: sns.histplot(fp["Prob"], bins=10, color="blue", alpha=0.5, label=f"FP(n={len(fp)})", kde=True)
            plt.title(f"{self.panel_name} - FN/FP Prob Dist"); plt.legend()
            plt.savefig(f"{self.output_dir}/phase11_error_dist.png"); plt.close()
        unc = df[(df["Prob"]>=0.40) & (df["Prob"]<=0.60)]
        n_p = (unc["Label"]==1).sum()
        print(f"Belirsiz bölge (0.4-0.6): {len(unc)} örnek | Patojenik: {n_p} (%{n_p/len(unc)*100:.1f})" if len(unc) else "Belirsiz bölge: 0 örnek")