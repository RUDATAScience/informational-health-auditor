import numpy as np
import pandas as pd
from scipy.stats import entropy
import warnings
import os

# ==========================================
# 1. 情報の健康診断デバッガ (コアクラス)
# ==========================================
class InformationalHealthDebugger:
    def __init__(self, df, value_col, beta_proxy_col=None):
        self.df = df
        self.value_col = value_col
        self.beta_proxy_col = beta_proxy_col
        self.N = len(df)
        
        # 尺度の最大値を自動検知 (5件法 or 10件法)
        self.max_scale = int(self.df[self.value_col].max())
        if self.max_scale not in [5, 10]:
            # 5, 10以外のスケールでも動作するように柔軟に対応
            self.max_scale = max(5, self.max_scale)

        if self.N < 1000:
            warnings.warn(f"警告: サンプルサイズが小さい(N={self.N})ため、マクロな同調圧力の検証精度が低下する可能性があります。")

    def audit_rejected_anomalies(self, threshold_sigma=2.5, beta_threshold=0.75):
        """【処方箋1】棄却アノマリーの復権と質的検証"""
        mean_val = self.df[self.value_col].mean()
        std_val = self.df[self.value_col].std()
        
        # 統計的な外れ値（低評価側）または絶対的な最低評価を特定
        outlier_condition = self.df[self.value_col] < (mean_val - threshold_sigma * std_val)
        outlier_condition = outlier_condition | (self.df[self.value_col] == 1)
            
        anomalies = self.df[outlier_condition]
        
        if self.beta_proxy_col is None or self.beta_proxy_col not in self.df.columns:
            return {"status": "βのプロキシデータがないため、質的検証はスキップされました。", "anomaly_count": len(anomalies)}

        if len(anomalies) == 0:
            return {"status": "アノマリー（評価1や外れ値）が存在しません。完全な同調状態の可能性があります。"}

        # アノマリーのうち、確信度（β）が高い層を抽出
        high_beta_threshold = anomalies[self.beta_proxy_col].quantile(beta_threshold)
        high_beta_anomalies = anomalies[anomalies[self.beta_proxy_col] >= high_beta_threshold]
        
        risk_score = len(high_beta_anomalies) / len(anomalies)
        
        return {
            "Total_Anomalies_Flagged": len(anomalies),
            "High_Beta_Anomalies (真のSOS候補)": len(high_beta_anomalies),
            "Risk_Score (0-1)": round(risk_score, 3),
            "Diagnosis": "⚠️警告: 棄却データに高確信度のSOSが混入しています。" if risk_score >= 0.25 else "✅ アノマリーは単なるノイズの可能性が高いです。"
        }

    def bayesian_stress_test(self, assumed_v2=0.5):
        """【処方箋2】潜在バイアスのベイジアン・ストレステスト"""
        value_counts = self.df[self.value_col].value_counts(normalize=True).sort_index()
        
        # スケール(1〜max_scale)を保証
        for i in range(1, self.max_scale + 1):
            if i not in value_counts:
                value_counts[i] = 1e-9 # ゼロ除算回避
        
        obs_p = value_counts.sort_index().values
        
        # 忖度先（Target）を分布の最頻値（Mode）に近い高評価側と仮定
        # 5件法なら4、10件法なら8〜9付近を想定
        target_idx = int(np.ceil(self.max_scale * 0.8)) - 1
        
        inferred_true_p = np.copy(obs_p)
        pull_factor = assumed_v2
        
        stolen_mass = inferred_true_p[target_idx] * pull_factor
        inferred_true_p[target_idx] -= stolen_mass
        
        # 奪われた質量を他の選択肢（低評価側）に均等に再配分
        remaining_indices = [i for i in range(self.max_scale) if i != target_idx]
        for idx in remaining_indices:
            inferred_true_p[idx] += stolen_mass / len(remaining_indices)
            
        kl_div = entropy(obs_p, inferred_true_p)
        
        return {
            "Assumed_Sontaku_v2": assumed_v2,
            "KL_Divergence": round(kl_div, 4),
            "Danger_Level": "High" if kl_div > 0.1 else "Low",
            "Diagnosis": f"⚠️KL距離が {round(kl_div, 4)} 爆発。現在のP値は偽像の可能性が高いです。" if kl_div > 0.1 else "✅ 観測分布は比較的安定しています。"
        }

    def small_data_ensemble_audit(self, ensemble_size=50):
        """【処方箋3】スモールデータ・アンサンブルへの再分割"""
        shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        num_chunks = max(1, self.N // ensemble_size)
        
        chunks = np.array_split(shuffled_df, num_chunks)
        variances = [chunk[self.value_col].var() for chunk in chunks]
        
        global_var = self.df[self.value_col].var()
        frozen_chunks = sum(1 for v in variances if v < global_var * 0.2)
        frozen_ratio = frozen_chunks / num_chunks
        
        return {
            "Num_Ensembles": num_chunks,
            "Global_Variance": round(global_var, 4),
            "Frozen_Chunks_Ratio": round(frozen_ratio, 3),
            "Diagnosis": f"⚠️ {frozen_ratio*100:.1f}%のグループで分散が異常消失（熱的死）。忖度空間のリスク大。" if frozen_ratio > 0.1 else "✅ 健全な『ぶれ（カオス）』が維持されています。"
        }

# ==========================================
# 2. 自動パイプライン処理関数
# ==========================================
def run_audit_on_csv(file_path, value_col, beta_proxy_col=None):
    """CSVを読み込んで一括診断を実行する"""
    print(f"\n{'='*50}\n📊 ファイル診断開始: {file_path}\n{'='*50}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return

    if value_col not in df.columns:
        print(f"エラー: 評価カラム '{value_col}' がCSV内に存在しません。")
        return
        
    print(f"データ件数(N): {len(df)}件 | 尺度最大値: {df[value_col].max()}")
    print(f"平均値: {df[value_col].mean():.2f} | 分散: {df[value_col].var():.2f}\n")

    debugger = InformationalHealthDebugger(df, value_col=value_col, beta_proxy_col=beta_proxy_col)
    
    print("--- 【処方箋1】棄却アノマリーの監査 ---")
    for k, v in debugger.audit_rejected_anomalies().items():
        print(f"  {k}: {v}")
        
    print("\n--- 【処方箋2】潜在バイアスのベイジアン・ストレステスト ---")
    for k, v in debugger.bayesian_stress_test(assumed_v2=0.5).items():
        print(f"  {k}: {v}")
        
    print("\n--- 【処方箋3】スモールデータ・アンサンブル監査 ---")
    for k, v in debugger.small_data_ensemble_audit(ensemble_size=50).items():
        print(f"  {k}: {v}")
    print("-" * 50)

# ==========================================
# 3. テスト用CSVデータ生成関数 (事例作成)
# ==========================================
def generate_test_data():
    """様々なパターンのテスト用CSVを生成する"""
    np.random.seed(42)
    os.makedirs("test_data", exist_ok=True)
    
    # パターン1: 【健全なデータ (5件法)】 分散が保たれている
    N_healthy = 2500
    healthy_vals = np.random.choice([1, 2, 3, 4, 5], size=N_healthy, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    healthy_betas = np.random.uniform(0.1, 0.9, size=N_healthy)
    pd.DataFrame({"rating_5": healthy_vals, "nlp_beta": healthy_betas}).to_csv("test_data/01_healthy_5pt.csv", index=False)
    
    # パターン2: 【汚染データ (5件法)】 忖度で「4」に集中。評価1は少ないがβが極めて高い（専門家の叫び）
    N_contam = 3000
    majority_vals = np.random.choice([3, 4, 5], size=2850, p=[0.05, 0.85, 0.10])
    majority_betas = np.random.uniform(0.1, 0.3, size=2850) # マジョリティは低β
    minority_vals = np.ones(150)
    minority_betas = np.random.uniform(0.8, 1.0, size=150) # マイノリティは高β(警告)
    
    df_contam_5 = pd.DataFrame({
        "rating_5": np.concatenate([majority_vals, minority_vals]),
        "nlp_beta": np.concatenate([majority_betas, minority_betas])
    })
    df_contam_5.to_csv("test_data/02_contaminated_5pt.csv", index=False)

    # パターン3: 【汚染データ (10件法)】 忖度で「8」に集中。より複雑なNPS(ネットプロモータースコア)等でのエラーを想定
    N_contam_10 = 5000
    maj_10 = np.random.choice([7, 8, 9, 10], size=4800, p=[0.1, 0.7, 0.15, 0.05])
    maj_betas_10 = np.random.uniform(0.1, 0.4, size=4800)
    min_10 = np.random.choice([1, 2], size=200, p=[0.5, 0.5])
    min_betas_10 = np.random.uniform(0.7, 1.0, size=200)
    
    df_contam_10 = pd.DataFrame({
        "nps_score_10": np.concatenate([maj_10, min_10]),
        "nlp_beta": np.concatenate([maj_betas_10, min_betas_10])
    })
    df_contam_10.to_csv("test_data/03_contaminated_10pt.csv", index=False)

    print("✅ テスト用CSVデータを 'test_data/' フォルダに生成しました。")

# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    # 1. テストデータの生成
    generate_test_data()
    
    # 2. 自動診断の実行
    # ① 健全なデータ（問題が検出されないことを確認）
    run_audit_on_csv("test_data/01_healthy_5pt.csv", value_col="rating_5", beta_proxy_col="nlp_beta")
    
    # ② 忖度に汚染された5件法（異常が検出されることを確認）
    run_audit_on_csv("test_data/02_contaminated_5pt.csv", value_col="rating_5", beta_proxy_col="nlp_beta")
    
    # ③ 忖度に汚染された10件法（NPS調査等でエラーが出るか確認）
    run_audit_on_csv("test_data/03_contaminated_10pt.csv", value_col="nps_score_10", beta_proxy_col="nlp_beta")
