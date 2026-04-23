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
        self.df = df.copy()
        self.value_col = value_col
        self.beta_proxy_col = beta_proxy_col
        self.N = len(self.df)
        
        # 尺度の最大値を自動検知 (5件法 or 10件法などを想定)
        self.max_scale = int(self.df[self.value_col].max())
        self.max_scale = max(5, self.max_scale) # 最低でも5件法スケールを保証

        if self.N < 1000:
            warnings.warn(f"警告: サンプルサイズが小さい(N={self.N})ため、マクロな同調圧力の検証精度が低下する可能性があります。")

    def audit_rejected_anomalies(self, threshold_sigma=2.5, beta_threshold=0.75):
        """【処方箋1】棄却アノマリーの復権と質的検証"""
        mean_val = self.df[self.value_col].mean()
        std_val = self.df[self.value_col].std()
        
        # 統計的な外れ値（低評価側）または絶対的な最低評価(1)を特定
        outlier_condition = (self.df[self.value_col] < (mean_val - threshold_sigma * std_val)) | (self.df[self.value_col] == 1)
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
        # 値を整数に丸めて集計（1.5などの小数が混ざるのを防ぐ）
        rounded_vals = self.df[self.value_col].round().astype(int)
        value_counts = rounded_vals.value_counts(normalize=True).sort_index()
        
        # スケール(1〜max_scale)の確率分布を保証
        for i in range(1, self.max_scale + 1):
            if i not in value_counts:
                value_counts[i] = 1e-9 # ゼロ除算回避
        
        obs_p = value_counts.sort_index().values
        
        # 忖度先（Target）を高評価側（80%の位置）と仮定
        target_idx = int(np.ceil(self.max_scale * 0.8)) - 1
        target_idx = min(target_idx, len(obs_p) - 1)
        
        inferred_true_p = np.copy(obs_p)
        pull_factor = assumed_v2
        
        stolen_mass = inferred_true_p[target_idx] * pull_factor
        inferred_true_p[target_idx] -= stolen_mass
        
        # 奪われた質量を他の選択肢に均等配分
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
        if global_var == 0 or pd.isna(global_var):
             return {"status": "データの分散がゼロ（完全に同一の回答）または計算不可能です。完全な熱的死状態です。"}

        frozen_chunks = sum(1 for v in variances if pd.notna(v) and v < global_var * 0.2)
        frozen_ratio = frozen_chunks / num_chunks
        
        return {
            "Num_Ensembles": num_chunks,
            "Global_Variance": round(global_var, 4),
            "Frozen_Chunks_Ratio": round(frozen_ratio, 3),
            "Diagnosis": f"⚠️ {frozen_ratio*100:.1f}%のグループで分散が異常消失（熱的死）。忖度空間のリスク大。" if frozen_ratio > 0.1 else "✅ 健全な『ぶれ（カオス）』が維持されています。"
        }


# ==========================================
# 2. 自動クリーニング機能付き実行パイプライン
# ==========================================
def clean_and_run_audit(file_path, value_col, beta_proxy_col=None):
    """
    外部CSVを読み込み、NAや非数値を自動クリーニングしてから診断を実行する
    """
    print(f"\n{'='*60}\n📊 ファイル診断開始: {file_path}\n{'='*60}")
    try:
        # 文字コードエラーを回避するため utf-8-sig や shift_jis のフォールバックを設定可能
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='shift_jis') # 日本語環境のExcel対策
    except FileNotFoundError:
        print(f"❌ エラー: ファイル '{file_path}' が見つかりません。")
        return

    if value_col not in df.columns:
        print(f"❌ エラー: 評価カラム '{value_col}' がCSV内に存在しません。")
        print(f"利用可能なカラム: {list(df.columns)}")
        return

    original_len = len(df)

    # --- 🧹 堅牢なデータクリーニング処理 ---
    # 1. 評価カラムを強制的に数値化（'N/A', '無回答', ' ' 等は NaN になる）
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # 2. βカラムがある場合も同様に数値化
    if beta_proxy_col and beta_proxy_col in df.columns:
        df[beta_proxy_col] = pd.to_numeric(df[beta_proxy_col], errors='coerce')
        # 両方揃っている行のみ残す
        df = df.dropna(subset=[value_col, beta_proxy_col])
    else:
        # 評価カラムの欠損値のみ削除
        df = df.dropna(subset=[value_col])

    cleaned_len = len(df)
    dropped_len = original_len - cleaned_len

    print(f"[クリーニング結果]")
    print(f"  - 元のデータ件数: {original_len}件")
    print(f"  - 削除された無効データ: {dropped_len}件 (空行, NA, 文字列など)")
    print(f"  - 分析対象の有効件数: {cleaned_len}件\n")

    if cleaned_len < 10:
        print("❌ エラー: 有効なデータ件数が少なすぎるため、分析を中断します。")
        return

    print(f"[データ概要]")
    print(f"  - 尺度最大値: {int(df[value_col].max())}")
    print(f"  - 平均値: {df[value_col].mean():.3f} | 分散: {df[value_col].var():.3f}\n")

    # デバッガクラスへクリーンなデータを渡す
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
    print("-" * 60)


# ==========================================
# 3. テスト用「汚れたデータ」の生成関数
# ==========================================
def generate_dirty_test_data():
    """NAや文字列が混ざったテスト用CSVを生成する"""
    np.random.seed(42)
    os.makedirs("test_data", exist_ok=True)
    
    N = 2500
    # 正常なデータ（忖度で4に集中）
    vals = np.random.choice([3, 4, 5], size=N, p=[0.1, 0.8, 0.1]).astype(object)
    betas = np.random.uniform(0.1, 0.4, size=N).astype(object)
    
    # ノイズを混入させる
    noise_indices = np.random.choice(N, size=150, replace=False)
    for idx in noise_indices[:50]:
        vals[idx] = "無回答"  # 文字列ノイズ
    for idx in noise_indices[50:100]:
        vals[idx] = np.nan    # NAノイズ
    for idx in noise_indices[100:150]:
        betas[idx] = "N/A"    # βカラムのノイズ
        
    # マイノリティ（評価1）を追加
    minority_vals = np.ones(100)
    minority_betas = np.random.uniform(0.8, 1.0, size=100)
    
    df_dirty = pd.DataFrame({
        "survey_score": np.concatenate([vals, minority_vals]),
        "nlp_certainty": np.concatenate([betas, minority_betas])
    })
    
    # 意図的に空行を数行追加
    empty_df = pd.DataFrame([{"survey_score": np.nan, "nlp_certainty": np.nan} for _ in range(5)])
    df_dirty = pd.concat([df_dirty, empty_df], ignore_index=True)
    
    # シャッフルして保存
    df_dirty = df_dirty.sample(frac=1).reset_index(drop=True)
    file_path = "test_data/04_dirty_realworld_data.csv"
    df_dirty.to_csv(file_path, index=False)
    print(f"✅ テスト用『汚れたデータ』を生成しました: {file_path}")


# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    # 1. 汚れたテストデータの生成
    generate_dirty_test_data()
    
    # 2. 自動クリーニングと診断の実行
    # ここに外部の任意のCSVパスを指定できます
    target_csv = "test_data/04_dirty_realworld_data.csv"
    
    clean_and_run_audit(
        file_path=target_csv, 
        value_col="survey_score",      # アンケートの評価列名を指定
        beta_proxy_col="nlp_certainty" # 確信度の列名（なければ None でOK）
    )
