import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
import warnings

class InformationalHealthDebugger:
    """
    大規模アンケートデータ（N>=2000）に潜む「情報の不健康状態（同調圧力や忖度）」を
    事後的に診断・監査するためのデバッガクラス。
    """
    def __init__(self, df, value_col, beta_proxy_col=None):
        """
        :param df: Pandas DataFrame (N>=2000を推奨)
        :param value_col: 評価値（1〜5のLikert尺度など）のカラム名
        :param beta_proxy_col: 確信度(β)の代替となるカラム名（例: 自由記述の文字数、NLPセンチメントスコア等）
        """
        self.df = df
        self.value_col = value_col
        self.beta_proxy_col = beta_proxy_col
        self.N = len(df)
        
        if self.N < 1000:
            warnings.warn("警告: サンプルサイズが小さいため、マクロな同調圧力の検証精度が低下する可能性があります。")

    def audit_rejected_anomalies(self, threshold_sigma=2.5, beta_threshold=0.75):
        """
        【処方箋1】棄却アノマリーの復権と質的検証
        外れ値として処理されがちなデータ（評価1など）の中に、高確信度（高β）の
        「真実の残骸（Fugitives）」がどの程度含まれているかを監査する。
        """
        mean_val = self.df[self.value_col].mean()
        std_val = self.df[self.value_col].std()
        
        # 統計的な外れ値（低評価側）を特定
        outlier_condition = self.df[self.value_col] < (mean_val - threshold_sigma * std_val)
        # Likert尺度の場合は、単純に最下位評価（1）をアノマリーとするアプローチも有効
        if mean_val > 3.5:
            outlier_condition = outlier_condition | (self.df[self.value_col] == 1)
            
        anomalies = self.df[outlier_condition]
        
        if self.beta_proxy_col is None:
            return {"status": "βのプロキシデータがないため、質的検証はスキップされました。", "anomaly_count": len(anomalies)}

        # アノマリーのうち、確信度（β）が高い層を抽出
        high_beta_anomalies = anomalies[anomalies[self.beta_proxy_col] >= anomalies[self.beta_proxy_col].quantile(beta_threshold)]
        
        risk_score = len(high_beta_anomalies) / max(len(anomalies), 1)
        
        return {
            "total_anomalies_flagged": len(anomalies),
            "high_beta_anomalies (真のSOS候補)": len(high_beta_anomalies),
            "risk_score (0-1)": round(risk_score, 3),
            "message": "外れ値の中に高確信度データが多く含まれる場合、安易な除去は情報の死を招きます。" if risk_score > 0.3 else "アノマリーは単なるノイズの可能性が高いです。"
        }

    def bayesian_stress_test(self, assumed_v2=0.5, target_value=4):
        """
        【処方箋2】潜在バイアスのベイジアン・ストレステスト
        観測された分布が「v2=0.5以上の忖度で歪められた偽像」であると仮定し、
        簡易的な逆問題計算で真の分布を推論。観測分布とのKLダイバージェンスを計算する。
        """
        # 観測された確率分布 (Observed P)
        value_counts = self.df[self.value_col].value_counts(normalize=True).sort_index()
        # 1〜5のスケールを保証
        for i in range(1, 6):
            if i not in value_counts:
                value_counts[i] = 1e-9 # ゼロ除算回避
        
        obs_p = value_counts.sort_index().values
        
        # 【簡易逆推論】v2の圧力でtarget_valueに集まっている分を、他の選択肢に再配分する
        # ※本来は事後分布からのMCMCサンプリング等が必要ですが、ここでは決定論的デバッグ手法として近似
        inferred_true_p = np.copy(obs_p)
        pull_factor = assumed_v2  # 忖度によって奪われた確率質量の割合
        
        target_idx = target_value - 1
        stolen_mass = inferred_true_p[target_idx] * pull_factor
        inferred_true_p[target_idx] -= stolen_mass
        
        # 奪われた質量を他の選択肢（特に低評価側）に均等に再配分（極端な最悪シナリオのストレステスト）
        remaining_indices = [i for i in range(5) if i != target_idx]
        for idx in remaining_indices:
            inferred_true_p[idx] += stolen_mass / len(remaining_indices)
            
        # KLダイバージェンスの計算 (Observed || Inferred True)
        kl_div = entropy(obs_p, inferred_true_p)
        
        return {
            "assumed_sontaku_v2": assumed_v2,
            "kl_divergence": round(kl_div, 4),
            "danger_level": "High" if kl_div > 0.1 else "Low",
            "message": f"KL距離が{round(kl_div, 4)}爆発しました。現在の有意差(P値)は偽像であるリスクが高いです。"
        }

    def small_data_ensemble_audit(self, ensemble_size=50):
        """
        【処方箋3】スモールデータ・アンサンブルへの再分割
        データを極小サブグループ（N<=50）にランダム分割し、グループごとの「分散」を計算。
        分散が異常に消失している（熱的死を起こしている）グループの割合を検知する。
        """
        # データをシャッフル
        shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        num_chunks = max(1, self.N // ensemble_size)
        
        chunks = np.array_split(shuffled_df, num_chunks)
        variances = [chunk[self.value_col].var() for chunk in chunks]
        
        # 分散の中央値と、極端に分散が低い（熱的死）チャンクの割合を計算
        median_variance = np.nanmedian(variances)
        
        # 分散が全体の分散の20%未満になっているチャンクを「凍結（熱的死）」とみなす
        global_var = self.df[self.value_col].var()
        frozen_chunks = sum(1 for v in variances if v < global_var * 0.2)
        frozen_ratio = frozen_chunks / num_chunks
        
        return {
            "num_ensembles": num_chunks,
            "median_ensemble_variance": round(median_variance, 4),
            "frozen_chunks_ratio": round(frozen_ratio, 3),
            "message": f"全{num_chunks}グループ中、{frozen_ratio*100:.1f}%のグループで分散が異常消失(熱的死)しています。忖度空間のリスク大。" if frozen_ratio > 0.1 else "健全な『ぶれ（カオス）』が維持されています。"
        }

# ==========================================
# 実行テスト用の疑似データ作成とデバッグ実行
# ==========================================
if __name__ == "__main__":
    # N=2000の「忖度に汚染された」疑似データを生成
    N = 2000
    
    # マジョリティ(約95%)は評価4に同調、低確信度
    majority_vals = np.random.choice([3, 4, 5], size=1900, p=[0.1, 0.8, 0.1])
    majority_betas = np.random.uniform(0.1, 0.4, size=1900)
    
    # マイノリティ(約5%)は評価1を告発、高確信度（専門家など）
    minority_vals = np.ones(100)
    minority_betas = np.random.uniform(0.8, 1.0, size=100)
    
    df_mock = pd.DataFrame({
        "rating": np.concatenate([majority_vals, minority_vals]),
        "nlp_certainty_score": np.concatenate([majority_betas, minority_betas]) # NLPで抽出したβと仮定
    })
    
    # デバッガの初期化と実行
    debugger = InformationalHealthDebugger(df_mock, value_col="rating", beta_proxy_col="nlp_certainty_score")
    
    print("--- 【処方箋1】棄却アノマリーの監査 ---")
    print(debugger.audit_rejected_anomalies())
    
    print("\n--- 【処方箋2】潜在バイアスのベイジアン・ストレステスト ---")
    print(debugger.bayesian_stress_test(assumed_v2=0.5))
    
    print("\n--- 【処方箋3】スモールデータ・アンサンブル監査 ---")
    print(debugger.small_data_ensemble_audit(ensemble_size=50))
