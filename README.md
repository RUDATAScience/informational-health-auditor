informational-health-auditor/
├── README.md                # プロジェクトの概要と使い方（下記参照）
├── requirements.txt         # 必要なPythonライブラリ一覧
├── .gitignore               # Gitの管理から除外するファイル設定
├── ih_debugger.py           # 【メインコード】情報の健康診断クラス
├── run_audit.py             # 【実行スクリプト】CSV読み込みと実行用
├── test_data/               # テスト用CSVが入るフォルダ
│   ├── 01_healthy_5pt.csv
│   └── 04_dirty_realworld_data.csv
└── docs/                    # 論文の図表（Fig A〜Jなど）を入れるフォルダ
    └── images/

# Informational Health Auditor (情報の健康診断ツール)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Informational Health Auditor** は、大規模アンケートデータや評価データ（$N \ge 2000$）に潜む「社会的望ましさバイアス（同調圧力や忖度）」によるデータの構造的崩壊（熱的死）を検知し、診断するためのアルゴリズム監査ツールです。

## 📖 プロジェクトの背景 (Background)

ビッグデータ時代において「サンプルサイズ $N$ を増やせば真実に近づく」という前提は、社会的バイアスが存在する系においては成立しません。本ツールは、統計力学と情報幾何学の観点から、同調圧力が特定の臨界点を超えた際に発生する**「警告シグナル（マイノリティの声）の非線形な蒸発（Information Cliff）」**を事後的に検知し、データサイエンスの標準的な前処理（外れ値除去など）が引き起こす自己矛盾をデバッグします。

> **Reference:** Kawahata, Y. (2026). *Proposing 'Informational Health Diagnostics' in Computational Social Science: Phase Transitions of Minority Signals Induced by Peer Pressure and the Epistemological Limits of Big Data.* Applied Sciences (Submitted).

## ✨ 主な機能 (Core Features)

本ツールは、既に集計済みのCSVデータに対して、3つの「情報の健康回復処方箋」を自動実行します。

1. **棄却アノマリーの監査 (Audit of Rejected Anomalies)**
   - 機械学習パイプラインで「外れ値（$3\sigma$）」として除去されるデータの中に、高確信度（高$\beta$）の真のSOSが含まれていないかをスコアリングします。
2. **ベイジアン・ストレステスト (Bayesian Stress Test)**
   - 観測された有意な分布が「忖度によって生成された偽像」であると仮定し、真の分布を推論。観測分布との **KLダイバージェンス** を計算してP値への過信を警告します。
3. **スモールデータ・アンサンブル監査 (Small-Data Ensemble Audit)**
   - 大規模データをあえて極小サブグループ（$N \le 50$）に分割し、グループ間の「分散（カオス）」を測定。「忖度による熱的死（分散ゼロ）」が発生している空間の割合を可視化します。

## 🚀 インストールと使い方 (Installation & Usage)

### 1. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
