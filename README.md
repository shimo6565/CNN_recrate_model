# CNN_recrate_model

MSS測定データをCNNで解析するためのコードです。  
研究 **「Harnessing Explainable AI to Explore Structure–Activity Relationships in Artificial Olfaction」** の再現実装を目的としています。

## 概要

本リポジトリは、人工嗅覚（Artificial Olfaction）におけるMSS（Membrane-type Surface stress Sensor）測定データを対象に、
CNN（Convolutional Neural Network）で分類を行うための再現実装です。


## 著作権・ライセンス表示（CC BY 4.0）

本リポジトリは、以下の原著論文に基づく再現実装です。論文情報の利用・参照にあたっては、CC BY 4.0 の条件に従って帰属表示を行います。

- 原著作者: Fukui, Y., Minami, K., Yoshikawa, G., Tsuda, K., & Tamura, R.
- 論文タイトル: *Harnessing Explainable AI to Explore Structure–Activity Relationships in Artificial Olfaction*
- 出典: *ACS Applied Materials & Interfaces* 2025, 17, 52728–52737
- ライセンス: Creative Commons Attribution 4.0 International (CC BY 4.0)
- ライセンスURL: https://creativecommons.org/licenses/by/4.0/

## データセットの取り扱い

- 論文関連データセットは NIMS Materials Data Repository (MDR) で公開されています。
- データ取得先（DOI）: 10.48505/nims.5556
- 本リポジトリには、当該データセットを同梱しません。
- データは必ず公式リポジトリからダウンロードしてください。
- 利用時は MDR 側の最新利用規約（商用利用可否・再配布条件など）を確認してください。

## 実装内容（現在）

- `TensorFlow / Keras` を用いたCNNモデル定義
- 入力形状: `(40, 14, 3)`
- 出力クラス数: `9`（デフォルト）
- 主要な層構成:
  - Conv2D (16 filters, kernel=(2,1)) × 3
  - Dropout (0.2)
  - MaxPooling2D (2,1)
  - Flatten
  - Dense (1024)
  - Dense (softmax)

## 動作環境

- Python 3.9+
- TensorFlow 2.x

## 実行方法

1. 依存パッケージをインストール

```bash
pip install tensorflow
```

2. モデル構造を確認

```bash
python main
```

`model.summary()` によりネットワーク構造が表示されます。

## 今後の拡張候補

- MSSデータの前処理パイプライン実装
- 学習・評価スクリプトの追加
- Explainable AI（例: SHAP, Grad-CAM）による解釈性解析
- 論文条件に合わせた実験設定・再現手順の整備

