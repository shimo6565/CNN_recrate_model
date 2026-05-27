"""
Score-CAM implementation for MSS sensor CNN.

Reference:
  Wang et al. (2020) Score-CAM: Score-Weighted Visual Explanations
    for Convolutional Neural Networks. arXiv:1910.01279
  Fukui et al. (2025) ACS Appl. Mater. Interfaces 17, 52728-52737
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


# ─── 定数 ─────────────────────────────────────────────────────────────────
_N_TIME   = 40      # 時間軸の点数
_DT_SEC   = 3       # サンプリング間隔 [s]
_T_MAX    = (_N_TIME - 1) * _DT_SEC   # = 117 s

# 4区間の定義（インデックス範囲）
# 40点 × 3s = 0~117s を 10点ずつ4分割
TIME_INTERVALS = [(0, 10), (10, 20), (20, 30), (30, 40)]
TIME_LABELS    = ['0-30s', '30-60s', '60-90s', '90-120s']


# ─── Score-CAM コア計算 ────────────────────────────────────────────────────

def compute_score_cam(model, input_sample,
                      layer_name='final_conv_layer',
                      target_class=None):
    """
    1サンプルに対して Score-CAM の重要度マップを計算する。

    Parameters
    ----------
    model        : 学習済み Keras モデル
    input_sample : ndarray, shape (40, 14, 3)
    layer_name   : 最終畳み込み層の名前
    target_class : 説明対象クラスのインデックス（None → 予測クラスを使用）

    Returns
    -------
    importance   : ndarray, shape (40, 14), 値域 [0, 1]
    pred_class   : int, モデルの予測クラス
    """
    x = input_sample[np.newaxis]                      # (1, 40, 14, 3)
    pred_proba = model.predict(x, verbose=0)[0]       # (num_classes,)
    pred_class = int(np.argmax(pred_proba))
    if target_class is None:
        target_class = pred_class

    # ── 最終畳み込み層の活性化マップを取得 ─────────────────────────────
    #    shape: (40, 14, 16) → 16 個のフィルタ
    extractor = Model(inputs=model.input,
                      outputs=model.get_layer(layer_name).output)
    act_maps  = extractor.predict(x, verbose=0)[0]    # (40, 14, 16)
    n_filters = act_maps.shape[-1]                    # 16

    scores    = np.zeros(n_filters, dtype=np.float32)
    norm_maps = np.zeros_like(act_maps)               # (40, 14, 16)

    for k in range(n_filters):
        act = act_maps[:, :, k]                       # (40, 14)

        # min-max 正規化 → [0, 1]
        v_min, v_max = act.min(), act.max()
        norm = (act - v_min) / (v_max - v_min + 1e-8)
        norm_maps[:, :, k] = norm

        # ポイントワイズマスキング（濃度軸にブロードキャスト）
        masked = input_sample * norm[:, :, np.newaxis]  # (40, 14, 3)

        # 対象クラスのスコア α_k
        scores[k] = float(
            model.predict(masked[np.newaxis], verbose=0)[0][target_class]
        )

    # ── 線形結合 Σ α_k · H(A_k) ─────────────────────────────────────────
    importance = np.einsum('k,ijk->ij', scores, norm_maps)  # (40, 14)
    importance = np.maximum(importance, 0)                   # ReLU
    if importance.max() > 0:
        importance /= importance.max()

    return importance, pred_class


def to_interval_importance(importance_map):
    """
    (40, 14) の重要度マップを 4区間の平均値に変換する。

    Returns
    -------
    result : ndarray, shape (14, 4)
             result[ch, k] = 区間 k におけるチャネル ch の平均重要度
    """
    result = np.zeros((14, 4), dtype=np.float32)
    for k, (start, end) in enumerate(TIME_INTERVALS):
        result[:, k] = importance_map[start:end, :].mean(axis=0)
    return result


# ─── 可視化 ────────────────────────────────────────────────────────────────

def plot_importance_map(importance_map, title='', ax=None,
                        figsize=(5, 4), vmax=1.0):
    """
    1サンプルの重要度マップをヒートマップで描画する。

    Parameters
    ----------
    importance_map : ndarray, shape (40, 14)
    title          : str
    ax             : matplotlib Axes（None の場合は新規作成）
    figsize        : tuple（ax=None のときのみ使用）
    vmax           : カラーバーの最大値
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    # importance_map.T → (14, 40): 行=チャネル, 列=時間
    # extent: x軸=0→117s,  y軸=ch1(上)→ch14(下)
    im = ax.imshow(
        importance_map.T,
        aspect='auto',
        cmap='jet',
        vmin=0, vmax=vmax,
        extent=[0, _T_MAX, 14.5, 0.5],
        interpolation='nearest',
    )
    ax.set_xlabel('time [s]')
    ax.set_ylabel('channel')
    ax.set_yticks(range(1, 15))
    ax.set_xticks([30, 60, 90])
    ax.set_xlim(0, _T_MAX)
    ax.set_ylim(14.5, 0.5)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    if standalone:
        plt.tight_layout()
        plt.show()
    return im


def plot_all_maps(importance_data, true_labels, pred_labels, solvents,
                  category_labels, save_path='importance_maps_all.png'):
    """
    全94分子の重要度マップを4区間ごとに描画する（論文 Figure 5 相当）。

    Parameters
    ----------
    importance_data : ndarray, shape (94, 14, 4)
    true_labels     : array-like, shape (94,)
    pred_labels     : array-like, shape (94,)
    solvents        : list[str], 長さ 94
    category_labels : list[str]
    """
    true_labels  = np.array(true_labels)
    pred_labels  = np.array(pred_labels)

    # カテゴリ順に並び替え
    sort_idx = np.argsort(true_labels, kind='stable')
    data_sorted    = importance_data[sort_idx]   # (94, 14, 4)
    true_sorted    = true_labels[sort_idx]
    pred_sorted    = pred_labels[sort_idx]
    solvents_sorted = [solvents[i] for i in sort_idx]

    fig, axes = plt.subplots(4, 1, figsize=(22, 14), sharex=True)

    for k, (ax, label) in enumerate(zip(axes, TIME_LABELS)):
        data_k = data_sorted[:, :, k].T   # (14, 94): 行=ch, 列=分子
        im = ax.imshow(data_k, aspect='auto', cmap='jet',
                       vmin=0, vmax=1,
                       extent=[-0.5, len(solvents) - 0.5, 14.5, 0.5])
        ax.set_ylabel('channel')
        ax.set_yticks(range(1, 15))
        ax.set_ylim(14.5, 0.5)
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

        # 正解/不正解をX軸ラベルに反映（最下段のみ）
        if k == 3:
            labels_x = [f'{"✓" if p == t else "✗"} {s[:10]}'
                        for s, t, p in zip(solvents_sorted, true_sorted, pred_sorted)]
            ax.set_xticks(range(len(solvents)))
            ax.set_xticklabels(labels_x, rotation=90, fontsize=5)

    # カテゴリ境界に縦線を引き、カテゴリ名をラベル表示
    boundaries = np.where(np.diff(true_sorted))[0] + 0.5
    cat_starts = [0] + (boundaries + 0.5).astype(int).tolist()
    cat_ends   = (boundaries - 0.5).astype(int).tolist() + [len(solvents) - 1]

    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color='white', linewidth=0.8, linestyle='--')

    # 最上段にカテゴリ名を中央表示
    for start, end, cat_idx in zip(cat_starts, cat_ends,
                                   sorted(set(true_sorted.tolist()))):
        mid = (start + end) / 2
        axes[0].text(mid, 0.1, category_labels[cat_idx],
                     ha='center', va='bottom', fontsize=7,
                     transform=axes[0].get_xaxis_transform(),
                     color='white', fontweight='bold')

    plt.suptitle('Importance Maps (Score-CAM) — all 94 molecules', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.show()


def plot_average_maps(avg_maps, category_labels,
                      save_path='importance_maps_avg.png'):
    """
    カテゴリごとの平均重要度マップを 3列グリッドで描画する（論文 Figure 6 相当）。

    Parameters
    ----------
    avg_maps        : list[ndarray(40, 14) | None], 長さ = クラス数
                      shape (40, 14): 時間40点 × チャネル14本
    category_labels : list[str]
    save_path       : 保存先パス
    """
    valid = [(lbl, m) for lbl, m in zip(category_labels, avg_maps)
             if m is not None]
    n     = len(valid)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for idx, (lbl, m) in enumerate(valid):
        plot_importance_map(m, title=lbl, ax=axes_flat[idx])
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle('Averaged Importance Maps (Score-CAM)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.show()
