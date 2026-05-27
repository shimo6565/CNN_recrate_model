import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

# GPUメモリの動的確保（メモリを使いすぎないように設定）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DATA_DIR = 'MSS_sensor_data'
NUM_CLASSES = 9
EPOCHS = 100

CATEGORY_LABELS = [
    'Acids', 'Alcohols', 'Aliphatic hydrocarbons', 'Aromatic hydrocarbons',
    'Esters', 'Ethers', 'Halogenated compounds', 'Ketones', 'Others',
]

CATEGORY_MAP = {
    # Acids (0)
    'Acetic acid': 0, 'Butanoic acid': 0, 'Formic acid': 0,
    'Hexanoic acid': 0, 'Pentanoic acid': 0, 'Propionic acid': 0,
    # Alcohols (1)
    '1-Butanol': 1, '1-Decanol': 1, '1-Dodecanol': 1, '1-Heptanol': 1,
    '1-Hexanol': 1, '1-Nonanol': 1, '1-Octanol': 1, '1-Pentanol': 1,
    '1-Propanol': 1, '2-Butanol': 1, '2-Decanol': 1, '2-Ethyl-1-butanol': 1,
    '2-Ethyl-1-hexanol': 1, '2-Heptanol': 1, '2-Pentanol': 1, '2-Propanol': 1,
    '3-Heptanol': 1, 'Cyclohexanol': 1, 'Ethanolamine': 1, 'Ethanol': 1,
    'Methanol': 1, '3-Chloro-1-propanol': 1,
    # Aliphatic hydrocarbons (2)
    '1-Decene': 2, '1-Octene': 2, 'Cyclohexane': 2, 'Cyclopentane': 2,
    'n-Decane': 2, 'n-Dodecane': 2, 'n-Heptane': 2, 'n-Hexane': 2,
    'n-Nonane': 2, 'n-Octane': 2, 'n-Pentane': 2, 'n-Undecane': 2,
    # Aromatic hydrocarbons (3)
    '1,2-Xylene': 3, 'Benzene': 3, 'Ethylbenzene': 3, 'Mesitylene': 3, 'Toluene': 3,
    # Esters (4)
    'Butyl acetate': 4, 'Ethyl acetate': 4, 'Ethyl butylate': 4,
    'Ethyl propionate': 4, 'Methyl acetate': 4, 'Methyl propionate': 4,
    # Ethers (5)
    '1,4-Dioxiane': 5, 'Diethyl ether': 5, 'Dihexyl ether': 5,
    'Morpholine': 5, 'Tetrahydrofuran': 5, 'Tetrahydropyran': 5,
    # Halogenated compounds (6)
    '1,1,2,2-Tetrabromoethane': 6, '1,1,2,2-Tetrachloroethane': 6,
    '1,2-Dibromobenzene': 6, '1,2-Dibromoethane': 6, '1,2-Dichlorobenzene': 6,
    '1,2-Dichloroethane': 6, '1,2-Difluorobenzene': 6, '1,3-Dichlorobenzene': 6,
    '1-Bromo-2-chlorobenzene': 6, '1-Chlorohexane': 6, '1-Chloropentane': 6,
    'Bromobenzene': 6, 'Bromoform': 6,
    'Carbon tetrachloride': 6, 'Chlorobenzene': 6, 'Chlorocyclohexane': 6,
    'Chloroform': 6, 'Dichloromethane': 6, 'Fluorobenzene': 6,
    'Hexafluorobenzene': 6, 'Iodobenzene': 6, 'Perfluoromethylcyclohexane': 6,
    # Ketones (7)
    'Acetone': 7, 'Cyclohexanone': 7, 'Cyclopentanone': 7, 'Diethyl ketone': 7,
    'Dipropyl ketone': 7, 'Ethyl butyl ketone': 7, 'Methyl butyl ketone': 7,
    'Methyl ethyl ketone': 7,
    # Others (8)
    'Acetonitrile': 8, 'Benzaldehyde': 8, 'Benzonitrile': 8,
    'Dimethyl sulfide': 8, 'Hexanal': 8, 'N,N-dimethylformamide': 8, 'Water': 8,
}


def load_sample(solvent_name, data_dir):
    """Load one solvent sample (3 concentrations) and return shape (40, 14, 3)."""
    conc_signals = []
    for conc in ['5%', '10%', '20%']:
        path = os.path.join(data_dir, f'{solvent_name}{conc}.csv')
        df = pd.read_csv(path, index_col=0)
        signal = df.values.astype(np.float32)  # (2400, 14)
        signal -= signal[0]                     # zero at t=0s
        signal = signal[::60]                   # every 3s at 20Hz → 40 points
        conc_signals.append(signal)

    sample = np.stack(conc_signals, axis=2)  # (40, 14, 3)

    # normalize per channel: max value across all time and concentrations = 1
    for ch in range(14):
        max_val = sample[:, ch, :].max()
        if max_val > 0:
            sample[:, ch, :] /= max_val

    return sample


def load_dataset(data_dir, category_map):
    solvents = list(category_map.keys())
    X = np.array([load_sample(s, data_dir) for s in solvents])  # (94, 40, 14, 3)
    y = np.array([category_map[s] for s in solvents])           # (94,)
    return X, y, solvents


def build_model(input_shape=(40, 14, 3), num_classes=9):
    # Functional API: Score-CAM が model.input / model.get_layer() を参照できるよう
    # Sequential ではなく Functional API で構築する
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (2, 1), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(16, (2, 1), padding='same', activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(16, (2, 1), padding='same', activation='relu',
                      name='final_conv_layer')(x)
    x = layers.MaxPooling2D((2, 1), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def loocv(X, y, solvents, num_classes=9, epochs=100,
          save_dir='saved_models'):
    """
    LOOCV を実行し、各イテレーションの学習済みモデルを save_dir に保存する。
    保存先: saved_models/model_000.keras … model_093.keras
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(X)
    true_labels, pred_labels = [], []

    for i in range(n):
        X_train = np.concatenate([X[:i], X[i+1:]])
        y_train = to_categorical(np.concatenate([y[:i], y[i+1:]]), num_classes)
        X_test  = X[i:i+1]

        model = build_model(X.shape[1:], num_classes)
        model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0)

        # モデルを保存（後から Score-CAM を実行できるよう）
        model.save(os.path.join(save_dir, f'model_{i:03d}.keras'))

        pred = int(np.argmax(model.predict(X_test, verbose=0)[0]))
        true_labels.append(int(y[i]))
        pred_labels.append(pred)

        status = 'OK' if pred == y[i] else 'NG'
        print(f'[{i+1:2d}/{n}] {solvents[i]:<35} true={CATEGORY_LABELS[y[i]]:<25} '
              f'pred={CATEGORY_LABELS[pred]:<25} {status}')

    acc = accuracy_score(true_labels, pred_labels)
    cm  = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    return acc, cm, true_labels, pred_labels


def run_score_cam(X, y, solvents, save_dir='saved_models'):
    """
    保存済みモデルを読み込んで Score-CAM を計算する（GPU 不要）。
    loocv() で saved_models/ を生成済みであることが前提。

    Returns
    -------
    raw_maps    : ndarray, shape (94, 40, 14)
        全分子の生の重要度マップ（時間40点 × チャネル14本）
    pred_labels : ndarray, shape (94,)
    """
    from score_cam import compute_score_cam
    import tensorflow as tf

    n = len(X)
    # raw_maps: (94, 40, 14) — 生の重要度マップ（Figure 6 のカテゴリ平均に使用）
    raw_maps    = np.zeros((n, 40, 14), dtype=np.float32)
    pred_labels = np.full(n, -1, dtype=int)

    for i in range(n):
        model_path = os.path.join(save_dir, f'model_{i:03d}.keras')
        if not os.path.exists(model_path):
            print(f'[{i+1:2d}] スキップ（未保存: {model_path}）')
            continue

        model = tf.keras.models.load_model(model_path)
        pred  = int(np.argmax(model.predict(X[i:i+1], verbose=0)[0]))
        pred_labels[i] = pred

        status = 'OK' if pred == y[i] else 'NG'
        print(f'[{i+1:2d}/{n}] {solvents[i]:<35} {status}')

        # 正解・不正解問わず全分子の Score-CAM を計算
        imp_map, _ = compute_score_cam(model, X[i])  # (40, 14)
        raw_maps[i] = imp_map

    return raw_maps, pred_labels


if __name__ == '__main__':
    import sys
    # 使い方:
    #   python3 main.py          # train + scorecam を両方実行
    #   python3 main.py train    # GPU環境でLOOCVを実行してモデルを保存
    #   python3 main.py scorecam # 保存済みモデルでScore-CAMのみ実行（CPU可）
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    assert mode in ('train', 'scorecam', 'all'), \
        f"mode は 'train' / 'scorecam' / 'all' のいずれかを指定してください"

    print('Loading dataset...')
    X, y, solvents = load_dataset(DATA_DIR, CATEGORY_MAP)
    print(f'  X shape: {X.shape}, y shape: {y.shape}')

    # ── フェーズ1: LOOCV（学習 + モデル保存）────────────────────────────
    if mode in ('train', 'all'):
        build_model().summary()
        print(f'\nRunning LOOCV (94 iterations × {EPOCHS} epochs) ...')
        acc, cm, true_labels, pred_labels = loocv(
            X, y, solvents, NUM_CLASSES, EPOCHS
        )

        print(f'\n=== Result ===')
        print(f'Accuracy: {acc:.3f}')
        print('\nConfusion matrix (rows=true, cols=pred):')
        header = ''.join(f'{i:4d}' for i in range(NUM_CLASSES))
        print(f'     {header}')
        for i, row in enumerate(cm):
            print(f'  {i}  ' + ''.join(f'{v:4d}' for v in row)
                  + f'  {CATEGORY_LABELS[i]}')

        np.save('loocv_true.npy', np.array(true_labels))
        np.save('loocv_pred.npy', np.array(pred_labels))
        print('\nSaved: loocv_true.npy, loocv_pred.npy')
        print('Saved: saved_models/model_000.keras … model_093.keras')

    # ── フェーズ2: Score-CAM（推論のみ・GPU不要）────────────────────────
    if mode in ('scorecam', 'all'):
        print('\nComputing Score-CAM from saved models ...')
        raw_maps, pred_labels_cam = run_score_cam(X, y, solvents)

        from score_cam import to_interval_importance, plot_all_maps, plot_average_maps
        true_labels_cam = y

        # importance_maps.npy: (94, 14, 4) — Figure 5 用・区間平均済み
        importance_data = np.stack(
            [to_interval_importance(raw_maps[i]) for i in range(len(raw_maps))]
        )  # (94, 14, 4)
        np.save('importance_maps.npy', importance_data)
        print('Saved: importance_maps.npy  shape:', importance_data.shape)

        # Figure 5 相当: 全94分子 × 4区間パネル
        plot_all_maps(importance_data, true_labels_cam, pred_labels_cam,
                      solvents, CATEGORY_LABELS)

        # Figure 6 相当: カテゴリ平均（正解サンプルのみ）— 横軸は40点 (0~117s)
        avg_maps = []
        for cat in range(NUM_CLASSES):
            mask = (true_labels_cam == cat) & (pred_labels_cam == cat)
            avg_maps.append(raw_maps[mask].mean(axis=0)   # (40, 14)
                            if mask.any() else None)
        plot_average_maps(avg_maps, CATEGORY_LABELS)
