import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(40, 14, 3), num_classes=9):
    """
    論文のFigure 2および付録コードに基づいたCNNモデルの構築
    input_shape: (データ点数 40, チャネル数 14, 濃度数 3) 
    num_classes: 分類数（カテゴリ分類の場合は9） 
    """
    model = models.Sequential()

    # 第1層: Convolution (filters=16, kernel=(2,1)) [cite: 104, 109]
    model.add(layers.Conv2D(filters=16, kernel_size=(2, 1), strides=(1, 1),
                            padding='same', activation='relu', input_shape=input_shape))
    
    # 第2層: Convolution
    model.add(layers.Conv2D(filters=16, kernel_size=(2, 1), strides=(1, 1),
                            padding='same', activation='relu'))
    
    # 第3層: Dropout (rate=0.2) [cite: 109]
    model.add(layers.Dropout(rate=0.2))
    
    # 第4層: Convolution
    model.add(layers.Conv2D(filters=16, kernel_size=(2, 1), strides=(1, 1),
                            padding='same', activation='relu'))
    
    # 第5層: Max Pooling (pool_size=(2,1)) [cite: 109]
    model.add(layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'))
    
    # Flatten & Dense Layers [cite: 109]
    model.add(layers.Flatten())
    model.add(layers.Dense(units=1024, activation='relu'))
    model.add(layers.Dense(units=num_classes, activation='softmax'))

    # コンパイル設定
    model.compile(optimizer='rmsprop', 
                  loss='categorical_crossentropy', 
                  metrics=['acc'])
    
    return model

# モデルのインスタンス化
model = build_model()

# モデル構造の確認
model.summary()

# 学習の実行例 (論文の設定: epochs=100, batch_size=1)
# model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)