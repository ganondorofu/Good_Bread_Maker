import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# データディレクトリの設定
base_dir = os.path.join(os.path.dirname(__file__), 'Input')
train_data_dir = os.path.join(base_dir, 'Train')
validation_data_dir = os.path.join(base_dir, 'Test')

# データジェネレータの設定
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# データジェネレータのセットアップ
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# VGG16モデルの読み込みと設定（トップ層を無効にして転移学習用にカスタマイズ）
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16の重みを固定（フリーズ）し、学習時に更新しないように設定
for layer in vgg_base.layers:
    layer.trainable = False

# 新しい分類層を追加
model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),  # 全結合層
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3クラス分類（生焼け、よく焼けている、焼きすぎ）
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# モデルの評価
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
