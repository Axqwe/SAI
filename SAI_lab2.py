import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# --- 1. загружаем датасет ---
data_root = './animal_dataset'
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_root,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_root,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Найденные классы: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 2. модель ---

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05, fill_mode='reflect'),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='reflect'),
    layers.RandomContrast(0.1)
], name="data_augmentation")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),

    data_augmentation,

    layers.Rescaling(1. / 255),

    layers.Conv2D(32, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. обучение модели ---
num_epochs = 25

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

print("\nНачинаем обучение...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    callbacks=[early_stop]
)

# --- 4. визуализация результатов ---

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Функция потерь (Loss)')
plt.xlabel('Эпоха')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Точность (Accuracy)')
plt.xlabel('Эпоха')
plt.legend()

plt.tight_layout()
plt.show()


def visualize_predictions(model, dataset, class_names, num_images=9):
    for images, labels in dataset.take(1):
        predictions = model.predict(images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)

        fig = plt.figure(figsize=(12, 12))
        for j in range(num_images):
            if j >= len(images):
                break
            ax = plt.subplot(3, 3, j + 1)

            img = images[j].numpy().astype("uint8")
            plt.imshow(img)

            color = 'green' if pred_classes[j] == labels[j] else 'red'
            plt.title(f"Предсказано: {class_names[pred_classes[j]]}\nИстина: {class_names[labels[j]]}",
                      color=color, fontsize=12)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


print("\nВизуализация примеров классификации:")
visualize_predictions(model, val_ds, class_names)


print("\nПостроение матрицы ошибок...")
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Матрица ошибок на валидационной выборке')
plt.show()