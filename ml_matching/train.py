import random
import os
import numpy as np
import matplotlib.pyplot as plt
from ml_matching.dataset import HomographyDataset
from ml_matching.arhitecture import SiameseModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from ml_matching.arhitecture import build_model

def plot_results(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['Loss'], label='Train Loss')
    plt.plot(history.history['val_Loss'], label='Val Loss')
    plt.title('Loss Convergence (Triplet Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dist_pos'], label='Positive Dist (Anchor-Pos)')
    plt.plot(history.history['dist_neg'], label='Negative Dist (Anchor-Neg)')
    plt.title('Embedding Distances')
    plt.xlabel('Epochs')
    plt.ylabel('Euclidean Distance')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(dataset_path, checkpoint_path, final_model_path):
    all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]

    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)

    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    train_gen = HomographyDataset(
        image_list=train_images,
        batch_size=32,
        patch_size=32,
        steps_per_epoch=500
    )

    val_gen = HomographyDataset(
        image_list=val_images,
        batch_size=32,
        patch_size=32,
        steps_per_epoch=100
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=4,
            verbose=1,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6
        ),

        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    base = build_model(input_shape=(32, 32, 1))
    model = SiameseModel(base)
    model.compile(optimizer='adam')

    dummy_img = np.zeros((1, 32, 32, 1), dtype=np.float32)
    model(dummy_img)
    print("Dummy passed through network")

    model_save_path = 'temp_best_siamese.keras'

    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
        except Exception as e:
            print("Starting from scratch...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    base.save(final_model_path)

    print(f"Model saved at {model_save_path}")

    plot_results(history)