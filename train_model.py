"""Treinamento de uma CNN enxuta para classificação de dígitos MNIST.

Foco em Edge AI:
    - Filtros pequenos (8 -> 16 -> 32) para reduzir parâmetros e MACs.
    - GlobalAveragePooling2D no lugar de Flatten + Dense largo
      (elimina ~80% dos parâmetros típicos de uma CNN MNIST).
    - BatchNormalization para estabilizar o treino mesmo com poucas épocas.
    - Dropout leve para regularizar antes do classificador.

Saídas:
    - model.h5                       (raiz, exigido pelo CI)
    - artifacts/model.keras          (formato nativo Keras 3)
    - artifacts/saved_model/         (TF SavedModel, p/ servir/converter)
    - artifacts/metrics.json         (hyperparams + métricas)
    - artifacts/training_curves.png  (loss e accuracy por época)
    - artifacts/confusion_matrix.png (matriz de confusão no test set)
"""

from __future__ import annotations

import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
ARTIFACTS_DIR = "artifacts"

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model() -> keras.Model:
    """CNN enxuta para Edge AI.

    Decisões de projeto:
        - 3 blocos Conv (8, 16, 32) com kernel 3x3 e padding 'same':
          reduzem parâmetros sem perder capacidade representativa em 28x28.
        - BatchNormalization após cada Conv: convergência rápida em poucas épocas.
        - MaxPooling após os dois primeiros blocos: corta resolução pela metade
          duas vezes (28 -> 14 -> 7), sem custo paramétrico.
        - GlobalAveragePooling2D substitui Flatten + Dense intermediário,
          o que tipicamente removeria centenas de milhares de parâmetros.
        - Dropout 0.25 antes do softmax para regularização leve.
    """
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(10, activation="softmax", name="probs")(x)

    return keras.Model(inputs, outputs, name="mnist_edge_cnn")


def plot_training_curves(history: keras.callbacks.History, path: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_title("Loss por época")
    ax_loss.set_xlabel("época")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(history.history["accuracy"], label="train")
    ax_acc.plot(history.history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy por época")
    ax_acc.set_xlabel("época")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de Confusão (test set)")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def measure_keras_latency(model: keras.Model, x: np.ndarray, n: int = 200) -> dict:
    """Mede latência por amostra (single-image inference) em CPU."""
    sample = x[:1]
    for _ in range(10):
        model.predict(sample, verbose=0)

    times_ms = []
    for i in range(n):
        s = x[i : i + 1]
        t0 = time.perf_counter()
        model.predict(s, verbose=0)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "samples": int(n),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=2,
    )
    train_seconds = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia: {test_acc:.4f}")
    print(f"Loss:     {test_loss:.4f}")

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1        (macro): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    latency = measure_keras_latency(model, x_test, n=200)
    print(
        f"Latência Keras (CPU, single-image): "
        f"mean={latency['mean_ms']:.3f} ms | p95={latency['p95_ms']:.3f} ms"
    )

    plot_training_curves(history, os.path.join(ARTIFACTS_DIR, "training_curves.png"))
    plot_confusion_matrix(cm, os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))

    model.save("model.h5")
    model.save(os.path.join(ARTIFACTS_DIR, "model.keras"))
    model.export(os.path.join(ARTIFACTS_DIR, "saved_model"))

    metrics = {
        "hyperparameters": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "adam",
            "learning_rate": 1e-3,
        },
        "model": {
            "name": model.name,
            "total_params": int(model.count_params()),
            "trainable_params": int(
                sum(np.prod(v.shape) for v in model.trainable_weights)
            ),
        },
        "training": {
            "train_seconds": round(train_seconds, 2),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        },
        "evaluation": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "confusion_matrix": cm.tolist(),
        },
        "latency_keras_cpu": latency,
        "artifacts": {
            "h5": "model.h5",
            "keras": f"{ARTIFACTS_DIR}/model.keras",
            "saved_model": f"{ARTIFACTS_DIR}/saved_model",
            "training_curves": f"{ARTIFACTS_DIR}/training_curves.png",
            "confusion_matrix": f"{ARTIFACTS_DIR}/confusion_matrix.png",
        },
    }

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    size_h5_kb = os.path.getsize("model.h5") / 1024
    print(f"\nmodel.h5 salvo ({size_h5_kb:.2f} KB)")
    print(f"Artefatos extras em ./{ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
