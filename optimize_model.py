"""Conversão e otimização TFLite com múltiplas técnicas para Edge AI.

Variantes geradas:
    1. fp32         — baseline sem otimização (referência).
    2. dynamic      — Dynamic Range Quantization (pesos INT8, ativações FP32).
    3. float16      — Float16 Quantization (boa precisão, ~50% menor).
    4. int8         — Full Integer Quantization (INT8 input/output) usando
                      representative_dataset; melhor opção para Edge real
                      (MCUs / NPUs / Coral). Esta variante é copiada para
                      `model.tflite` na raiz (validada pelo CI).

Para cada variante medimos: tamanho do arquivo, acurácia no test set inteiro
e latência por amostra (mean / p50 / p95) com `tf.lite.Interpreter` em CPU.

Saídas:
    - model.tflite                          (raiz, INT8, exigido pelo CI)
    - artifacts/tflite/<variant>.tflite     (uma por técnica)
    - artifacts/tflite/comparison.json      (resultados estruturados)
    - artifacts/tflite/comparison.md        (tabela markdown legível)
"""

from __future__ import annotations

import json
import os
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

ARTIFACTS_DIR = "artifacts"
TFLITE_DIR = os.path.join(ARTIFACTS_DIR, "tflite")
PRIMARY_VARIANT = "int8"
LATENCY_SAMPLES = 500


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return x_test, y_test


def representative_dataset_fn(x_train: np.ndarray, n: int = 200):
    """Gerador para calibração da quantização INT8.

    ~200 amostras é suficiente para MNIST e mantém o tempo de calibração baixo
    no CI. Cada chamada produz um único tensor float32 com shape (1, 28, 28, 1).
    """
    idx = np.random.default_rng(0).choice(len(x_train), size=n, replace=False)
    samples = x_train[idx]

    def gen():
        for s in samples:
            yield [np.expand_dims(s, axis=0).astype(np.float32)]

    return gen


def convert_fp32(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_dynamic_range(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_float16(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_int8(model: tf.keras.Model, x_train: np.ndarray) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn(x_train, n=200)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def evaluate_tflite(
    tflite_path: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    latency_samples: int = LATENCY_SAMPLES,
) -> dict:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))

    def prepare_input(sample_f32: np.ndarray) -> np.ndarray:
        if in_dtype == np.int8 and in_scale > 0:
            q = sample_f32 / in_scale + in_zero
            return np.round(q).clip(-128, 127).astype(np.int8)
        if in_dtype == np.uint8 and in_scale > 0:
            q = sample_f32 / in_scale + in_zero
            return np.round(q).clip(0, 255).astype(np.uint8)
        return sample_f32.astype(in_dtype)

    correct = 0
    for i in range(len(x_test)):
        sample = np.expand_dims(x_test[i], axis=0)
        interpreter.set_tensor(input_details["index"], prepare_input(sample))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        pred = int(np.argmax(out))
        if pred == int(y_test[i]):
            correct += 1
    accuracy = correct / len(x_test)

    n_lat = min(latency_samples, len(x_test))
    for i in range(10):
        interpreter.set_tensor(
            input_details["index"], prepare_input(np.expand_dims(x_test[i], axis=0))
        )
        interpreter.invoke()

    times_ms = []
    for i in range(n_lat):
        sample = np.expand_dims(x_test[i], axis=0)
        prepared = prepare_input(sample)
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details["index"], prepared)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times_ms)

    return {
        "accuracy": float(accuracy),
        "size_kb": round(os.path.getsize(tflite_path) / 1024, 2),
        "input_dtype": str(np.dtype(in_dtype).name),
        "output_dtype": str(np.dtype(output_details["dtype"]).name),
        "latency_ms": {
            "samples": int(n_lat),
            "mean": float(times.mean()),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
        },
    }


def render_markdown_table(results: list[dict], baseline_acc_h5: float) -> str:
    header = (
        "| Variante | Tamanho (KB) | Acurácia | Δ vs FP32 | Mean (ms) | p95 (ms) | I/O dtype |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
    )
    fp32 = next((r for r in results if r["variant"] == "fp32"), None)
    fp32_size = fp32["size_kb"] if fp32 else None
    fp32_acc = fp32["accuracy"] if fp32 else baseline_acc_h5

    rows = []
    for r in results:
        delta = (r["accuracy"] - fp32_acc) * 100.0
        size_str = f"{r['size_kb']:.2f}"
        if fp32_size:
            size_str += f" ({r['size_kb'] / fp32_size * 100:.0f}%)"
        rows.append(
            "| {variant} | {size} | {acc:.4f} | {delta:+.2f} pp | {mean:.3f} | {p95:.3f} | {io} |".format(
                variant=r["variant"],
                size=size_str,
                acc=r["accuracy"],
                delta=delta,
                mean=r["latency_ms"]["mean"],
                p95=r["latency_ms"]["p95"],
                io=f"{r['input_dtype']}/{r['output_dtype']}",
            )
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    os.makedirs(TFLITE_DIR, exist_ok=True)

    print("Carregando modelo Keras…")
    model = tf.keras.models.load_model("model.h5")

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test, y_test = load_test_data()

    print("Avaliando baseline Keras (.h5)…")
    _, acc_h5 = model.evaluate(x_test, y_test, verbose=0)
    size_h5_kb = round(os.path.getsize("model.h5") / 1024, 2)
    print(f"  .h5: {size_h5_kb} KB | acc={acc_h5:.4f}")

    variants = [
        ("fp32", convert_fp32, {}),
        ("dynamic", convert_dynamic_range, {}),
        ("float16", convert_float16, {}),
        ("int8", convert_int8, {"x_train": x_train}),
    ]

    results = []
    for name, fn, kwargs in variants:
        print(f"\nConvertendo: {name}…")
        tfl_bytes = fn(model, **kwargs) if kwargs else fn(model)
        out_path = os.path.join(TFLITE_DIR, f"{name}.tflite")
        with open(out_path, "wb") as f:
            f.write(tfl_bytes)

        print(f"  Avaliando {name}…")
        r = evaluate_tflite(out_path, x_test, y_test)
        r["variant"] = name
        r["path"] = out_path
        results.append(r)
        print(
            f"  size={r['size_kb']} KB | acc={r['accuracy']:.4f} | "
            f"mean={r['latency_ms']['mean']:.3f} ms | p95={r['latency_ms']['p95']:.3f} ms"
        )

    primary = next(r for r in results if r["variant"] == PRIMARY_VARIANT)
    shutil.copyfile(primary["path"], "model.tflite")
    print(
        f"\nmodel.tflite (raiz) <- {primary['variant']} "
        f"({primary['size_kb']} KB, acc={primary['accuracy']:.4f})"
    )

    fp32 = next(r for r in results if r["variant"] == "fp32")
    print("\n--- Resumo comparativo ---")
    print(f"Baseline .h5: {size_h5_kb} KB | acc={acc_h5:.4f}")
    for r in results:
        delta_acc_pp = (r["accuracy"] - fp32["accuracy"]) * 100.0
        size_ratio = r["size_kb"] / fp32["size_kb"] * 100.0
        print(
            f"  {r['variant']:<8} {r['size_kb']:>8.2f} KB "
            f"({size_ratio:5.1f}% do FP32) | acc={r['accuracy']:.4f} "
            f"(Δ={delta_acc_pp:+.2f} pp) | mean={r['latency_ms']['mean']:.3f} ms"
        )

    md_table = render_markdown_table(results, baseline_acc_h5=acc_h5)
    md_doc = (
        "# Comparativo TFLite\n\n"
        f"- Baseline Keras (`model.h5`): **{size_h5_kb} KB**, "
        f"accuracy = **{acc_h5:.4f}**\n"
        f"- Variante exportada como `model.tflite`: **{PRIMARY_VARIANT}**\n\n"
        + md_table
        + "\nLatência medida com `tf.lite.Interpreter` em CPU, "
        f"single-image inference, n={LATENCY_SAMPLES} amostras após warmup.\n"
    )
    with open(os.path.join(TFLITE_DIR, "comparison.md"), "w") as f:
        f.write(md_doc)

    summary = {
        "baseline_h5": {"size_kb": size_h5_kb, "accuracy": float(acc_h5)},
        "primary_variant": PRIMARY_VARIANT,
        "variants": results,
    }
    with open(os.path.join(TFLITE_DIR, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRelatórios em {TFLITE_DIR}/comparison.{{md,json}}")


if __name__ == "__main__":
    main()
