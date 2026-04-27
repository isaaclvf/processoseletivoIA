# Comparativo TFLite

- Baseline Keras (`model.h5`): **136.64 KB**, accuracy = **0.9797**
- Variante exportada como `model.tflite`: **int8**

| Variante | Tamanho (KB) | Acurácia | Δ vs FP32 | Mean (ms) | p95 (ms) | I/O dtype |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 30.50 (100%) | 0.9797 | +0.00 pp | 0.039 | 0.055 | float32/float32 |
| dynamic | 14.27 (47%) | 0.9800 | +0.03 pp | 0.030 | 0.041 | float32/float32 |
| float16 | 20.45 (67%) | 0.9797 | +0.00 pp | 0.042 | 0.079 | float32/float32 |
| int8 | 14.56 (48%) | 0.9749 | -0.48 pp | 0.040 | 0.057 | int8/int8 |

Latência medida com `tf.lite.Interpreter` em CPU, single-image inference, n=500 amostras após warmup.
