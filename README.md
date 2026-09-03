*[Versión en español](README.es.md)*

# Neder Engine

Inference accelerator for **Jetson Orin**. Makes quantized LLM/VLM decoding
**40–47 % faster** with the same measured quality, without touching your model.
Installs with `docker load`.

```
without accel     73.33 t/s
with Neder       116.34 t/s
────────────────────────────────
1.59× — measured by the installer on an Orin Nano Super
```

## Measured across five architectures

Decode, `Q4_K_M`, no conversion. Includes one family unrelated to Qwen and two
text-only models, so it doesn't look tuned to a single case:

| model | family | llama.cpp | with Neder | factor |
|---|---|---|---|---|
| TinyLlama-1.1B | Llama, text-only | 63.12 t/s | **90.80** | **1.44×** |
| SmolVLM2-2.2B | SmolLM + SigLIP | 36.98 | **52.38** | **1.42×** |
| Qwen3-VL-2B | Qwen3-VL | 35.57 | **50.82** | **1.43×** |
| Qwen2.5-VL-3B | Qwen2.5-VL | 22.90 | **33.49** | **1.46×** |
| Qwen3-4B | Qwen3, text-only | 17.25 | **25.32** | **1.47×** |

In real production — with a vision tower loaded alongside — **30.36 → 38.73
t/s (1.28×)**. We publish both numbers because an isolated benchmark always
looks better than a machine doing its actual job.

## Versus NVIDIA's TensorRT Edge-LLM

Same Jetson, same model, 50 iterations, 3 repetitions, TensorRT v0.9.0:

| context | llama.cpp | **Neder** | TensorRT v0.9.0 | advantage |
|---|---|---|---|---|
| 0 | 71.84 | **115.55** | 87.53 ± 0.76 | **1.32×** |
| 128 | 74.61 | **126.39** | 88.73 ± 2.48 | **1.42×** |
| 512 | 70.89 | **115.61** | 89.47 ± 3.01 | **1.29×** |
| 1000 | 67.21 | **106.55** | 85.00 ± 2.46 | **1.25×** |

Three caveats, because a comparison that only wins isn't credible:

- **INT4 AWQ and `Q4_K_M` don't promise the same thing** — AWQ calibrates on
  data and produces a new model you must re-validate; we run your GGUF as-is.
  And calibration isn't exclusive to their pipeline: `llama-imatrix` (shipped
  in the image) does the same on your box — measured: **4.5 % better
  perplexity, identical speedup** (1.583× plain vs 1.585× calibrated).
- **We measured against a reduced TensorRT**: its new kernels require CUDA
  12.8 and JetPack 6.2 ships 12.6 (their official JetPack 6.2 recipe doesn't
  link as documented). On JetPack 7 the gap could narrow. Not measured, not
  claimed.
- **This number went down**: against v0.6.0 we reported 1.67×–1.86×; NVIDIA
  improved 32 % in three releases. A number that expires gets marked, not
  erased.

Beyond speed: getting TensorRT Edge-LLM running on JetPack 6.2 cost us **a
full day** (versions that don't compile, official docs that don't link, forced
re-exports on a rented x86 GPU, 20 hand-patches to their export chain). This
is `docker load` and go.

## Quality: exactly what is guaranteed

It doesn't touch your weights and computes the same formula as `llama.cpp`.
Measured through the kernel's own path: perplexity **20.1092 with it vs
20.1130 without — 0.02 %**, which is 37× less than `llama.cpp` shifts between
its own evaluation modes. Generated text normally matches byte for byte; on a
near-tie under greedy sampling (we measured one case: two candidates 0.06
logprob apart) it can land on the other side — exactly like llama.cpp's own
CPU vs CUDA backends. Neither output is "more correct", and at real
temperatures the distinction is unobservable.

## Where it does NOT work

All measured, none assumed:

- **Datacenter GPUs: it loses.** On an A100 it does 0.69×–0.79× — the geometry
  is chosen for the Orin's 8 SMs; an A100 has 108. Not sold there.
- **Orin family only (`sm_87`)**: Nano, NX and AGX with the same binary.
  Jetson Thor (`sm_110`) is not covered.
- **Decode only.** Prefill already runs on tensor cores 55× faster per token;
  there is nothing to gain there.
- **Doesn't accelerate detection** (YOLO, DETR…): that's matrix–matrix on
  tensor cores; this is a matrix–vector kernel.

## Try it free — 5 days, non-commercial, one device

One command, on the Jetson:

```
curl -fsSL https://neder.kesheratmex.workers.dev/instalar | bash
```

It downloads the engine (1.4 GB, checksum-verified, resumable), loads it into
docker, signs your trial license **instantly** (it asks for your email right
in the terminal), finds the GGUF models on your box — or offers a 0.4 GB test
model — and measures the speedup with and without the engine, on your own
machine. At the end it can leave an **OpenAI-compatible server** running,
already accelerated: point your app or agent at `http://<jetson>:8080/v1`.

Running an agent on the box? Give it this instead (zero questions):

```
curl -fsSL https://neder.kesheratmex.workers.dev/instalar | NEDER_CORREO=you@example.com bash
```

Uninstalling is the mirror command — it removes everything the installer put
on the box and touches nothing else:

```
curl -fsSL https://neder.kesheratmex.workers.dev/uninstall | bash
```

When the trial expires nothing breaks: the engine falls back on its own to the
normal `llama.cpp` path and your service keeps running, minus the gain. A
commercial license is the same file with a different date — reach out in an
issue or via the profile email.

## Requirements

| | |
|---|---|
| Hardware | Jetson Orin `sm_87` — Nano, NX or AGX |
| System | JetPack 6.x / L4T R36 · CUDA 12.6 · `nvidia` runtime |
| Formats | `Q8_0` · `Q4_0` · `Q4_K` · `Q6_K` (a `Q4_K_M` is covered 197/197) |
| Memory | +388 MB with in-place repacking |
| Installer | `docker run … instalar` — checks the box, changes nothing, and measures |

All figures were measured on a Jetson Orin Nano Super (8 GB) with the GPU
clock verified at 1020 MHz during every run.

---

*Proprietary binary with encrypted kernels; per-device licensing. This repo
distributes the product and receives requests — the source code is not
public.*
