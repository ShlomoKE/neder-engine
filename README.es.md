*[English version](README.md)*

# Neder Engine

Acelerador de inferencia para **Jetson Orin**. Hace el decode de LLMs y VLMs
cuantizados entre un **40 % y un 47 % más rápido**, con la misma calidad
medida, sin tocar tu modelo. Se instala con `docker load`.

```
sin acelerar      73.33 t/s
con Neder        116.34 t/s
────────────────────────────────
1.59× — medido por el instalador en un Orin Nano Super
```

## Medido en cinco arquitecturas

Decode, `Q4_K_M` sin convertir nada. Incluye una familia sin parentesco con
Qwen y dos modelos solo de texto, para que no parezca afinado a un caso:

| modelo | familia | llama.cpp | con Neder | factor |
|---|---|---|---|---|
| TinyLlama-1.1B | Llama, solo texto | 63.12 t/s | **90.80** | **1.44×** |
| SmolVLM2-2.2B | SmolLM + SigLIP | 36.98 | **52.38** | **1.42×** |
| Qwen3-VL-2B | Qwen3-VL | 35.57 | **50.82** | **1.43×** |
| Qwen2.5-VL-3B | Qwen2.5-VL | 22.90 | **33.49** | **1.46×** |
| Qwen3-4B | Qwen3, solo texto | 17.25 | **25.32** | **1.47×** |

En producción real —con la torre de visión cargada al lado— **30.36 → 38.73
t/s (1.28×)**. Damos los dos números porque un banco aislado siempre sale más
favorable que una máquina haciendo su trabajo.

## Contra TensorRT Edge-LLM de NVIDIA

Mismo Jetson, mismo modelo, 50 iteraciones, 3 repeticiones, TensorRT v0.9.0:

| contexto | llama.cpp | **Neder** | TensorRT v0.9.0 | ventaja |
|---|---|---|---|---|
| 0 | 71.84 | **115.55** | 87.53 ± 0.76 | **1.32×** |
| 128 | 74.61 | **126.39** | 88.73 ± 2.48 | **1.42×** |
| 512 | 70.89 | **115.61** | 89.47 ± 3.01 | **1.29×** |
| 1000 | 67.21 | **106.55** | 85.00 ± 2.46 | **1.25×** |

Tres advertencias, porque una comparación que solo gana no es creíble:

- **INT4 AWQ y `Q4_K_M` no prometen lo mismo** — AWQ calibra con datos y
  produce un modelo nuevo que hay que revalidar; nosotros corremos tu GGUF tal
  cual. Y la calibración no es exclusiva suya: `llama-imatrix` (incluido en la
  imagen) hace lo mismo en tu caja — medido: **4.5 % mejor perplejidad, misma
  aceleración** (1.583× plano, 1.585× calibrado).
- **Medimos contra una TensorRT recortada**: sus kernels nuevos piden CUDA 12.8
  y JetPack 6.2 trae 12.6 (su receta oficial para JetPack 6.2 no enlaza tal
  cual está documentada). Con JetPack 7 podría acortar. No está medido y no se
  afirma.
- **Este número bajó**: contra la v0.6.0 dábamos 1.67×–1.86×; NVIDIA mejoró un
  32 % en tres releases. Un número que caduca se marca, no se borra.

Aparte de la velocidad: poner en marcha TensorRT Edge-LLM en JetPack 6.2 nos
costó **un día entero** (versiones que no compilan, doc oficial que no enlaza,
re-exports forzados en una GPU x86 alquilada, 20 parches a mano a su cadena de
exportación). Esto es `docker load` y arrancar.

## Calidad: lo que se garantiza exactamente

No toca tus pesos y calcula la misma fórmula que `llama.cpp`. Medido por el
propio camino del kernel: perplejidad **20.1092 con él contra 20.1130 sin él —
0.02 %**, 37 veces menos de lo que `llama.cpp` cambia entre sus propios modos
de evaluación. El texto normalmente coincide byte a byte; en un casi-empate del
muestreo greedy (medimos un caso: dos candidatos a 0.06 de logprob) puede caer
al otro lado, igual que entre los backends CPU y CUDA de `llama.cpp`. Ninguna
salida es «más correcta», y con temperatura real la distinción es inobservable.

## Dónde NO funciona

Todo medido, no supuesto:

- **GPUs de centro de datos: pierde.** En una A100 da 0.69×–0.79× — la
  geometría está elegida para los 8 SMs del Orin; una A100 tiene 108. No se
  vende ahí.
- **Solo familia Orin (`sm_87`)**: Nano, NX y AGX sin recompilar. Jetson Thor
  (`sm_110`) no está cubierto.
- **Solo decode.** El prefill ya usa tensor cores y va 55× más rápido por
  token; no hay nada que ganar ahí.
- **No acelera detección** (YOLO, DETR…): eso es matriz–matriz con tensor
  cores; esto es un kernel matriz–vector.

## Pruébalo gratis — 30 días, no comercial, un dispositivo

Un comando, en el Jetson:

```
curl -fsSL https://neder.kesheratmex.workers.dev/instalar | bash
```

Baja el motor (1.4 GB, con suma verificada y reanudable), lo carga en docker,
firma tu licencia de prueba **al instante** (te pide el correo ahí mismo en la
terminal), encuentra los modelos GGUF de tu caja — u ofrece uno de prueba de
0.4 GB — y mide la aceleración con y sin el motor, en tu propia máquina. Al
final puede dejarte un **servidor compatible OpenAI** ya corriendo, acelerado:
apunta tu app o tu agente a `http://<jetson>:8080/v1`.

¿Traes un agente en la caja? Dale esto (cero preguntas):

```
curl -fsSL https://neder.kesheratmex.workers.dev/instalar | NEDER_CORREO=tu@correo.com bash
```

Desinstalar es el comando espejo — quita todo lo que puso el instalador y no
toca nada más:

```
curl -fsSL https://neder.kesheratmex.workers.dev/uninstall | bash
```

Al caducar no se rompe nada: el motor se repliega solo al camino normal de
`llama.cpp` y tu servicio sigue, sin la ganancia. La licencia comercial es el
mismo fichero con otra fecha — escríbenos en un issue o al correo del perfil.

## Requisitos

| | |
|---|---|
| Hardware | Jetson Orin `sm_87` — Nano, NX o AGX |
| Sistema | JetPack 6.x / L4T R36 · CUDA 12.6 · runtime `nvidia` |
| Formatos | `Q8_0` · `Q4_0` · `Q4_K` · `Q6_K` (un `Q4_K_M` queda cubierto 197/197) |
| Memoria | +388 MB con repaquetado en sitio |
| Instalador | `docker run … instalar` — revisa la caja, no cambia nada, y mide |

Todas las cifras se midieron en un Jetson Orin Nano Super (8 GB) con el reloj
de GPU verificado a 1020 MHz durante cada corrida.

---

*Binario propietario con kernels cifrados; licencia por dispositivo. Este repo
distribuye el producto y recibe las solicitudes — el código fuente no es
público.*
