# Réplica de HSPP en AnuraSet

Réplica de la CNN de doble flujo HPSS + convoluciones asimétricas de
*"Structure-aware acoustic scene classification: a feature decoupling framework
using HPSS and asymmetric convolutions"* (Liu & Fan, Sci Reports 2026), adaptada
de clasificación de escenas acústicas de etiqueta única a **detección multietiqueta
de especies de anuros** en AnuraSet.

Nótese que el paper no propone un método AST/Transformer. Propone una CNN
liviana y la posiciona *frente a* AST (12.4M vs 86M params, 24.1 ms vs
68.3 ms de inferencia, 72.1% vs 73.1% de exactitud). AST es uno de sus baselines.

## Instalación

**Opción A — script de setup (recomendada, hace todo lo de abajo en un paso):**

```bash
./setup.sh
```

Crea `.venv`, instala `requirements.txt`, extrae los datos, corre `src.splits`,
precomputa `data/features.h5` y ejecuta la suite de tests — cada paso es
idempotente (se puede re-ejecutar sin duplicar trabajo). Usa
`PYTHON_BIN=/ruta/a/python3.11 ./setup.sh` si Python 3.11 no está en
`~/.local/bin/python3.11`.

**Opción B — pasos manuales:**

```bash
uv venv --python ~/.local/bin/python3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

tar -xf train.7z -C data/          # 62,191 clips, 7.7 GB
tar -xf test.7z  -C data/          # 31,187 clips (las etiquetas vienen del origen, más abajo)
```

`requirements.txt` fija todas las dependencias del proyecto (torch, torchaudio,
librosa, h5py, fastapi, etc.). Python 3.11 está fijado deliberadamente: 3.14 no
tiene wheels confiables de torch, y librosa necesita numba, que se retrasa
respecto a las versiones nuevas de CPython.

### Dónde va la data

El repo espera esta estructura relativa a la raíz del proyecto (todo bajo
`data/`, definido en `src/config.py`):

```
paper1/
├── train.7z, test.7z            # archivos comprimidos, se extraen a data/
├── train.csv                    # metadata de entrenamiento (nombre, etiquetas)
├── test_files.csv               # lista de archivos de test
├── AnuraSet_v1.0.0/
│   ├── audio/…                  # wavs crudos por sitio/fecha (para score_test.py)
│   └── metadata.csv             # etiquetas oficiales del test upstream, col. `subset`
└── data/
    ├── train/                   # extraído de train.7z — wavs de entrenamiento
    ├── test/                    # extraído de test.7z  — wavs de test (sin CSV de labels)
    ├── splits/                  # generado por `python -m src.splits`
    ├── features.h5              # generado por `python -m src.precompute` (5.8 GB)
    └── features_test.h5         # generado por `python -m src.precompute` sobre data/test
```

Ningún script crea `train.7z` / `test.7z` / `AnuraSet_v1.0.0/` por sí mismo —
esos deben colocarse manualmente en la raíz del repo antes de correr
`./setup.sh` o los pasos manuales. Todo lo que cuelga de `data/` (splits,
`.h5`) sí se genera automáticamente y los scripts de precompute/split detectan
si ya existe para no repetir trabajo.

## Device support: Apple Silicon (MPS) y NVIDIA (CUDA)

El código se desarrolló y midió en Apple Silicon (M4 Pro) usando el backend
**MPS** de PyTorch, y esa sigue siendo la ruta por defecto en esa máquina — no
se cambió nada de esa configuración. Se agregó soporte **condicional** para
GPUs NVIDIA (CUDA) para poder correr exactamente el mismo código sin
modificaciones en otra computadora con una tarjeta NVIDIA.

**Selección automática de dispositivo (`src/engine.py::pick_device`):**

```python
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

El orden es **CUDA → MPS → CPU**. En la Mac (sin CUDA disponible) esto elige
`mps` exactamente como antes. En una máquina con GPU NVIDIA, elige `cuda`
automáticamente — sin flags, sin variables de entorno, sin tocar `train.py`,
`serve.py` ni `predict_test.py` (los tres llaman a `pick_device()` en vez de
hardcodear `"mps"`).

**Qué cambia según el dispositivo, sin intervención manual:**

| Comportamiento | MPS (Apple Silicon) | CUDA (NVIDIA) |
|---|---|---|
| `pin_memory` en el DataLoader | `False` (memoria unificada, no hay copia DMA que aceleres) | `True` (acelera la copia host→GPU sobre PCIe) |
| Semillado (`train.set_seed`) | `torch.mps.manual_seed` (no-op explícito) | `torch.cuda.manual_seed_all` (cubre todas las GPUs visibles) |
| Reproducibilidad bit a bit | No alcanzable — el orden de reducción de Metal no es fijo | Alcanzable con `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`, pero **no se activa** aquí a propósito, para mantener el mismo contrato de "media±sd entre semillas" en ambos dispositivos |
| `channels_last` | Falla en el backward de MPS (probado y descartado) | No probado en este proyecto; en general sí soportado en CUDA — pendiente de validar si se usa esa máquina para más que una corrida de humo |
| Precisión | Solo fp32 (Metal no tiene fp64; autocast/bf16 no ganan nada en MPS, ver medición abajo) | fp32 por defecto igual que en MPS; autocast/AMP en CUDA sí suele acelerar (no medido todavía en este repo — pendiente al validar en la máquina nueva) |

**Instalar el wheel de torch correcto en la máquina NVIDIA:** `requirements.txt`
no fija una build de CUDA específica porque el wheel correcto depende de la
versión del driver de esa máquina. Antes de `uv pip install -r
requirements.txt`, instalar torch con el índice de PyTorch para la versión de
CUDA disponible, por ejemplo:

```bash
# Ejemplo para CUDA 12.1 -- ajustar cu121 según `nvidia-smi` / la política del
# entorno de destino. Ver https://pytorch.org/get-started/locally/
VIRTUAL_ENV=.venv uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
VIRTUAL_ENV=.venv uv pip install -r requirements.txt   # el resto de dependencias
```

Instalando torch primero con el índice CUDA, el `torch>=2.2` de
`requirements.txt` ya está satisfecho y `uv` no lo reemplaza por un wheel
CPU-only.

**Verificar qué dispositivo se detectó** antes de lanzar un entrenamiento largo:

```bash
.venv/bin/python -c "from src.engine import pick_device; print(pick_device())"
```

**Qué falta re-medir en la máquina NVIDIA (no asumir que los números de MPS
transfieren):** los tiempos por época de la sección "Medido en esta máquina"
más abajo, si `num_workers=0` sigue siendo óptimo (el ratio DataLoader/modelo
que lo justifica fue medido solo contra el throughput de MPS), y si autocast/AMP
sí aporta una ganancia real en CUDA — a diferencia de MPS, donde se midió y
se descartó explícitamente.

## Pipeline

```bash
.venv/bin/python -m src.splits                    # split agrupado en 3 vías + asserts de fuga
.venv/bin/python -m src.precompute                # -> data/features.h5 (5.8 GB, ~16 min)
.venv/bin/python -m pytest tests/ -q               # 19 tests de verificación (~4.5 min)
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
  .venv/bin/python -u -m src.train --arm dual     # un brazo de la ablación
./run_final_ablation.sh                           # la ablación reportable de 4 brazos x 3 semillas
.venv/bin/python -m src.analyze                   # tabla de ablación
.venv/bin/python -m src.analyze --tuning          # barrido de tuning, ordenado por dev
.venv/bin/python -m src.analyze --complexity      # params / FLOPs
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/uvicorn src.serve:app --port 8000
```

El paquete `test.7z` contiene 31,187 wavs **sin CSV de etiquetas**, por lo que durante
los experimentos `val` fue la única estimación retenida. Las etiquetas se localizaron
después en el origen, en `AnuraSet_v1.0.0/metadata.csv` (columna `subset`; train=62,191 /
test=31,187, coincidencia exacta con el split local) y el conjunto de test se evaluó
**una sola vez** al final con umbrales ajustados en dev. Ver `RESULTS_REPORT.md` §7.

## Configuración actual (adoptada)

La configuración con la que corre la ablación reportable. La única fuente de verdad es
`src/config.py`; esta tabla registra *qué evidencia respalda cada valor*. Detalle completo en
`TUNING.md`, estado actual en `HANDOFF.md`.

| Parámetro | Valor | Evidencia |
|---|---|---|
| `MIXUP_P` | **0.7** | **Adoptado.** +0.0174 en 3 semillas vs 0.5, gana 3/3, sd 0.0027 vs 0.0172 (reducción de varianza de 6×). La única mejora de tuning confirmada del proyecto |
| `MIXUP_ALPHA` | 0.2 | Beta(0.2,0.2), con forma de U — mezclas mayormente casi limpias |
| `STEM_POOL` | 2 | 4.02× menos FLOPs (11.216 → 2.792 G), 0 params extra. `stem_pool=1` parecía dar +0.018 pero fue un artefacto de una sola época que costaba 3.8× de cómputo |
| `PER_CLASS_FUSION` | True | **Corrección de exactitud, no una ganancia.** Con α escalar, dual puntuaba *por debajo* de solo-percusivo (0.6541 < 0.6886). α por clase (+41 params) elimina eso; cuesta −0.0035 mAP, dentro del ruido. Reproducible entre semillas (r=0.939) |
| `ALPHA_TAU` | 1.0 | `tau=0.3` rechazado: 2/3 semillas, p=0.750, perdió la semilla 43 por −0.0177 |
| `ALPHA_LR_MULT` | 1.0 | `10` rechazado — perjudicó de forma medible (0.7012 vs 0.7151) |
| `LR_SCHEDULE` | cosine + warmup de 2 épocas | Eliminó una lotería de ±0.035 en la mejor época. `attr_flatlr` −0.0018 ⇒ **sin ganancia medible en mAP**; se mantiene por estabilidad |
| optimizador | AdamW, lr 1e-3, grupos sin decay | **Nunca ablacionado.** Es exactitud (γ/β de BN y `a_raw` no deberían recibir decay), no una mejora medida |
| pérdida | BCEWithLogits | ASL declarada por adelantado, puntuó −0.0098. Rechazada |
| `GRAD_CLIP` | 0.0 | Norma pre-clip medida en 0.132 — un umbral de 1.0 nunca se activaba y costaba ~16%/paso |
| `BATCH_SIZE` | 64 | El paper usó 32, que subutiliza MPS. 192 gana 11% más pero cambia el LR efectivo — rechazado para preservar la comparabilidad |
| épocas / paciencia | 40 / 20 | `ep60` ganó ±0.000 |
| `HPSS_KERNEL` | 17 (394 ms) | Barrido {5,9,13,25} = 116–580 ms. **Resultado nulo**: dispersión 0.0150 < σ 0.0172, sin tendencia |
| `HPSS_MARGIN` | 1.0 | Preserva `X_h + X_p = X_raw` con error de 1.5e-05 — es decir, HPSS aporta **cero** información |
| `SPEC_TIME_MASK` | 12 | El 40 del paper se ajustó sobre clips de 431 frames; los nuestros tienen 130. Transferido como duración, no como entero |
| `SPEC_FREQ_MASK` / máscaras | 8 / 2 | |
| split | tres vías, agrupado por grabación | dev selecciona; val se lee **una vez**; test se evalúa una vez al final con umbrales ajustados en dev |
| params | 15,824,344 dual / 7,983,212 single | 15,777,394 para el control de capacidad de ancho 1.41 |

**La arquitectura es la del paper, intacta** — tipos de capa, orden de bloques, anchos
de canales y los pares de convoluciones asimétricas 1×n / n×1 son todos los publicados. Cada
cambio de arriba es de optimizador, aumentación o del lado de la entrada, que es lo que mantiene
esto como una réplica de la ablación del paper y no como un modelo distinto.

**Léase esto con honestidad:** de los cuatro cambios que en su momento se acreditaron con +0.109
sobre la configuración antigua, tres miden cero o negativo individualmente y el cuarto (AdamW)
nunca se ablacionó. MixUp 0.7 es la única mejora confirmada. El resto es ingeniería
defendible, no ganancia demostrada.

## Medido en esta máquina (M4 Pro, 24 GB, MPS)

| | valor |
|---|---|
| extracción de features HPSS | 29.8 ms/clip → ~16 min para 62,191 clips en 10 procesos |
| features.h5 | 5.8 GB (3 flujos: X_h, X_p, X_raw), fp16 |
| brazo dual | **~190 s/época** (`STEM_POOL=2`) |
| brazos de flujo único | **~92 s/época** |
| raw-wide (`--width 1.41`) | ~255 s/época |
| proporción de la época dedicada a eval | 11.5 s de 185 s = **6.2%** — optimizar eval no tiene sentido |
| autocast bf16 | funciona en MPS, no gana **nada**: dual 1.02×, raw 0.97× (más lento) |
| DataLoader (num_workers=0) | 2,500 muestras/s — 38× más rápido que el modelo, así que no es el cuello de botella |
| params (dual / single) | 15,824,344 / 7,983,212 |
| σ entre semillas (split de tres vías) | **0.0172** — no el 0.0078 de la escalera de dos vías |

`num_workers=0` es el valor medido como mejor: los features están precomputados, así que cada
muestra es una lectura HDF5 pequeña y el overhead de crear workers es costo puro.

## Desviaciones respecto al paper, y por qué

| # | Paper | Aquí | Razón |
|---|---|---|---|
| 1 | Softmax + entropía cruzada, 10 clases | Sigmoide + BCE, 42 logits | La tarea es multietiqueta: 0–8 especies por clip |
| 2 | Exactitud | mAP / macro-F1 sobre 34 clases | 36% de los clips son todo-negativos, así que la exactitud no significa nada |
| 3 | Máscara temporal de SpecAugment 40 | **12** | 40 frames es el 31% de un clip de 130 frames; 2 máscaras borrarían el 60% de un canto |
| 4 | Tamaño de lote 32 | 64 | 32 subutiliza MPS; el throughput es plano de 32→64, así que 64 no cuesta nada |
| 5 | Salida `Softmax` | logits crudos | `BCEWithLogitsLoss` aplica la sigmoide internamente y de forma estable |

## Dos inconsistencias internas encontradas en el paper

**1. Conteo de parámetros.** El paper reporta 12.4M. Su progresión de canales declarada
(`64→64→128→256→512` — cinco números para cuatro bloques) es ambigua y ninguna lectura
reproduce 12.4M:

| lectura | params |
|---|---|
| literal (la nuestra): 1→64, 64→128, 128→256, 256→512 | **15.82M** |
| terminando en 256 canales | 4.16M |
| cinco bloques | 16.07M |
| `ci→co` en ambas convs de un bloque | 10.54M |

**2. FLOPs — la más grave.** El paper reporta 2.86 GFLOPs con entrada de 10 s.
Eso es aritméticamente inalcanzable con el pooling colocado después de cada bloque como
describe el texto: el bloque 1 solo, a 128×431, cuesta ~6.9 G, superando ya el
total reportado. Buscando strides de stem a 431 frames:

| downsample previo al bloque 1 | GFLOPs de doble flujo |
|---|---|
| ninguno (128×431) | 37.13 |
| /2 (64×216) | 9.37 |
| **/4 (32×108)** | **2.32** ← el más cercano al 2.86 reportado |
| /8 (16×54) | 0.56 |

Así que el paper necesariamente reduce la resolución bastante antes del bloque 1. Nuestro
calendario literal mide 11.2 G a 3 s. Añadir un stem pool de /2 daría 2.78 G — esencialmente
la cifra del paper con ~4× menos cómputo — y es el siguiente experimento natural.

Ninguno de los dos números se obtuvo por ingeniería inversa a partir de la arquitectura; ambos se
reportan tal como se midieron.

## Disciplina de datos

**Los splits están agrupados por grabación padre.** Los segmentos adyacentes de 3 s provienen de la
misma grabación de ~58 segmentos y son casi duplicados (mismo individuo, mismo
fondo, a segundos de distancia). Un split aleatorio por filas filtra val dentro de train y produce
números halagadores y sin sentido. `src/splits.py` agrupa por `site_date_time`
(1,074 grupos) y verifica la disyunción de grupos y nombres de archivo.

Split: 53,340 train / 8,851 val entre 921 / 153 grabaciones.

**Clases raras.** La cabeza tiene los 42 logits para que los índices sigan alineados con
`train.csv`, pero las métricas principales cubren solo las **34 especies con ≥100
positivos**:

- `SCIFUS`, `SCINAS` tienen **cero** positivos — el AP queda indefinido, e incluirlas
  arrastraría el macro-mAP 4.8% hacia abajo sin ninguna razón.
- `LEPFLA` (7), `RHISCI` (11), `RHIORN` (21), `LEPELE` (34), `AMEPIC` (68),
  `SCIRIZ` (73) se reportan en una tabla separada como estadísticamente irrelevantes.

**Leave-one-site-out es un experimento de 5 clases, no de 42.** Solo `BOAFAB`, `DENMIN`,
`LEPLAT`, `PHYCUV`, `PITAZU` aparecen en más de un sitio; 35 de 42 son
de un solo sitio y ninguna aparece en los cuatro. Un modelo con un sitio retenido no puede predecir
especies que nunca vio.

**Los umbrales** se ajustan por clase solo en val, nunca en test. Las clases con <5 positivos
en val recaen en 0.5 y quedan marcadas.

## Por qué no Ray

Considerado y rechazado para cada componente, porque una sola máquina con una GPU
no particionable elimina cada propuesta de valor:

- **Ray Data/Core** para el precómputo — es un trabajo único de 16 minutos;
  `multiprocessing.Pool` más una máscara `done` da paralelismo y capacidad de reanudar
  sin la serialización al object store de arrays de 66 KB.
- **Ray Train** — `TorchTrainer` envuelve `torch.distributed`, que **no tiene backend
  MPS**. Varios workers en un mismo dispositivo MPS se reparten en el tiempo las mismas colas
  de comandos y compiten entre sí en vez de acelerar.
- **Ray Tune** — los 4 brazos de la ablación no son una búsqueda de hiperparámetros; los cuatro
  se reportan, así que aplicarles early-stopping sería incorrecto. Los trials además quedarían
  forzados a concurrencia 1.
- **Ray Serve** — el caso más fuerte, ya que el preprocesamiento en CPU (~30 ms de HPSS)
  domina el forward pass y separar los despliegues CPU/GPU es una fortaleza real de
  Serve. Pero `FastAPI` + `ProcessPoolExecutor` captura eso localmente sin
  levantar un clúster para un solo modelo.

## Notas sobre MPS

MPS no es una librería aparte — es un backend dentro de PyTorch estándar
(`torch.device("mps")`, el mismo `pip install torch`). Esta sección documenta
el comportamiento medido en la máquina Apple Silicon; ver la sección
"Device support: Apple Silicon (MPS) y NVIDIA (CUDA)" más arriba para cómo el
mismo código se comporta en una GPU NVIDIA. Consecuencias prácticas de MPS:

- `pin_memory=False`: existe para DMA asíncrono sobre PCIe, que la memoria
  unificada no tiene.
- Solo fp32. Metal no tiene fp64, y autocast/bf16 son inmaduros.
- `channels_last` **falla en el backward de MPS** aquí (`view size is not compatible…`)
  — probado y descartado.
- El orden de reducción no es determinista, así que se reporta media±desv. estándar entre semillas
  en lugar de afirmar reproducibilidad bit a bit.
- macOS crea los workers del DataLoader con spawn, por lo que los handles de `h5py` no se pueden
  serializar; `AnuraFeatures.__getstate__` descarta el handle y cada worker lo reabre de forma diferida.
- Aproximadamente 3–6× más lento que la RTX 3090 del paper para este modelo.

## Verificación (`tests/`, los 19 pasando)

El test que carga con el peso es **overfit-32**: 32 muestras, sin aumentación, sin dropout,
250 pasos. La pérdida pasó de 0.6925 → 0.000098, confirmando que la arquitectura y el cableado
de la pérdida son correctos. También se verificó: formas de salida en todos los brazos, conteo de params
en rango, mapa espacial no degenerado antes de la atención, el peso de fusión arranca en 0.5
y recibe gradiente, sin sigmoide en `forward`, límites de SpecAugment, rango del target de
MixUp.

Comprobaciones de sanidad en la ruta de features: la propiedad de máscara suave de HPSS `S_h + S_p == S`
se cumple con error de 1.5e-05, y las componentes armónicas son más suaves en el tiempo que las percusivas
en **12/12** clips probados (razón de gradiente frecuencia/tiempo armónico ~1.3 vs
percusivo ~0.4), confirmando que la descomposición separa la estructura horizontal de la
vertical como se pretende.
