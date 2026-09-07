# Catálogo de Tuning de Entrenamiento — HSPP en AnuraSet

Cada cambio del lado de entrenamiento hecho a la réplica, con la razón por la que se hizo y
el concepto detrás.

**Leyenda de estados:** ✅ adoptado · 🔬 barrido · ❌ revertido · ⏸ pendiente

**Documentos complementarios:** `IMPLEMENTATION.md` (registro técnico/de decisiones completo) ·
`HANDOFF.md` (estado actual y siguientes pasos)

> **Cómo leer los números.** Hay dos regímenes de split en juego y **no son
> comparables**. La escalera de ablación de 4 brazos × 3 semillas usó el split de dos vías
> (train = 53,340) y reporta `val_mAP`. El barrido de tuning usa el split de tres vías
> (train = 44,408, −17% de datos) y reporta `dev_mAP`. Dentro del barrido, `base`
> (dev 0.7151) es la única referencia válida.
>
> **σ = 0.0078 ESTÁ OBSOLETO — usar σ ≈ 0.017.** Esa cifra venía de la escalera de dos
> vías. Medida en el split de tres vías, la propia sd de 3 semillas de dual es **0.0172**
> (0.7151 / 0.7180 / 0.6869), 2.2× más grande, porque 17% menos datos de entrenamiento significa
> más varianza entre corridas. Toda conclusión de una sola semilla en este documento es
> por tanto *más débil* de lo que se escribió originalmente. Donde un veredicto de abajo
> descansa en un delta de una sola semilla por debajo de 0.017, ahora está marcado como tal en lugar de
> reevaluado silenciosamente.
>
> **n=3 limita la significancia.** Una prueba exacta de permutación por inversión de signos emparejada sobre 3 semillas
> tiene 8 asignaciones, así que la p bilateral mínima alcanzable es **0.25**. Nada en
> este proyecto puede alcanzar p<0.05. Reportar "consistente entre semillas" o "un empate".

---

## Tabla resumen

| # | Cambio | Estado | Efecto |
|---|---|---|---|
| 1 | Adam → AdamW + grupos sin decay | ✅ | exactitud; γ/β de BN y `a_raw` ya no reciben decay. **Nunca ablacionado** — nunca se corrió un brazo |
| 2 | LR cosine + warmup de 2 épocas | ✅ | eliminó la oscilación de ±0.035. `attr_flatlr` −0.0018 = **sin ganancia medible en mAP** |
| 3 | Stem pool /2 | ✅ | 4.02× menos FLOPs, 0 params extra; habilitó las corridas de 3 semillas |
| 4 | Barrido de probabilidad de MixUp | ✅ | **ADOPTADO `p=0.7`** — +0.0174 en 3 semillas, 3/3 semillas, sd 0.0027 vs 0.0172 |
| 5 | Temperatura / multiplicador de LR de alpha | ❌ | **negativo** — dispersión de α ↑, mAP ↔/↓. `tau0.3` no adoptado (2/3 semillas, p=0.750) |
| 6 | Calendario más largo para `raw` | ⏸ | verificación de honestidad sobre el empate dual-vs-raw; **nunca corrido** |
| 7 | Tamaño de lote 32 → 64 | ✅ | throughput gratis + estadísticas de BN más limpias |
| 8 | Máscara temporal de SpecAugment 40 → 12 | ✅ | evita borrar el evento etiquetado |
| 9 | Recorte de gradiente | ❌ | nunca se activó (norma 0.132), costó ~16%/paso |
| 10 | AsymmetricLoss | ❌ | peor (0.7053 vs 0.7151), −0.0098. Rechazada |
| 11 | Semillado reproducible | ✅ | flujos de RNG desacoplados; habilita media±desv. estándar |
| 12 | Split de tres vías train/dev/val | ✅ | expuso el optimismo de los umbrales; protege val |
| 13 | Fusión por clase (atribución) | ✅ | corrigió el *orden*, no el mAP absoluto (−0.0035) |
| 14 | **Control de capacidad (raw-wide)** | ✅ | **raw-wide ≥ dual → H2.** La descomposición no aporta nada con params igualados |
| 15 | **Barrido del kernel de HPSS {5,9,13,25}** | ❌ | **resultado nulo.** Dispersión 0.0150 < σ 0.017; sin tendencia. La longitud del kernel no importa |

---

## LA CONFIGURACIÓN FINAL ADOPTADA

Lo que corre la ablación reportable (`run_final_ablation.sh`, etiquetas `*_s{seed}_3way`).
Verificada contra `runs/dual_s42_3way/summary.json`.

```
arm              dual | raw | harmonic | percussive   (4 brazos x 3 semillas)
mixup_p          0.7          <- ADOPTADO, la única mejora de tuning confirmada
mixup_alpha      0.2
stem_pool        2
per_class_fusion True         <- solo dual
alpha_tau        1.0          <- tau0.3 NO adoptado
alpha_lr_mult    1.0          <- alr10 rechazado (perjudicó)
lr_schedule      cosine + warmup de 2 épocas, LR mínimo LR*0.01
optimizador      AdamW, lr 1e-3, grupos sin decay para gamma/beta de BN y a_raw
pérdida          BCEWithLogits    <- ASL rechazada (-0.0098)
grad_clip        0.0              <- norma medida 0.132, nunca se activó
batch_size       64
épocas           40, paciencia 20
hpss_kernel      17, margin 1.0   <- el barrido no encontró un valor mejor
spec_time_mask   12  (el 40 del paper reescalado de 431 -> 130 frames)
spec_freq_mask   8, n_masks 2
split            tres vías agrupado por grabación, seleccionado en dev, val leído UNA VEZ
params           15,824,344 dual / 7,983,212 single
```

**Resumen honesto de lo que realmente está respaldado por evidencia:** de los cuatro cambios acreditados
con el +0.109 sobre la config antigua, **tres miden cero o negativo individualmente**
(fusión por clase −0.0035, LR cosine −0.0018, stem pool un artefacto de una sola época) y
el cuarto (AdamW) **nunca se ablacionó**. MixUp 0.7 es la única mejora confirmada de
todo el proyecto. El resto de la configuración es ingeniería defendible, no mejora
medida, y debería describirse de esa manera.

---

## 1. `Adam → AdamW` ✅

**Qué.** Se cambió el optimizador; se dividieron los parámetros en grupos con decay / sin decay.

**Propósito.** Dos defectos separados en una línea de código.

### Concepto — weight decay desacoplado

La regularización L2 y el weight decay son idénticos con SGD simple pero **no** con
optimizadores adaptativos. Adam añade `λ·w` al gradiente, que luego pasa por el
denominador adaptativo:

```
Adam:   w ← w − lr · (ĝ + λw) / (√v̂ + ε)      ← decay dividido por la escala del gradiente
AdamW:  w ← w − lr · ĝ / (√v̂ + ε) − lr · λw   ← decay aplicado directamente
```

Con Adam, un parámetro con gradientes históricos pequeños recibe un decay efectivo *grande*
y viceversa — la intensidad de la regularización se vuelve un accidente del historial de
gradientes en lugar de un hiperparámetro elegido. AdamW (Loshchilov & Hutter) los desacopla.

### El segundo defecto, más grande

`Adam(model.parameters(), weight_decay=1e-4)` aplica decay a **cada** tensor, incluidos
los γ y β de BatchNorm. El propósito entero de BN es aprender una escala y un desplazamiento; empujar γ
hacia cero se opone directamente a eso. La práctica estándar excluye todos los parámetros 1-D.

División medida:

```
decay:    24 tensores, 15,815,236 params
no_decay: 39 tensores,      9,108 params
          └ 32 γ/β de BatchNorm + 6 sesgos + 1 a_raw
```

**Por qué excluir `a_raw` es exactitud, no higiene.** Aplicar decay al peso de fusión lo
sesga hacia `sigmoid(0) = 0.5` — **confundiendo exactamente la cantidad que la ablación
mide.**

### Por qué la familia Adam en lugar de SGD (la elección del paper, conservada)

Los pasos adaptativos por parámetro sí ayudan genuinamente al entrenamiento multietiqueta de cola larga, donde
la magnitud del gradiente que llega a las 42 salidas abarca ~3 órdenes de magnitud entre
SPHSUR (13,258 positivos) y LEPFLA (7). Una sola tasa de aprendizaje global de SGD o bien
subentrenaría la cola o desestabilizaría la cabeza.

---

## 2. Calendario de LR cosine + warmup ✅

**Qué.** `LambdaLR` — warmup lineal de 2 épocas hasta `LR`, luego decaimiento cosine hasta
`LR × 0.01`, con paso **por lote**. El paper no especifica ningún calendario.

### Propósito — el problema medido

Un 1e-3 plano durante 31 épocas produjo:

```
ep 15  0.6281      ep 24  0.6541 ← "mejor", guardado
ep 17  0.6059  ↓   ep 26  0.6050  ↓
ep 20  0.5855  ↓↓  ep 30  0.6527
ep 22  0.6142      ep 31  0.6471
```

Una **banda de ±0.035 sin descenso**, mientras la pérdida de entrenamiento se mantenía plana en 0.053–0.055.

### Concepto — orbitar vs entrar en la cuenca

Un tamaño de paso apropiado en la época 1 es demasiado grande en la época 25. El optimizador
sobrepasa repetidamente el mínimo, dando vueltas a un radio fijado por `lr × |∇|`. El recocido
encoge ese radio para que los pasos tardíos se asienten en lugar de orbitar.

### El daño mayor — y la razón real por la que esto importaba

`best.pt` se selecciona con `val_mAP > best`. Bajo una oscilación de ±0.035 eso guarda
**la época que tuvo suerte**, no el mejor modelo — inflando cada brazo en una cantidad
desconocida y específica de cada brazo.

Esa es precisamente la razón por la que una dispersión de 0.040 mAP entre los tres brazos superiores era ilegible
frente a ±0.035 de ruido. **El calendario ausente era lo que hacía inconcluyente la tabla de ablación
original.** Este cambio fue primero sobre la confiabilidad de la medición,
y segundo sobre la exactitud.

### Por qué el paso por lote

El paso por época da 40 valores discretos de LR (una escalera); el paso por lote da 33,320
y un decaimiento genuinamente suave. Importa más al final de la corrida, donde un
calendario por época mantiene un LR aún demasiado grande durante una época entera de 833 pasos.

### Por qué el warmup

Las estadísticas móviles de BatchNorm no significan nada al inicializar, así que los primeros cientos
de pasos producen gradientes grandes y mal escalados. Una rampa de 2 épocas (1,666 pasos)
evita gastar **los pasos más grandes de la corrida en las peores estimaciones de gradiente**.

### Un conflicto que conviene conocer

Cosine hace su recocido hasta el mínimo exactamente en `epochs`; un early stopping a mitad del recocido descarta
la fase de LR bajo donde ocurre la convergencia. El calendario y el detenedor quieren
cosas opuestas. Resolución: la paciencia se eleva a `max(patience, epochs//2)` cuando
cosine está activo, y el detenedor sobrevive solo como guarda contra descontroles.

Curva medida: pico 1.00e-03 en el paso 1,665; mínimo 1.00e-05; recocido monótono.

---

## 3. Stem pool `STEM_POOL=2` ✅

**Qué.** Un `MaxPool2d(2)` antes del bloque 1.

```
STEM_POOL=1:  11.216 GFLOPs, mapa del bloque 4 16×16, 819 s/época
STEM_POOL=2:   2.792 GFLOPs, mapa del bloque 4  8×8,  ~230 s/época
```

**4.02× menos FLOPs a costo cero de parámetros** — el pooling no tiene pesos, y el
promediado global absorbe el mapa más pequeño.

### Propósito — primero fidelidad, después velocidad

El paper reporta **2.86 GFLOPs a 10 s**. Con pooling solo después de cada bloque como
describe su texto, **el bloque 1 solo, a 128×431, cuesta ~6.9 G — ya 2.4× el
total reportado para toda la red.** Buscando strides de stem a 431 frames:

| downsample previo al bloque 1 | entrada al bloque 1 | GFLOPs dual |
|---|---|---:|
| ninguno | 128×431 | 37.13 |
| /2 | 64×216 | 9.37 |
| **/4** | **32×108** | **2.32** ← el más cercano a 2.86 |
| /8 | 16×54 | 0.56 |

**El paper necesariamente hace un downsample de ~/4 antes del bloque 1, que nunca describe.** Así que
correr *con* un stem pool es discutiblemente más fiel a su arquitectura real que
la lectura literal de su prosa.

### Concepto — dónde viven los FLOPs vs dónde viven los parámetros

```
bloque 4: 5,900,288 de 7,807,680 params por flujo = 75.6%   ← parámetros cargados al final
bloque 1: corre a resolución completa de entrada               ← FLOPs cargados al inicio
```

Para recortar parámetros, tocar el bloque 4. Para recortar FLOPs, tocar el bloque 1. El stem pool reduce a la mitad
ambas dimensiones espaciales *antes* de la convolución más costosa.

### Por qué no `/4`

Dejaría un mapa del bloque 4 de 4×4 — más pequeño que la convolución 7×7 de atención
espacial destinada a controlarlo. `test_spatial_map_not_degenerate` impone el piso
de 8×8.

### El efecto habilitante

Esto es lo que hizo asequibles las corridas de 3 semillas: la escalera de 12 corridas bajó de ~53 h a
~13 h.

---

## 4. Compuerta de probabilidad de MixUp 🔬

**Qué.** `MIXUP_P`, antes 1.0 (implícito).

### Propósito

Cada muestra recibía SpecAugment **y** MixUp — corrompida dos veces, el 100% del tiempo.
Consistente con la firma medida: la pérdida de entrenamiento se estancó en 0.053 con **ninguna
divergencia train/val a lo largo de 31 épocas**, es decir, **limitada por la aumentación, no
por la capacidad**, pese a 15.8M parámetros sobre 53k muestras.

### Concepto

MixUp entrena sobre combinaciones convexas `λx_i + (1−λ)x_j` con etiquetas suaves correspondientes,
imponiendo un comportamiento localmente lineal entre ejemplos. Valioso — pero el modelo también debe
ver ejemplos *limpios* para aprender cómo se ve un canto sin corromper. Con `p=1.0`
combinado con SpecAugment siempre activo, nunca lo hace.

`MIXUP_ALPHA=0.2` da Beta(0.2, 0.2), que tiene forma de U — mezclas mayormente casi limpias
con algunas fuertes ocasionales. Multietiqueta no necesita cambios porque BCE acepta objetivos
suaves.

### Resultado — barrido de `{0.3, 0.5, 0.7}`, luego CONFIRMADO con 3 semillas

Cribado (una sola semilla, split de tres vías):

```
mix0.7   dev 0.7266   ← el mejor de todo el barrido, +0.0115 sobre base
base     dev 0.7151   (p=0.5)
mix0.3   dev 0.7152   (+0.0001 — nada)
```

Confirmación (`run_confirm.sh`, semillas 42/43/44, umbral preregistrado +0.0078):

```
                 s42     s43     s44      media   vs base   p      victorias
mixup_p=0.7    0.7266  0.7212  0.7244   0.7241   +0.0174   0.250   3/3
mixup_p=0.5    0.7151  0.7180  0.6869   0.7067
dif. por semilla +0.0115 +0.0031 +0.0375
```

**ADOPTADO.** `src/config.py: MIXUP_P = 0.7` (2026-09-04). Supera el umbral, gana en
3/3 semillas, y p=0.250 es el *piso* con n=3 — no puede hacerlo mejor. La parte más fuerte
del resultado es la varianza: **sd 0.0027 vs 0.0172 de base, una reducción de 6×.**
Estabiliza el entrenamiento tanto como lo mejora.

Dos advertencias honestas. La media está inflada por la corrida de la semilla 44 de base (0.6869, un valor
atípico bajo frente a sus propios 0.7151/0.7180) — el +0.0375 de esa semilla hace la mayor parte del
trabajo. Y que `mix0.3` no haga nada mientras `mix0.7` ayuda significa que el efecto no es simplemente
monótono en la intensidad de la aumentación; 0.7 puede estar cerca de un óptimo local que no se
acotó por arriba (0.9 sin probar).

### La sorpresa, que vale señalar

Yo esperaba que *menos* aumentación ayudara — esa era la lectura de sobreajuste. **Más
ayudó.** Así que el estancamiento no era exceso de aumentación: el modelo está genuinamente limitado por
capacidad o por datos y se beneficia de regularización adicional. El razonamiento que
motivó la prueba estaba a medias equivocado; **probar ambas direcciones en lugar de solo 0.3 es
lo que lo detectó.**

---

## 5. Temperatura de alpha + LR dedicado ❌ — resultado negativo

**Qué.** `ALPHA_TAU` (`sigmoid(a_raw/τ)`) y `ALPHA_LR_MULT` (grupo propio de AdamW).

### Propósito — un hecho de magnitud de gradiente, no una observación de rendimiento

19–20 de 34 alphas por clase terminaron dentro de 0.05 de su inicialización en 0.5
(semilla 42: 19/34, 43: 11/34, 44: 20/34), desv. estándar entre clases ~0.083 — y aun así **la
correlación entre semillas es r = 0.939**. La dirección se aprende de forma confiable; la magnitud no puede
desplazarse.

Esta distinción importa para la disciplina de splits: es un hecho de *optimización*
derivado de valores de parámetros, no del rendimiento por clase, así que actuar sobre él no
filtra información de val.

### Concepto — por qué α está privado de gradiente

En la inicialización neutral, `∂a/∂a_raw = sigmoid'(0) = 0.25` — la región útil más plana de la
sigmoide. Y `a_raw` es un único escalar por clase compitiendo contra
15.8M parámetros con una misma tasa de aprendizaje compartida. Dos remedios independientes:

- **τ < 1** reescala el *mapeo*: el mismo desplazamiento crudo produce un cambio mayor en α.
  Crucialmente `sigmoid(0/τ) = 0.5` para cualquier τ, así que la inicialización no sesgada sobrevive.
- **El multiplicador de LR** reescala el *paso*.

Ninguno le dice al modelo qué flujo preferir; ambos solo hacen que el destino sea
alcanzable en 40 épocas.

### Resultado — mecánicamente correcto, prácticamente equivocado

```
corrida         τ    alr   rango de α     desv. α  atascados<0.05   dev_mAP
base           1.0   1.0   0.352-0.713   0.0913     16/34      0.7151
tau0.3         0.3   1.0   0.275-0.820   0.1471     10/34      0.7265
alr10          1.0  10.0   0.145-0.924   0.1938     10/34      0.7012
tau0.3_alr10   0.3  10.0   0.080-0.934   0.2137      5/34      0.7037
```

La dispersión de α crece **monótonamente** con ambas perillas; el conteo de atascados cae de 16 → 5. El
mecanismo hace exactamente lo que se diseñó. **Pero el mAP de dev no sigue** —
la configuración más dispersa puntúa *por debajo* del baseline.

### `tau0.3` pasó a 3 semillas y NO fue adoptado

`tau0.3` fue la única variante de α que parecía prometedora en el cribado (+0.0114), así que
entró en `run_confirm.sh` junto con `mix0.7`:

```
                 s42     s43     s44      media   vs base   p      victorias
alpha_tau=0.3  0.7265  0.7004  0.7268   0.7179   +0.0113   0.750   2/3
base           0.7151  0.7180  0.6869   0.7067
dif. por semilla +0.0114 -0.0177 +0.0400
```

Nominalmente supera el umbral de +0.0078, pero: **2/3 semillas, p=0.750, y perdió la semilla 43
por −0.0177** — un fallo mayor que su propia ganancia media. El +0.0400 de la semilla 44 es
contra el valor atípico bajo de base, así que la media no es confiable. **NO adoptado;
`ALPHA_TAU` se queda en 1.0.**

⚠️ **Desviación de la letra de la regla preregistrada, declarada abiertamente.** La regla
decía que si ambos candidatos superaban el umbral y quedaban dentro de 0.0078 *entre sí*,
eso es un empate entre ellos y se mantiene el valor por defecto. `mix0.7` y `tau0.3` terminaron
a 0.0062 de distancia — dentro de esa ventana — así que leída literalmente la cláusula habría rechazado
a *ambos*, incluido `mix0.7`. La cláusula se escribió para evitar una elección al azar entre
dos ajustes **rivales** de una misma perilla. `mixup_p` y `alpha_tau` son perillas
**independientes** (intensidad de aumentación vs temperatura de fusión), así que el desempate no aplica
como se pretendía, y en cambio cada uno se juzgó por sus propios méritos: `mix0.7` adoptado con 3/3 semillas,
`tau0.3` rechazado con 2/3 semillas. La regla estaba mal especificada, no los resultados;
el preregistro futuro debería delimitar los desempates a niveles del mismo factor.

### Concepto revisado

Que α se quedara cerca de 0.5 era un **síntoma que la red toleraba, no el cuello de botella.**
Liberar α le permite comprometerse en exceso y temprano con gradientes por clase estimados a partir de tan solo
149 positivos (DENELE, la clase de eval más rara). Un compromiso sesgo-varianza clásico: menos
sesgo en α, más varianza, pérdida neta.

### Consecuencia para la hoja de ruta

Esto **mata la fusión dinámica/condicionada a la entrada**, antes el siguiente paso mejor
posicionado. Una compuerta por clip otorga estrictamente *más* libertad — y más libertad acaba de fracasar.

En cambio **promueve el control con α congelado en 0.5**: si congelado iguala a aprendido, el
mecanismo por clase es decorativo y la ganancia real vino de mover la fusión al
nivel de logits con una cabeza compartida. El resultado #13 de abajo refuerza esto.

---

## 6. Calendario más largo para `raw` ⏸

**Qué.** `--epochs 60` específicamente para el brazo `raw`.

**Propósito.** `raw` alcanzó su máximo en las épocas **36, 37, 16 de 40** — dos de tres semillas seguían
mejorando cuando el presupuesto terminó.

### Concepto — por qué esto no es solo "más pasos"

Cosine hace su recocido hasta el mínimo exactamente en `epochs`, así que `--epochs 60` es un **calendario
distinto**, no una extensión: decaimiento más suave, más tiempo con LR moderado.

`ep60` en el brazo dual dio dev 0.7151 — idéntico a base, sin ganancia. Pero dual alcanza su máximo
en las épocas 14–25, así que no tenía nada que ganar. `raw` es el brazo que plausiblemente sí.

### Por qué importa para la honestidad, no para el rendimiento

Dual vs raw es actualmente un empate (+0.0078, p=0.437). Si una corrida más larga de `raw` mejora,
el empate se desplaza **en contra** de dual y el titular cambia. Es una verificación sobre nuestro propio
resultado preferido — que es exactamente por lo que debería correrse.

---

## 7. Tamaño de lote 32 → 64 ✅

**Qué.** El paper usa 32.

**Propósito.** 32 subutiliza MPS — el overhead de lanzamiento de kernels por paso domina.
El throughput medido era plano de 32 a 64, así que 64 es gratis.

### Concepto — un segundo beneficio

BatchNorm estima μ y σ por lote. Con 32 esas estimaciones son más ruidosas, lo que tanto
añade ruido de gradiente como contamina las estadísticas móviles usadas en la evaluación. 64
da estadísticas más limpias sin costo de throughput.

Por encima de 64 el modelo pasa a estar limitado por ancho de banda en lugar de por lanzamiento, así que la ganancia
se detiene — que es por lo que 64 y no 128.

---

## 8. Máscara temporal de SpecAugment 40 → 12 ✅

**Qué.** Se reescaló el ancho de máscara del paper.

### Propósito

Los 40 frames del paper sobre clips de **431 frames** son el 9.3% por máscara. Aplicado literalmente a
nuestros clips de **130 frames** se convierte en **30.8% por máscara, 61.5% con dos máscaras.**

### Concepto — transferir la razón, no el entero

Para clasificación de *escenas*, borrar el 60% puede sobrevivir: un "aeropuerto" es ambiente
estacionario, así que el 40% restante lleva la misma textura. Para **detección de eventos** es
fatal — un canto de rana es un evento localizado de 200–500 ms, y una máscara de 40 frames (928 ms)
puede borrarlo por completo **mientras la etiqueta sigue diciendo presente.** Eso entrena al modelo
a alucinar la especie a partir del fondo solamente.

12 frames = 278 ms, 18.5% en el peor caso con dos máscaras — preservando la *fracción* que el
paper realmente aplicó. `test_specaugment_preserves_shape_and_masks` codifica
`SPEC_TIME_MASK × SPEC_N_MASKS < 0.3 × N_FRAMES` para que el razonamiento sobreviva a ediciones
futuras.

### Subtileza que conviene conocer

Las máscaras se rellenan con `0.0`, y como el enmascarado corre *después* de la normalización ese valor es
la **media** del split de train (−27.6 dB), no silencio (≈ −3.87 en unidades normalizadas). Esta
es la recomendación original de SpecAugment y es la mejor elección aquí: un parche
relleno con la media se lee como *no informativo*, mientras que un parche relleno con silencio afirma
*confiadamente vacío* — una afirmación más fuerte y más engañosa para una tarea de detección.

---

## 9. Recorte de gradiente ❌ — añadido, medido, revertido

**Qué.** `GRAD_CLIP=1.0` → `0.0`.

**Propósito.** Añadido como seguro barato: las clases ultra-raras (LEPFLA, 7 positivos)
contribuyen gradientes dispersos grandes cuando aparecen.

### Luego medido

Norma media del gradiente antes del recorte: **0.132** — el umbral nunca podía activarse. Benchmark
aislado: **241 → 280 ms/paso (+16%)** por un no-op. A lo largo de un barrido de 13 horas, eso es
tiempo real gastado en nada.

### Concepto — el recorte no es ni gratis ni neutral

Requiere una reducción de norma sobre todo el modelo en cada paso. Peor, un umbral que se activa
*constantemente* reescala silenciosamente cada actualización, cambiando la tasa de aprendizaje efectiva de una
forma que no aparece en ningún log.

### Cómo se revirtió — que es el punto

No se eliminó. El flag permanece, y `grad_norm` se sigue registrando (muestreado cada 50
pasos, <1% de costo) para que `GRAD_CLIP=0` siga siendo una decisión **monitoreada** en lugar de una
no monitoreada. Si `grad_norm` sube hacia 1, `--grad-clip 0.5` lo reactiva.

*Corrección registrada:* inicialmente atribuí una desaceleración de 240 → 363 s/época al
recorte. El benchmark aislado mostró +16%; el resto era contención de una corrida
concurrente de pytest.

---

## 10. AsymmetricLoss 🔬

**Qué.** `--loss asl` (Ridnik et al.), preregistrada como comparación declarada
en lugar de una elección guiada por resultados.

**Propósito.** Construida a propósito para el desbalance negativo:positivo de 26.8:1.

### Concepto — dos mecanismos

- **Enfoque asimétrico** (`γ_neg=4.0, γ_pos=0.0`): los negativos fáciles reciben peso
  `p⁴ → 0` y desaparecen del gradiente; los positivos conservan peso completo. Con el 96.4% de
  la matriz de etiquetas en cero, la mayor parte de la señal de gradiente de BCE son negativos fáciles.
- **Desplazamiento de probabilidad** (`clip=0.05`): los negativos predichos por debajo de 0.05 contribuyen
  **exactamente cero** — un piso duro, no una reducción de peso. Esto importa en bioacústica,
  donde los cantos lejanos y tenues son omitidos rutinariamente por los anotadores, así que algunos "negativos"
  son positivos mal etiquetados.

Su ventaja sobre `pos_weight` es que no requiere **ninguna constante por clase** —
esquivando en lugar de acotar el problema de que la ponderación por frecuencia inversa
le daría a LEPFLA un factor de ~8,883, permitiendo que un solo ejemplo domine un lote.

### Resultado

**dev 0.7053 vs base 0.7151 — peor.**

Su pérdida de entrenamiento (0.2178) no es comparable con la de BCE; objetivo distinto. Una rareza:
`asl` muestra una brecha dev→val *positiva* (+0.0168) donde la mayoría de las corridas son negativas. No
interpretable con una sola semilla, pero vale observarlo si se persigue.

---

## 11. Semillado reproducible ✅

**Qué.** Se semilló el generador del DataLoader y el RNG de MixUp como flujos separados.

**Propósito.** El DataLoader **nunca estuvo semillado** — el orden de mezcla venía del RNG global
de torch, que `spec_augment` también consume dentro de `__getitem__`.

### Concepto — acoplamiento de flujos de RNG

Como ambos extraían de un mismo flujo, el orden de los lotes dependía de **cuántas extracciones de
aumentación habían ocurrido**. Cambiar `SPEC_N_MASKS` cambiaría silenciosamente también el orden de los lotes,
acoplando dos cosas que deben permanecer independientes — y convirtiendo cualquier ablación de
aumentación en parte una ablación de orden de lotes.

Ahora:

| Flujo | Mecanismo |
|---|---|
| Inicialización de pesos | `torch.manual_seed(seed)` |
| Orden de lotes | `torch.Generator` dedicado → `DataLoader(generator=...)` |
| λ + permutación de MixUp | `np.random.default_rng(seed)`, flujo propio |
| SpecAugment | RNG global de torch, semillado |
| Procesos worker | `seed_worker()` resemilla numpy/random (PyTorch solo semilla torch) |

### Lo que no se afirma: reproducibilidad bit a bit

Las reducciones paralelas de MPS no garantizan el orden de suma, y la suma en punto flotante no es
asociativa, así que corridas con la misma semilla divergen — lentamente, y luego de forma material una vez que la divergencia
alcanza la selección de umbrales basada en `argmax` y la elección de la mejor época. No hay equivalente
en MPS de `use_deterministic_algorithms(True)`.

Lo que el semillado *sí* compra: inicialización, orden de lotes y extracciones de aumentación
idénticos — de modo que las diferencias entre semillas miden la **sensibilidad a la semilla** en lugar de
ruido de RNG no semillado. Esa es la precondición para reportar media±desv. estándar.

---

## 12. Split de tres vías train/dev/val ✅

**Qué.** Se recortó dev desde train mediante un split agrupado anidado; el pliegue de val es
idéntico byte a byte al anterior.

```
train  44,408 clips / 767 grabaciones   gradientes + estadísticas de normalización
dev     8,932 clips / 154 grabaciones   early stopping, checkpointing, umbrales
val     8,851 clips / 153 grabaciones   leído UNA VEZ, con umbrales ajustados en dev
```

### Propósito

Val estaba haciendo tres trabajos incompatibles: early stopping, ajuste de umbrales por clase
(**42 parámetros ajustados**), y selección de arquitectura. Normalmente el remedio es
retener el conjunto de test — pero `test.7z` contiene **31,187 wavs y ningún CSV de etiquetas**, así que
no había un segundo conjunto retenido disponible mientras corrían estos experimentos. (Las etiquetas del origen
se localizaron después y test se evaluó una vez al final; ver `RESULTS_REPORT.md`
§7.4. Ninguna decisión de tuning aquí estuvo informada por test.)

### Concepto — cada uso de un conjunto retenido lo gasta

El ajuste de umbrales ajusta 42 parámetros; evaluarlos sobre los mismos datos infla
el macro-F1. Medido:

```
macro-F1 en dev ajustado  0.5569   ← umbrales ajustados en dev, evaluados en dev
macro-F1 en val ajustado  0.5133   ← los mismos umbrales, evaluados en val retenido
```

Val queda **por debajo** de la cifra ajustada en dev. Parte del antiguo "0.4568 → 0.6426 por
ajuste de umbrales" estaba ajustado sobre lo mismo que evaluaba en lugar de ser generalización real.

### Por qué val quedó idéntico byte a byte

12 corridas de ablación completadas se evaluaron sobre él; perturbarlo las volvería silenciosamente
incomparables. `test_val_fold_matches_completed_runs` impone esto contra un respaldo
almacenado, y `src/splits.py` imprime una confirmación explícita.

### Costo

Train se reduce de 53,340 → 44,408 (−17%), lo que baja cada número de tuning por
construcción. `base` con dev 0.7151 es la única referencia válida para las comparaciones
de tuning — **nunca comparar corridas de tuning contra el 0.7633 de la escalera.**

---

## 13. Fusión por clase — el resultado de atribución 🔬

**Qué.** `--scalar-fusion` vuelve a correr el α global único del paper bajo ajustes actuales
por lo demás idénticos, aislando lo que la fusión por clase realmente aportó.

### Resultado

```
base          (α por clase, 42 pesos)     dev 0.7151
attr_scalar   (α escalar único del paper) dev 0.7116     Δ = −0.0035
```

**−0.0035 está bien dentro del ruido de semillas medido de 0.0078.** En mAP absoluto, en
este split y con esta semilla, la fusión por clase aportó **aproximadamente nada.**

### Por qué esto no invalida el cambio

Los dos hechos van juntos y ambos importan:

1. **En mAP absoluto:** sin ganancia medible (−0.0035, dentro del ruido).
2. **En el orden:** con fusión escalar en la configuración *original*, dual puntuaba
   **por debajo de su propio mejor flujo de entrada** (0.6541 vs percussive 0.6886) — la fusión estaba
   destruyendo información. Con fusión por clase, dual supera a ambas entradas con
   significancia (+0.0260 sobre percussive, p=0.007).

Así que la afirmación honesta es más estrecha que "la fusión por clase mejora el modelo":
**elimina una patología** en la que el modelo fusionado rendía por debajo de sus propios componentes.
Eso es una corrección de exactitud, no una mejora de rendimiento.

### El corolario

La mayor parte del +0.109 sobre la configuración antigua debe venir de los **otros tres**
cambios simultáneos — LR cosine, AdamW, y el stem pool. Las corridas de atribución
restantes (`attr_stem1`, `attr_flatlr`) lo repartirán.

Nótese también el α de `attr_scalar`: 0.474, y se detuvo temprano en la época 13. Combinado con
el hallazgo #5 (más libertad para α perjudicó), el panorama es que **α simplemente no es donde está el
rendimiento** en esta tarea — lo que refuerza el argumento para correr el
control con α congelado en 0.5 en lugar de construir fusiones más elaboradas.

---

## 14. Control de capacidad — raw-wide vs dual ✅ (el experimento decisivo)

**Qué.** `--width 1.41` ensancha un solo flujo de mel crudo a 15,777,394 params, dentro del
0.3% de los 15,824,344 de dual. La arquitectura queda como la única diferencia.

### Propósito — romper un factor de confusión que ninguna corrida anterior podía

Dual superó a raw por +0.0078 (p=0.437) en la escalera mientras usaba **2× los parámetros**.
Dos explicaciones encajaban con todos los experimentos hasta ese punto: H1 la descomposición provee
una entrada mejor estructurada; H2 dual simplemente es más grande y HPSS es irrelevante.

Esto es inusualmente limpio de probar aquí porque `margin=1.0` preserva
`X_h + X_p = X_raw` (verificado con error de 1.5e-05), así que **HPSS añade literalmente cero
información.** Dual no puede estar ganando por saber más — solo mediante sesgo inductivo o
capacidad.

### Resultado — 3 semillas, split de tres vías, seleccionado en dev

```
              s42     s43     s44      media    sd
raw-wide    0.7219  0.7126  0.7341   0.7229   0.0108
dual        0.7151  0.7180  0.6869   0.7067   0.0172
por semilla +0.0068 -0.0054 +0.0473
                              dif. media +0.0162   p=0.500   raw-wide gana 2/3
```

**Veredicto: H2.** Por la regla preregistrada (`raw-wide ≥ dual`), la descomposición
no aporta nada a capacidad igualada — un solo flujo de mel crudo *nominalmente supera*
al modelo de doble flujo HPSS. **El mecanismo central del paper no se transfiere a
la detección multietiqueta de anuros.**

⚠️ **Reportar esto como un empate, no como una victoria.** p=0.500 en una prueba de inversión de signos emparejada. La
media de +0.0162 está impulsada casi por completo por una semilla (+0.0473), y la corrida de la semilla 44 de dual
(0.6869) es un valor atípico bajo frente a sus propios 0.7151/0.7180. Quítese esa corrida y el
efecto se evapora en buena medida. La afirmación defendible es **"a capacidad igualada, el doble flujo
HPSS no es mejor que un solo flujo de mel crudo"** — que sigue siendo un resultado
negativo para el paper, porque la afirmación del paper requiere que dual *gane*.

### Efecto secundario que reencuadró todo el documento

La sd de tres vías de dual de **0.0172** es lo que retiró la cifra σ=0.0078. Ver el
encabezado. Esto vindicó retroactivamente la desconfianza en el +0.018 de `attr_stem1`.

### Advertencia que sobrevive a cualquiera de los dos resultados

El beneficio declarado del paper es la *robustez entre dispositivos* en DCASE 2020 Task 1A.
AnuraSet usa un solo tipo de grabadora, así que el mecanismo puede estar bajo prueba sin su
ventaja principal. Corresponde incluirlo en el escrito de todas formas.

---

## 15. Barrido del kernel de HPSS ❌ — una hipótesis bien motivada, falsificada

**Qué.** `--kernel {5, 9, 13, 25}` vs el 17 heredado, requiriendo cada uno su propio
archivo de features precomputado de 6 GB. Brazo: `percussive` (el más barato, y el flujo en el que el modelo
se apoya). Una sola semilla, preregistrado.

### Propósito — el último hiperparámetro del paper sin probar, sobre bases físicas

Con `hop=512 / 22,050 Hz` cada frame es de 23.2 ms, así que el span del filtro de mediana es:

```
k=5  116 ms      k=13  302 ms      k=25  580 ms
k=9  209 ms      k=17  394 ms  <- valor por defecto heredado
```

Los cantos de anuros aquí duran **200–500 ms**. Así que k=17 pregunta "¿persiste la energía a lo largo de
394 ms?" — *más largo que muchos de los cantos que debe separar*. Un canto tonal de 250 ms falla esa
prueba y se enruta al flujo **percusivo** independientemente de su verdadera estructura.
Corroborante: la proporción de energía armónica crece monótonamente con k (0.383 / 0.394 /
0.410 / 0.425 / 0.456 para k = 5/9/13/17/25), así que con el valor por defecto solo ~42% de la energía
llega al flujo armónico — consistente con que solo-percusivo (0.7373) supere a
solo-armónico (0.7109) con p=0.015.

**Predicción hecha por adelantado:** un kernel más corto separa los cantos de rana
mejor, y la curva de sensibilidad corre en sentido *opuesto* a la del paper.

### Resultado — la predicción fue incorrecta

```
k     span     dev_mAP   vs k17
25    580 ms   0.7288    +0.0090
5     116 ms   0.7219    +0.0021
17    394 ms   0.7198     —
9     209 ms   0.7192    -0.0006
13    302 ms   0.7138    -0.0060
```

**Resultado nulo.** La dispersión total entre los cinco kernels es **0.0150 — por debajo de la sd
entre semillas de 0.0172 en este split.** No hay tendencia monótona en ninguna dirección, y k=13
queda *por debajo de sus dos vecinos*, que es la firma del ruido y no de una
curva. k=25 es nominalmente el mejor, lo opuesto a la predicción, pero +0.0090 con una
sola semilla no es un hallazgo.

**Conclusión: la longitud del kernel de HPSS no afecta de forma medible el rendimiento en esta
tarea.** `HPSS_KERNEL` se queda en 17.

### Por qué esto todavía vale la pena reportar

El argumento físico era sólido y las mediciones de proporción de energía confirmaron que el mecanismo
*sí* desplaza contenido entre flujos — y aun así el modelo posterior es
indiferente a ello. Combinado con #14 (la descomposición no aporta nada a capacidad
igualada), la lectura coherente es que **la red es insensible a cómo HPSS
reparte la energía porque no está explotando la descomposición en primer
lugar.** Dos experimentos independientes apuntando a la misma conclusión.

Mecánica verificada antes de confiar en los números: cada corrida recibió un `--h5` distinto,
y cada archivo lleva su propio atributo `hpss_kernel` estampado (5/9/13/25 confirmados).
`features.h5` (k=17) es anterior al código de estampado, así que reporta "sin estampar" — cosmético
solamente. `summary.json` no registra la ruta `--h5`, que es una brecha real de reporte
que vale cerrar si el barrido se extiende alguna vez.

---

## Lecciones transversales

**Las reversiones y los resultados negativos llevaron más información que las adopciones.**
El recorte de gradiente *parecía* gratis hasta que se midió (norma 0.132, costo 16%). Las perillas de alpha
funcionaron exactamente como se diseñaron y aun así empeoraron las cosas — cerrando la fusión
dinámica, la dirección sobre la que había estado más confiado. MixUp se movió en sentido opuesto
a la predicción. La fusión por clase, la contribución principal, resultó valer
~0.003 en mAP absoluto.

**La confiabilidad de la medición vino antes que la exactitud.** El calendario de LR (#2), la
corrección del semillado (#11), y el split de dev (#12) produjeron poca o ninguna ganancia directa de mAP.
Importaron porque sin ellos los números no se podían leer: la lotería de checkpoints
volvía sin sentido la tabla de ablación, los flujos de RNG acoplados convertían las ablaciones de
aumentación en parte ablaciones de orden de lotes, y ajustar umbrales sobre el split de reporte
inflaba el macro-F1.

**Transferir hiperparámetros requiere transferir la razón, no el entero.**
SpecAugment 40 → 12 (#8) es el caso más claro: el mismo número significaba 9.3% de un clip
en el paper y 30.8% aquí. `HPSS_KERNEL=17` era el otro candidato — 394 ms de
suavizado por mediana frente a cantos de 200–500 ms — y se **barrió y volvió nulo**
(#15). La lección se sostiene para #8 pero no generaliza: un hiperparámetro puede ser
dimensionalmente incorrecto y aun así no importar, si el modelo no está explotando el
mecanismo que controla.

**Medir antes de optimizar, y conservar la medición después de decidir.** El patrón
usado en todo el proyecto: añadir la perilla, medir si se activa, revertir si no —
pero dejar el diagnóstico en su lugar (muestreo de `grad_norm`, `frac_mixed`, `lr` por
época, α por clase en `summary.json`) para que cada decisión siga monitoreada en lugar de
supuesta.
