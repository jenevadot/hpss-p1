# HANDOFF — Réplica de HSPP en AnuraSet

**Escrito:** 2026-09-06 12:10
**Repo:** `/Users/jent/UTEC/ciclo5/deepLearning/paper1` (git, rama `main`, 7 commits)
**Documentos complementarios:**
- `RESULTS_REPORT.md` — **EL ENTREGABLE.** Resultados finales. Léase esto primero.
- `KEYNOTE_PROPOSAL.md` — especificación de presentación de 19+5 diapositivas, lista para generar.
- `TUNING.md` — los 15 cambios del lado de entrenamiento. **Léase su encabezado**: sigma es 0.017.
- `IMPLEMENTATION.md` — registro técnico/de decisiones completo. Consúltese por sección.

---

## 0. ESTADO: EL PROYECTO ESTÁ COMPLETO

Réplica de Liu & Fan (Sci Reports 2026): CNN de doble flujo HPSS + convoluciones
asimétricas, reorientada de clasificación de escenas acústicas de etiqueta única con 10 clases a
**detección multietiqueta de anuros con 42 especies** en AnuraSet (62,191 clips x 3 s).

**Nada está corriendo. Las 53 corridas terminaron. Conjunto de test evaluado. Documentación al día.**

**El hallazgo, por cinco vías independientes:**

1. **Ablación de 4 brazos (PRINCIPAL).** dual 0.7241 +- 0.0027 vs raw 0.7278 +- 0.0114 en
   dev, **p=1.000 — un empate**, con 1.98x los parámetros.
2. **Conjunto de test del origen (PRINCIPAL).** dual 0.6988 +- 0.0095 vs raw 0.6925 +- 0.0082,
   **p=0.750 — un empate**, con el signo nominal INVERTIDO respecto a dev.
3. **Control de capacidad.** raw ensanchado a 15.78M empata/supera a dual (p=0.500).
4. **Barrido de kernel.** k en {5,9,13,25} vs 17: nulo (dispersión 0.0150 < sigma 0.0172).
   Mi predicción preregistrada fue INCORRECTA — k=25 nominalmente el mejor.
5. **Concordancia por clip.** dual y raw coinciden en el **99.63%** de 1.31M decisiones
   binarias (r=0.979). Computan sustancialmente la misma función.

**También establecido:**
- **La fusión escalar es un defecto genuino** (dual puntuó por debajo de solo-percusivo,
  0.6541 < 0.6886). El alpha por clase a nivel de logits (+41 params) lo corrige, r=0.939
  entre semillas — cuesta **-0.0035 mAP**. Una corrección de exactitud, NO una mejora de rendimiento.
- **MixUp 0.7 es la única mejora de tuning confirmada** (+0.0174, 3/3 semillas, reducción de
  varianza de 6x). Sobrevivió 1 de 11 intervenciones.
- **La ventaja real de dual es la estabilidad**: sd en dev 0.0027 vs raw 0.0114 (4.2x), la
  menor brecha dev->val, y la mayor autoconsistencia en test. Secundario, no una victoria.
- **El ensamblado por semillas supera cualquier diferencia arquitectónica medida** (+0.022 a
  +0.032 vs dual-raw +0.0063). dual+raw fusionados = 0.7329 vs 0.6988 del mejor individual.
- **El +0.109 sobre la config antigua NO es descomponible.** Tres de cuatro cambios miden
  ~cero individualmente; AdamW nunca se ablacionó.

---

## 1. NADA ESTÁ CORRIENDO

Todas las colas drenadas. Todos los scripts salieron limpiamente.

```bash
./watch_ladder.sh                        # 53/53 completas
.venv/bin/python -m src.score_test       # reimprime la tabla de test
.venv/bin/python -m src.analyze          # reimprime la tabla de ablación
```

### Si se reanuda el trabajo, la respuesta honesta es: no hay ningún paso siguiente obligatorio.

El entregable está terminado. Opcionales, en orden de prioridad:

1. **`raw` a 60 épocas** (~2 h). `raw` alcanzó su máximo en 36/37/16 de 40 en la escalera antigua.
   Si más presupuesto lo mejora, el empate se desplaza AÚN MÁS en contra de dual. Vale la pena
   correrlo precisamente porque solo puede perjudicar la tesis.
2. **Control con alpha congelado en 0.5** (~4 h). Si congelado ~ aprendido, el mecanismo
   por clase es decorativo y la ganancia vino de la fusión a nivel de logits + la cabeza compartida.
3. **Brazo AdamW vs Adam** (~4 h). La única decisión de configuración acreditada con la ganancia que
   nunca se probó. Requiere un nuevo flag `--optimizer`; todavía no existe.
4. **Estudio de ensamblado preregistrado.** La sección 7.5(b) del reporte se calculó
   post-hoc. Una versión limpia sería una contribución real.

**Muerto — no gastar tiempo aquí:** mejorar dual (cinco experimentos dicen que no hay
beneficio de descomposición que mejorar); tuning del kernel (nulo); fusión dinámica/condicionada
a la entrada (más libertad para alpha perjudicó de forma medible).

### DISCO: 21 GB libres, 95% usado

Recuperable si se necesita:
```bash
rm -rf data/test          # 3.9 GB. Solo audio; features_test.h5 ya está construido
rm -f test.7z             # archivo de 3.3 GB
rm -rf preds/             # 238 MB. Regenerable con src.predict_test
```
`data/features_test.h5` (2.9 GB) es necesario para reevaluar test sin volver a extraer.

---

## 2. LA CONFIGURACIÓN FINAL ADOPTADA

Verificada contra `runs/dual_s42_3way/summary.json`.

```
arm              dual | raw | harmonic | percussive    (4 brazos x 3 semillas)
mixup_p          0.7          <- ADOPTADO (config.py), la única mejora confirmada
mixup_alpha      0.2
stem_pool        2
per_class_fusion True         <- solo dual
alpha_tau        1.0          <- tau0.3 NO adoptado (2/3 semillas, p=0.750)
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

**Lo que realmente está respaldado por evidencia:** de los cuatro cambios acreditados con +0.109 sobre
la config antigua, **tres miden cero o negativo individualmente** (fusión por clase
−0.0035, LR cosine −0.0018, stem pool un artefacto de una sola época) y el cuarto
(**AdamW) nunca se ablacionó** — nunca se corrió un brazo. MixUp 0.7 es la única
mejora confirmada. Todo lo demás es ingeniería defendible, no mejora medida,
y debe describirse de esa manera.

---

## 3. RESULTADOS

### 3.0 PRINCIPAL: la ablación de 4 brazos y el conjunto de test

Detalle completo en `RESULTS_REPORT.md`. Resumen:

```
DEV (métrica de selección)                TEST (etiquetas del origen, evaluado una vez)
arm          dev_mAP media+-sd            arm          test_mAP media+-sd
raw        0.7278 +- 0.0114               dual       0.6988 +- 0.0095
dual       0.7241 +- 0.0027               raw        0.6925 +- 0.0082
percussive 0.7174 +- 0.0091               percussive 0.6843 +- 0.0090
harmonic   0.6646 +- 0.0049               harmonic   0.6492 +- 0.0015

raw - dual  +0.0037  p=1.000  1/3         dual - raw  +0.0063  p=0.750  2/3
```

**El signo se invierte entre dev y test y ninguno se acerca al piso de p=0.250.**
Ambos son empates. Confirmado en el piso (3/3 semillas) en ambos splits: todos los brazos fuertes
superan a harmonic; dual supera a percussive.

Las etiquetas de test están en `AnuraSet_v1.0.0/metadata.csv` (columna `subset`, train=62,191 /
test=31,187 — coincidencia exacta con nuestro split; 0 discrepancias de etiquetas en las 62,191 filas de train;
join de 31,187/31,187). **El split está definido EN EL ORIGEN, no construido localmente.**

### 3.1 Control de capacidad — corroboración

```
              s42     s43     s44      media    sd
raw-wide    0.7219  0.7126  0.7341   0.7229   0.0108
dual        0.7151  0.7180  0.6869   0.7067   0.0172
por semilla +0.0068 -0.0054 +0.0473
                    dif. media +0.0162   p=0.500   raw-wide gana 2/3
```

**H2.** La descomposición no aporta nada a capacidad igualada. **Reportar como empate,
no como victoria:** p=0.500, y la media está impulsada por una semilla contra el valor atípico bajo
de dual (0.6869). La afirmación defendible — *"a capacidad igualada, el doble flujo HPSS no es
mejor que un solo flujo de mel crudo"* — sigue siendo negativa para el paper, cuya afirmación
requiere que dual gane.

### 3.2 Barrido de kernel — NULO

```
k=25 (580 ms) 0.7288 | k=5 (116 ms) 0.7219 | k=17 (394 ms) 0.7198
k=9 (209 ms)  0.7192 | k=13 (302 ms) 0.7138
```

Dispersión 0.0150 < σ 0.0172, sin tendencia monótona, k=13 por debajo de sus dos vecinos. Mi
predicción (más corto es mejor, curva opuesta a la del paper) fue **incorrecta** — k=25
es nominalmente el mejor. La longitud del kernel no importa aquí.

### 3.3 Corridas de confirmación

```
                 s42     s43     s44      media   vs base   p      victorias
mixup_p=0.7    0.7266  0.7212  0.7244   0.7241   +0.0174   0.250   3/3  ADOPTAR
alpha_tau=0.3  0.7265  0.7004  0.7268   0.7179   +0.0113   0.750   2/3  rechazar
base           0.7151  0.7180  0.6869   0.7067
```

Ver `TUNING.md` #5 para la **desviación documentada** respecto a la letra de la
cláusula de desempate preregistrada (estaba mal delimitada — apuntaba a perillas independientes en lugar
de a niveles de un mismo factor).

### 3.4 Atribución — no descomponible

```
base          (los cuatro cambios)      dev 0.7151
attr_scalar   (α escalar del paper)     dev 0.7116   -0.0035
attr_flatlr   (sin LR cosine)           dev 0.7133   -0.0018
attr_stem1    (sin stem pool)           dev 0.7334   +0.018  <- pico de una sola época, ep13
```

El +0.018 de `attr_stem1` **no es real**: nunca superó su valor de ep13 en las
13 épocas siguientes y pasó buena parte de ellas por debajo de base. Seleccionar el máximo
sobre 40 épocas está sesgado hacia arriba para la secuencia más ruidosa.

**El +0.109 sobre la config antigua no es descomponible en estos cuatro factores.** Cada remoción
cuesta ~nada, AdamW nunca se probó, y la comparación con la config antigua abarca tanto un
cambio de split (dos vías→tres vías) como un cambio de métrica (val→dev), así que parte de eso es un
artefacto de medición y no una mejora.

### 3.5 Escalera antigua de dos vías — NO REPORTABLE

```
dual        0.7633 ± 0.0029      raw   0.7555 ± 0.0139   (p=0.437)
percussive  0.7373 ± 0.0059      harm  0.7109 ± 0.0086
```

Se conserva en disco (`*_stem2`) como registro, pero **val hacía la selección, el checkpointing,
el ajuste de umbrales Y el reporte** — sesgada optimistamente por construcción. Las corridas
`*_3way` las reemplazan para el paper.

---

## 4. DISCIPLINA DE SPLITS — LÉASE ANTES DE TOCAR CUALQUIER COSA

```
train  44,408 clips / 767 grabaciones   gradientes + estadísticas de normalización
dev     8,932 clips / 154 grabaciones   early stopping, checkpointing, umbrales
val     8,851 clips / 153 grabaciones   leído UNA VEZ al final, umbrales ajustados en dev
```

**Las etiquetas de test EXISTEN** — en `AnuraSet_v1.0.0/metadata.csv`, localizadas después de que las 12
corridas terminaran. Así que dev y val guiaron cada decisión y test se evaluó **una vez** al final
con umbrales ajustados en dev. Ese orden es correcto y debe preservarse: test
no jugó ningún papel en la selección.

- **Seleccionar por `dev_mAP`.** Nunca por `val_mAP`.
- **Nunca** elegir una arquitectura inspeccionando los AP por clase de val/dev. Dos ideas
  ya se rechazaron por esto: inicializar α desde los AP por clase de flujo único,
  y añadir un tercer flujo `raw` "porque raw gana en BOALUN/PHYDIS".
- Preregistrar la regla de selección de cualquier barrido nuevo en el encabezado del script — y **delimitar
  los desempates a niveles del mismo factor** (el error que ya se cometió una vez).

**σ = 0.0078 ESTÁ OBSOLETO. Usar σ ≈ 0.017.** La cifra antigua venía de la escalera de dos
vías; la sd de tres vías de dual es 0.0172, 2.2× más grande, porque 17% menos datos de
entrenamiento significa más varianza.

**n=3 limita la significancia a p=0.25.** Un test exacto de inversión de signos emparejado sobre 3 semillas tiene
8 asignaciones. Nada aquí puede alcanzar p<0.05. Reportar "consistente entre semillas" o
"un empate" — nunca "significativo".

---

## 5. SIGUIENTES PASOS

**Reemplazado por §1.** Nada es obligatorio; los seguimientos opcionales están listados allí.

---

## 6. EL TITULAR HONESTO, TAL COMO ESTÁ

> En detección multietiqueta de anuros, el marco de doble flujo HPSS del paper **no es
> mejor que un modelo de flujo único sobre el espectrograma mel sin descomponer cuando
> se iguala el conteo de parámetros** (raw-wide 0.7229 vs dual 0.7067, p=0.500 — un empate).
> Como `X_h + X_p = X_raw` exactamente, HPSS no añade información; la ventaja observada
> a capacidad no igualada (+0.0078, p=0.437) es atribuible al conteo de parámetros.
>
> Un barrido del kernel del filtro de mediana de HPSS entre 116–580 ms no encontró **ningún
> efecto medible** (dispersión 0.0150 < σ entre semillas 0.0172), pese a confirmarse que el kernel sí
> desplaza energía entre flujos. El modelo es indiferente a la descomposición que se le
> entrega.
>
> La **fusión de escalar único del paper es un defecto genuino**: hizo que el modelo dual
> puntuara *por debajo de su propio mejor flujo de entrada* (0.6541 vs 0.6886). Reemplazarla por
> fusión por clase a nivel de logits (+41 parámetros) elimina esa patología y es
> reproducible entre semillas (r=0.939) — pero cuesta **−0.0035 mAP**, dentro del ruido.
> Una corrección de exactitud, no una ganancia de rendimiento.
>
> Dos de las cifras reportadas en el paper son **internamente inconsistentes** — los 2.86
> GFLOPs son aritméticamente inalcanzables a partir de su arquitectura declarada, lo que implica un
> downsample temprano de ~/4 no documentado.

Ese es un resultado negativo legítimo y publicable más una contribución metodológica
concreta. No debe inflarse a "HPSS funciona" — y la fusión por clase
no debe inflarse a "nuestra corrección mejora la exactitud".

---

## 7. MAPA DEL CÓDIGO

| Archivo | Rol | A qué prestar atención |
|---|---|---|
| `src/config.py` | **todos** los hiperparámetros, única fuente de verdad | `MIXUP_P=0.7`, `STEM_POOL=2`, `PER_CLASS_FUSION=True`, `GRAD_CLIP=0.0`, `ALPHA_TAU=1.0` |
| `src/features.py` | la ÚNICA ruta de features | importado por precompute **y** serve — nunca reimplementar |
| `src/splits.py` | split agrupado de tres vías + aserciones | `verify_split` corre dentro de `make_split` |
| `src/dataset.py` | dataset HDF5, SpecAugment, MixUp | `build_datasets` retorna `(train, dev, val, norm)` |
| `src/model.py` | HSPPNet | `_alpha()` es la única definición de α |
| `src/engine.py` | bucle de entrenamiento, scheduler, grupos de parámetros | dev guía la selección; val se evalúa una vez |
| `src/analyze.py` | `--tuning` / `--complexity` / `--per-class` / `--recordings` | agrupa por config para que nunca se promedien corridas incomparables |
| `run_final_ablation.sh` | **el entregable**, 4×3 sobre el split de tres vías | etiqueta `*_3way`, las 12 completas |
| `src/predict_test.py` | inferencia por lotes en test | escribe `preds/*.csv` |
| `src/score_test.py` | evalúa las 12 contra las etiquetas del origen | **reglas preregistradas en el docstring** |
| `src/analyze_test_preds.py` | concordancia entre brazos, ensambles | comprobaciones de consistencia sin etiquetas |
| `run_ablation.sh` | escalera antigua de dos vías | etiqueta `*_stem2`; reemplazada |
| `run_tuning.sh` / `run_confirm.sh` / `run_capacity.sh` / `run_kernel_sweep.sh` | completas, todas salieron | reglas de selección en los encabezados |
| `watch_ladder.sh` | progreso en vivo desde `history.json` | |

**19 tests, todos pasando.** `.venv/bin/python -m pytest tests/ -q` (~4.5 min).

### Tres trampas que ya picaron

1. **Vinculación de valores por defecto en tiempo de import.** `def f(x=C.SOMETHING)` congela el valor al
   importar. Convirtió `--stem-pool 1` en un no-op silencioso una vez. `src/engine.py:145` todavía tiene
   `mixup_p=C.MIXUP_P` de esta forma — inocuo solo porque `train.py` siempre lo pasa
   explícitamente. **No reintroducir el patrón.**
2. **Medir antes de optimizar.** `GRAD_CLIP=1.0` costó ~16%/paso por un no-op.
3. **El máximo sobre épocas está sesgado.** Comparar las mejores épocas de dos corridas favorece a la
   corrida más ruidosa. Esto es lo que hizo que `attr_stem1` pareciera una mejora de +0.018.

### Medido, así que no volver a discutirlo

```
tiempo por época   dual ~190 s, single ~92 s, raw-wide ~255 s (STEM_POOL=2, MPS)
proporción de eval 11.5 s de una época de 185 s = 6.2% -- optimizar eval no tiene sentido
autocast bf16      funciona en MPS pero no gana NADA: dual 1.02x, raw 0.97x (más lento)
batch 64->192      gana 11% pero cambia el LR efectivo -- rechazado, rompe la comparabilidad
num_workers=0      es el medido como mejor (el loader es 38x más rápido que el modelo)
FLOPs              STEM_POOL=1 -> 11.216 G ; STEM_POOL=2 -> 2.792 G (4.02x)
MPS                sin reproducibilidad bit a bit; channels_last falla en el backward; sin fp64
```

---

## 8. HECHOS QUE CONVIENE NO VOLVER A DERIVAR

```
corpus            62,191 clips, 42 especies, 1,074 grabaciones, 4 sitios, 3.0 s @ 22.05 kHz
desbalance        3.59% de celdas positivas, 26.8:1 neg:pos, 36.2% de clips todo-negativos
                  la prevalencia abarca 4 órdenes de magnitud (SPHSUR 21.3% -> LEPFLA 0.011%)
clases de eval    34 de 42 (>=100 positivos); 2 con cero positivos, 6 ultra-raras excluidas
modelo            15,824,344 params dual / 7,983,212 single / 15,777,394 raw-wide
params bloque 4   5,900,288 de 7,807,680 por flujo = 75.6%
proporción armónica  crece con k: 0.383/0.394/0.410/0.425/0.456 para k=5/9/13/17/25
números del paper 12.4M params / 2.86 GFLOPs -- AMBOS internamente inconsistentes
```


---

---

## 9. REANUDAR EL TRABAJO — PEGAR ESTO EN UNA SESIÓN NUEVA

> Reanudando la réplica de HSPP/AnuraSet en
> `/Users/jent/UTEC/ciclo5/deepLearning/paper1`.
>
> **El proyecto está COMPLETO.** Nada está corriendo; las 53 corridas terminaron y el conjunto
> de test está evaluado. Léase `RESULTS_REPORT.md` primero (el entregable), luego `HANDOFF.md`
> para el estado. `KEYNOTE_PROPOSAL.md` es la especificación de la presentación. `TUNING.md` tiene los 15
> cambios de entrenamiento — **nótese su encabezado: sigma es 0.017, no 0.0078**.
> `IMPLEMENTATION.md` es la referencia completa; consúltese por sección, no leer las 1,691
> líneas.
>
> **El hallazgo:** el doble flujo HPSS empata con un baseline de flujo único de mel crudo con 1.98x los
> parámetros — en dev (p=1.000), val, y el conjunto de test del origen (p=0.750, signo
> invertido) — más un control de capacidad igualada (p=0.500), un barrido de kernel nulo, y
> 99.63% de concordancia de decisiones por clip entre dual y raw. La fusión escalar del paper
> es un defecto genuino que el alpha por clase corrige, a costa de -0.0035 mAP (una corrección
> de exactitud, no una ganancia). MixUp 0.7 es la única mejora de tuning confirmada del proyecto.
>
> **No hay ningún paso siguiente obligatorio.** Los seguimientos opcionales están en HANDOFF §1, priorizados.
>
> Reglas de base que siguen vigentes:
> - **Seleccionar por `dev_mAP`.** Test se evaluó UNA VEZ al final con umbrales ajustados en
>   dev y no debe convertirse en una segunda superficie de selección. No reajustar
>   umbrales en test; no ordenar los brazos por test después de haberlo visto.
> - **sigma es 0.017** en el split de tres vías. El 0.0078 antiguo venía de la escalera de dos
>   vías y subestima la varianza en 2.2x.
> - **n=3 limita la p de permutación bilateral a 0.250.** Nada aquí puede alcanzar p<0.05. Decir
>   "consistente entre semillas" o "un empate" — nunca "significativo".
> - **Reportar los empates como empates.** dev raw-dual p=1.000; test dual-raw p=0.750; capacidad
>   p=0.500. Todos empates.
> - **Nunca comparar corridas `*_3way` contra corridas `*_stem2`** (split distinto Y
>   métrica de selección distinta) ni contra las corridas de capacidad con `mixup_p=0.5`.
> - **Ninguna decisión de arquitectura justificada por inspección de AP por clase.** Tres ideas ya
>   se rechazaron sobre esta base.
> - Preregistrar las reglas de selección en el encabezado del script, y **delimitar los desempates a
>   niveles del mismo factor** (ese error se cometió una vez — ver TUNING.md #5).
> - Mantener `RESULTS_REPORT.md`, `KEYNOTE_PROPOSAL.md`, `HANDOFF.md`, `TUNING.md`,
>   `IMPLEMENTATION.md` al día. Todos están intencionalmente **sin commitear** — no
>   commitearlos.
> - Correr un trabajo a la vez; un solo dispositivo MPS, las corridas concurrentes compiten.
> - **El disco está al 95% (21 GB).** Recuperable: `data/test/` 3.9 GB, `test.7z` 3.3 GB,
>   `preds/` 238 MB. Conservar `data/features_test.h5` (2.9 GB) para reevaluar test.
