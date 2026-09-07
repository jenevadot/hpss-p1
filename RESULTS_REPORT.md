# REPORTE DE RESULTADOS — Réplica de HSPP en AnuraSet

**Generado:** 2026-09-05 20:15
**Fuente:** 12 corridas, `runs/{dual,raw,harmonic,percussive}_s{42,43,44}_3way/summary.json`
**Firma de configuración:** verificada como idéntica en las 12 corridas (`mixup_p=0.7`,
`stem_pool=2`, `alpha_tau=1.0`, `alpha_lr_mult=1.0`, `grad_clip=0.0`,
`lr_schedule=cosine`, `selection_split=dev`, `width=1.0`)

---

## 1. TITULAR

> En detección multietiqueta de anuros, la arquitectura de doble flujo HPSS del paper
> **no es mejor que una CNN de flujo único sobre el espectrograma mel sin descomponer**
> (mAP en dev 0.7241 ± 0.0027 vs 0.7278 ± 0.0114; p emparejada=1.000 — un empate), usando
> **1.98× los parámetros** (15,824,344 vs 7,983,212).
>
> **Confirmado en el conjunto de test del origen** (31,187 clips, 538 grabaciones no vistas, etiquetas
> de `AnuraSet_v1.0.0/metadata.csv`, evaluado una vez con umbrales ajustados en dev):
> dual 0.6988 ± 0.0095 vs raw 0.6925 ± 0.0082, **p=0.750 — un empate**, con el signo
> invertido respecto a dev. Dos modelos indistinguibles a través de tres conjuntos de evaluación.
>
> Cinco líneas de evidencia independientes coinciden: la ablación directa de 4 brazos, el conjunto de
> test del origen, un control de capacidad con parámetros igualados (raw-wide ≥ dual, p=0.500), un barrido
> del kernel de HPSS abarcando 116–580 ms (nulo, dispersión 0.0150 < σ 0.0172), y la concordancia de
> predicciones por clip (dual vs raw **99.63%** de las celdas, r=0.979). Como
> `X_h + X_p = X_raw` exactamente (verificado con error de 1.5e-05), HPSS añade **cero información** —
> así que no hay mecanismo por el cual pudiera ayudar más allá del sesgo inductivo, y ninguna
> evidencia de que lo haga.
>
> La **fusión de escalar único del paper es un defecto genuino**: hizo que el modelo dual
> puntuara *por debajo de su propio mejor flujo de entrada* (0.6541 vs 0.6886). La fusión por clase
> a nivel de logits (+41 parámetros) elimina esa patología y es reproducible
> entre semillas (r=0.939) — pero cuesta **−0.0035 mAP**, dentro del ruido. Una corrección
> de exactitud, no una ganancia de rendimiento.

---

## 2. LA TABLA DE ABLACIÓN (el entregable)

4 brazos × 3 semillas, split de tres vías agrupado, **seleccionado en dev**, val leído **una vez** con
umbrales ajustados en dev.

| arm | s42 | s43 | s44 | **dev_mAP** | val_mAP | macro-F1 en val | params |
|---|---|---|---|---|---|---|---|
| raw | 0.7253 | 0.7403 | 0.7178 | **0.7278 ± 0.0114** | 0.7081 ± 0.0109 | 0.6362 ± 0.0028 | 7,983,212 |
| dual | 0.7266 | 0.7212 | 0.7244 | **0.7241 ± 0.0027** | 0.7185 ± 0.0071 | 0.6409 ± 0.0134 | 15,824,344 |
| percussive | 0.7198 | 0.7073 | 0.7250 | **0.7174 ± 0.0091** | 0.6946 ± 0.0114 | 0.6273 ± 0.0093 | 7,983,212 |
| harmonic | 0.6605 | 0.6632 | 0.6701 | **0.6646 ± 0.0049** | 0.6687 ± 0.0015 | 0.5911 ± 0.0099 | 7,983,212 |

### Pruebas por pares — permutación exacta de inversión de signos emparejada sobre dev_mAP

| comparación | dif. media | p | semillas ganadas | dif. por semilla |
|---|---|---|---|---|
| raw − dual | +0.0037 | **1.000** | **1/3** | −0.0013, +0.0191, −0.0066 |
| dual − percussive | +0.0067 | 0.500 | 2/3 | +0.0068, +0.0139, −0.0006 |
| raw − percussive | +0.0104 | 0.750 | 2/3 | +0.0055, +0.0330, −0.0072 |
| dual − harmonic | +0.0595 | **0.250** | **3/3** | +0.0662, +0.0579, +0.0543 |
| percussive − harmonic | +0.0528 | **0.250** | **3/3** | +0.0594, +0.0440, +0.0549 |
| raw − harmonic | +0.0632 | **0.250** | **3/3** | +0.0649, +0.0770, +0.0478 |

**Con n=3 el piso de la permutación es p=0.250.** Nada en este proyecto puede alcanzar
p<0.05. Léase p=0.250 como "consistente en las tres semillas" — la afirmación más fuerte
disponible — y p≥0.500 como "un empate".

---

## 3. LO QUE ESTÁ ESTABLECIDO

### 3.1 dual vs raw es un empate, y la media *favorece a raw* — pero léase con cuidado

`raw − dual = +0.0037, p=1.000`. Esta es la p máxima que la prueba puede retornar.

**Matiz importante que una media sola oculta: raw gana solo 1 de 3 semillas.** Dual gana
las semillas 42 y 44; raw gana la semilla 43 por un amplio +0.0191, que arrastra la media hacia lo positivo.
Así que *no* es correcto decir "raw es mejor" — el registro por semilla es 2–1 a favor
de dual mientras que la media se inclina en el otro sentido. **La única afirmación defendible es que
los dos son indistinguibles.**

Ambos encuadres deben aparecer juntos o el resultado queda mal reportado.

### 3.2 La fusión supera a sus propios flujos de entrada — la corrección por clase funciona

- dual − harmonic **+0.0595, 3/3 semillas, p=0.250**
- dual − percussive +0.0067, 2/3 semillas, p=0.500

Con el α escalar del paper, dual puntuaba **por debajo** de solo-percusivo (0.6541 vs 0.6886).
Con α por clase está por encima de ambos. **El mecanismo de fusión ahora es correcto.** Simplemente
no supera a omitir HPSS por completo.

### 3.3 Harmonic es el flujo más débil por un margen amplio y consistente

harmonic 0.6646 está **−0.0632 por debajo de raw (3/3 semillas)** y −0.0528 por debajo de percussive
(3/3). Ambos p=0.250. Este es el efecto más grande y más reproducible de la tabla.

Consistente con la división de energía medida: con `HPSS_KERNEL=17` solo el **~42%** de
la energía espectral llega al flujo armónico. Los cantos de anuros son en gran medida
de banda ancha/pulsátiles, así que la componente percusiva lleva la mayor parte de la señal
discriminativa.

### 3.4 La ventaja real de dual es la varianza, no la exactitud

| arm | sd en dev | brecha dev→val |
|---|---|---|
| **dual** | **0.0027** | **−0.0056** |
| harmonic | 0.0049 | +0.0041 |
| percussive | 0.0091 | −0.0228 |
| raw | 0.0114 | −0.0198 |

Dual es **4.2× más estable entre semillas que raw** (0.0027 vs 0.0114) y tiene la
menor brecha dev→val de los tres brazos fuertes (−0.0056 vs −0.0198). También tiene el
mejor val_mAP (0.7185 vs raw 0.7081) y macro-F1 en val (0.6409 vs 0.6362) pese a la
media más baja en dev.

**Este es un hallazgo secundario legítimo y debe etiquetarse como tal.** dev es la
métrica de selección preregistrada; val se reporta una vez por transparencia y **no** se
usó para elegir. Ordenar los brazos por val_mAP después de haberlo visto sería exactamente el
error de selección que el split de tres vías existe para prevenir. Formúlese como *"dual es más
estable y muestra una menor brecha de generalización"*, nunca como *"dual gana en val"*.

---

## 4. EXPERIMENTOS CORROBORANTES

### 4.1 Control de capacidad — parámetros igualados

Ensanchando un solo flujo raw al tamaño de dual (`--width 1.41` → 15,777,394 params,
dentro del 0.3% de 15,824,344):

```
              s42     s43     s44      media    sd
raw-wide    0.7219  0.7126  0.7341   0.7229   0.0108
dual        0.7151  0.7180  0.6869   0.7067   0.0172
                    dif. media +0.0162   p=0.500   raw-wide gana 2/3
```

**H2 confirmada:** la descomposición no aporta nada a capacidad igualada. Reportar
como empate (p=0.500); la media está inflada por una semilla (+0.0473) contra el valor atípico
bajo de dual (0.6869).

*Nota: estas corridas usaron `mixup_p=0.5` (previo a la adopción), así que son internamente
comparables pero no directamente comparables con §2.*

### 4.2 Barrido del kernel de HPSS — nulo

```
k     span     dev_mAP   vs k17
25    580 ms   0.7288    +0.0090
5     116 ms   0.7219    +0.0021
17    394 ms   0.7198      —
9     209 ms   0.7192    -0.0006
13    302 ms   0.7138    -0.0060
```

Dispersión **0.0150 < σ 0.0172**; sin tendencia monótona; k=13 queda por debajo de sus dos vecinos.
La predicción preregistrada (kernel más corto gana, curva opuesta a la del paper)
fue **falsificada** — k=25 es nominalmente el mejor. La longitud del kernel no importa de forma medible.

El argumento físico era no obstante sólido: la proporción de energía armónica crece
monótonamente con k (0.383/0.394/0.410/0.425/0.456 para k=5/9/13/17/25), así que el
kernel *sí* desplaza contenido entre flujos. El modelo simplemente es indiferente a ello —
consistente con que no explota la descomposición en absoluto.

### 4.3 Tuning — sobrevivió 1 de 11 intervenciones

| cambio | resultado | veredicto |
|---|---|---|
| **MixUp p 0.5 → 0.7** | **+0.0174, 3/3 semillas, sd 0.0027 vs 0.0172** | **ADOPTADO** |
| temperatura de α 0.3 | +0.0113, 2/3 semillas, p=0.750 | rechazado |
| LR de α ×10 | 0.7012 vs 0.7151 | rechazado |
| AsymmetricLoss | 0.7053, −0.0098 | rechazado |
| 60 épocas | ±0.000 | rechazado |
| MixUp p 0.3 | +0.0001 | rechazado |
| recorte de gradiente | norma 0.132 — nunca se activó, costó 16%/paso | revertido |

### 4.4 Atribución — el +0.109 sobre la config antigua NO es descomponible

```
base          (los cuatro cambios)      dev 0.7151
attr_scalar   (sin fusión por clase)    dev 0.7116   -0.0035
attr_flatlr   (sin LR cosine)           dev 0.7133   -0.0018
attr_stem1    (sin stem pool)           dev 0.7334   +0.018   <- NO ES REAL
AdamW                                                nunca ablacionado
```

El +0.018 de `attr_stem1` es un **pico de una sola época en la época 13**: la corrida nunca superó
ese valor en las 13 épocas siguientes y pasó buena parte de ellas *por debajo* de base.
Seleccionar el máximo sobre 40 épocas está sesgado hacia arriba para la secuencia más ruidosa.

Cada remoción individual cuesta ~nada, AdamW nunca se probó, y la comparación con la config
antigua abarca tanto un cambio de split (dos vías → tres vías) como un cambio de métrica
(val → dev). **MixUp 0.7 es la única mejora confirmada del proyecto.** Todo lo demás
en la configuración es ingeniería defendible, no ganancia demostrada.

---

## 5. MÉTODO

```
corpus      62,191 clips / 42 especies / 1,074 grabaciones / 4 sitios / 3.0 s @ 22.05 kHz
desbalance  3.59% de celdas positivas, 26.8:1 neg:pos, 36.2% de clips todo-negativos
            la prevalencia abarca 4 órdenes de magnitud (SPHSUR 21.3% -> LEPFLA 0.011%)
eval        34 de 42 clases (>=100 positivos); 2 con cero positivos, 6 ultra-raras excluidas

split       train 44,408 / dev 8,932 / val 8,851 clips
            767 / 154 / 153 grabaciones, AGRUPADO POR GRABACIÓN (los clips de una
            misma grabación son casi duplicados; un split aleatorio filtra)
            dev: early stopping, selección de checkpoint, ajuste de umbrales
            val: leído UNA VEZ al final con umbrales ajustados en dev
```

El paquete provisto contenía 31,187 wavs de test **sin CSV de etiquetas**, así que val fue la única
estimación retenida disponible mientras corrían los experimentos, y cada decisión se tomó sobre
dev. Las etiquetas de test se localizaron en el origen *después* (§7.4) y se evaluaron una vez. Sus
538 grabaciones son disjuntas por grupo de las 1,074 grabaciones de entrenamiento.

**Preregistrado antes de correr:** selección solo por `dev_mAP`; ninguna decisión de arquitectura
justificada por inspección de AP por clase; todos los brazos reportados incluyendo los perdedores.

**σ ≈ 0.017 en este split.** El 0.0078 que se cita a menudo venía de un split de dos vías
anterior con 17% más datos de entrenamiento y subestima la varianza en 2.2×.

---

## 6. LIMITACIONES

1. **n=3 semillas → piso de permutación p=0.250.** Ningún resultado aquí puede alcanzar p<0.05.
2. **Un solo tipo de grabadora.** El beneficio declarado del paper es la *robustez entre dispositivos*
   en DCASE 2020 Task 1A. AnuraSet usa un solo modelo de grabadora, así que el mecanismo puede estar
   bajo prueba sin su ventaja principal. Esto acota la conclusión; no
   rescata la afirmación, ya que la afirmación nunca se restringió a datos multidispositivo.
3. **Una semilla hace el trabajo pesado en dos lugares.** La semilla 43 de raw (+0.0191) impulsa la
   media raw−dual; la semilla 44 de dual (0.6869) infla tanto el margen del control de capacidad como
   el de MixUp.
4. **Las etiquetas de test se localizaron tarde** (`metadata.csv` del origen), así que dev y val guiaron
   cada decisión y test se evaluó una vez al final. Ese es el orden correcto, pero
   significa que test no jugó ningún papel en la selección del modelo — por diseño.
5. **Cuatro clases están por debajo del azar** (DENCRU AP 0.0096 con 602 positivos). La causa es la
   **diversidad de grabaciones, no el soporte**: AP media 0.350 con ≤2 grabaciones en val vs
   0.764 con más.
6. **`tail_mAP` es NaN, y eso es correcto.** Las 6 especies ultra-raras viven en 1–3
   grabaciones, así que un split disjunto por grupo deja cero positivos en val. Reportar 0.0
   sería fabricar.
7. **MPS no es reproducible bit a bit** (orden de reducción no determinista), que es
   por lo que todo se reporta como media ± sd en lugar de números individuales.
8. **Las propias cifras del paper son internamente inconsistentes.** Los 2.86 GFLOPs son
   aritméticamente inalcanzables a partir de su arquitectura declarada — nuestra implementación
   fiel da 11.216 GFLOPs, lo que implica un downsample temprano de ~/4 no documentado.

---

## 7. INFERENCIA Y EVALUACIÓN EN EL CONJUNTO DE TEST (etiquetas del origen obtenidas)

Los 12 checkpoints se corrieron sobre los clips retenidos de `test.7z`.

```
features   data/features_test.h5  -- 31,187 clips, 2.90 GB, 91 s de extracción
           cero NaN en X_h / X_p / X_raw (las 40 advertencias de desbordamiento fp32 en el
           matmul mel son benignas y se recuperan en el espacio logarítmico; también ocurren
           en los features de entrenamiento)
salidas    preds/{arm}_s{seed}_3way.csv          probabilidades sigmoides por clase
           preds/{arm}_s{seed}_3way_binary.csv   umbrales ajustados en dev aplicados
           preds/ENSEMBLE_{arm}.csv              probabilidades medias de 3 semillas
normalización  el norm.json propio de cada corrida (estadísticas del split de TRAIN). Recalcular
               sobre test sería fuga en tiempo de test.
umbrales       thresholds.npy por corrida (ajustados en dev). Nunca reajustados en test.
```

Las secciones 7.1–7.3 se calcularon **antes** de que las etiquetas estuvieran disponibles y son
comprobaciones puras de consistencia. §7.4 tiene las puntuaciones reales, obtenidas después de localizar
las etiquetas del origen — y confirma la predicción preregistrada.

### 7.1 La calibración se transfiere a grabaciones no vistas

Tasa de positivos predicha en 538 grabaciones disjuntas de todas las 1,074 grabaciones de
entrenamiento, contra la prior de entrenamiento de **0.0359**:

| arm | tasa de positivos del ensamble | vs prior |
|---|---|---|
| dual | 0.0321 | −0.0038 |
| raw | 0.0322 | −0.0037 |
| percussive | 0.0317 | −0.0042 |
| harmonic | 0.0318 | −0.0041 |

Los cuatro caen dentro de 0.004 de la prior en grabaciones completamente no vistas — subpredicción
leve, consistente entre brazos. Las tasas de semilla individual abarcaron 0.0353–0.0394.

### 7.2 La estabilidad entre semillas confirma el orden de varianza del lado de dev

Correlación media por pares de probabilidades entre semillas dentro de cada brazo:

```
dual        0.9641   <- el más estable, coincidiendo con su sd en dev de 0.0027
percussive  0.9554
raw         0.9544
harmonic    0.9486
```

**dual es el brazo más autoconsistente tanto en test como en dev.** Esta es
corroboración independiente de §3.4 a partir de datos que no jugaron ningún papel en la selección — la
única pieza de evidencia genuinamente nueva de esta sección.

### 7.3 Los brazos hacen predicciones casi idénticas

Concordancia entre brazos sobre predicciones binarizadas en 1,309,854 celdas:

```
                  dual      raw   percussive  harmonic
dual            1.0000   0.9792     0.9729    0.9691     <- correlación de probabilidades
raw             0.9792   1.0000     0.9764    0.9634
percussive      0.9729   0.9764     1.0000    0.9525
harmonic        0.9691   0.9634     0.9525    1.0000

concordancia binarizada:  dual vs raw 99.63%   dual vs percussive 99.56%
                          raw vs percussive 99.58%   peor par 99.34%
```

**dual y raw coinciden en el 99.63% de las celdas y correlacionan en r=0.979.** Las dos
arquitecturas no solo puntúan parecido en agregado — están tomando las *mismas
decisiones por clip*.

Esta es la pieza individual de evidencia más fuerte del reporte de que la descomposición HPSS
no está haciendo un trabajo independiente. Un empate en mAP (§3.1) podría en principio ocultar dos modelos
con fortalezas distintas que se cancelan. No lo hace: son casi duplicados
clip por clip. Combinado con `X_h + X_p = X_raw` (HPSS añade cero información), la
lectura coherente es que **ambas redes convergen a sustancialmente la misma función
independientemente de si la entrada está descompuesta.**

Nótese que incluso `harmonic` — 0.0632 por debajo de raw en mAP de dev, la mayor brecha de la
tabla — todavía coincide con dual en el 99.52% de las celdas. Con una tasa de positivos del 3.59% la mayoría de
las celdas son negativos fáciles, así que la alta concordancia es en parte estructural. El *orden* de
las correlaciones es la señal, no su nivel absoluto.

### 7.4 ETIQUETAS DE TEST OBTENIDAS Y EVALUADAS — la predicción preregistrada se sostuvo

`AnuraSet_v1.0.0/metadata.csv` se descargó y **contiene las etiquetas de test.**

**Procedencia verificada antes de calcular cualquier puntuación:**

```
metadata.csv       93,378 filas, columna subset: train=62,191 / test=31,187
                   coincidencia EXACTA con nuestro split local, en ambos lados
columnas de especie idénticas a C.SPECIES, en orden
verificación train todas las 62,191 filas vs nuestro train.csv -> 0 discrepancias de etiquetas
clave de join      "{fname}_{min_t}_{max_t}.wav" -> 31,187/31,187, 0 sin emparejar
prior de test      tasa de positivos 0.0360 (train 0.0359); 36.6% todo-negativos
cobertura          las 34 EVAL_CLASSES tienen positivos en test (mín. 10)
                   4 especies tienen cero (LEPELE, RHISCI, LEPFLA, SCIRIZ) -- ninguna
                   es clase de eval, así que la métrica de 34 clases no se ve afectada
```

**Esto corrige la hipótesis previa sobre la procedencia.** La partición 2:1
por sitio está **definida en el origen por los autores del dataset**, no construida
localmente: la propia columna `subset` de `metadata.csv` reproduce nuestro split train/test
exactamente, en ambos lados. El paquete del curso simplemente entregó `train.csv` filtrado a
`subset=='train'` y omitió las etiquetas de test. Nada se cortó a mano.

**Regla de evaluación, preregistrada en `src/score_test.py` antes de correr:** los 12
checkpoints evaluados una vez; mAP sobre las mismas 34 clases de eval; **umbrales ajustados en dev,
nunca reajustados en test**; media ± sd entre semillas; pruebas de permutación de inversión de signos emparejadas.

```
arm             s42      s43      s44     test_mAP media+-sd   test_F1 media+-sd
dual          0.6903   0.6971   0.7090   0.6988 +- 0.0095    0.6668 +- 0.0096
raw           0.7012   0.6916   0.6848   0.6925 +- 0.0082    0.6535 +- 0.0142
percussive    0.6842   0.6753   0.6932   0.6843 +- 0.0090    0.6384 +- 0.0077
harmonic      0.6500   0.6501   0.6475   0.6492 +- 0.0015    0.5897 +- 0.0088
```

```
comparación              dif. media    p    victorias   por semilla
dual - raw                 +0.0063  0.750   2/3   -0.0109 +0.0055 +0.0242
dual - percussive          +0.0145  0.250   3/3   +0.0061 +0.0218 +0.0158
raw - percussive           +0.0083  0.500   2/3   +0.0169 +0.0163 -0.0084
dual - harmonic            +0.0496  0.250   3/3   +0.0403 +0.0470 +0.0616
percussive - harmonic      +0.0351  0.250   3/3   +0.0342 +0.0253 +0.0457
raw - harmonic             +0.0433  0.250   3/3   +0.0511 +0.0416 +0.0373
```

**La predicción preregistrada fue un empate, y test entregó un empate.**
`dual − raw = +0.0063 con p=0.750`, 2/3 semillas, con raw ganando la semilla 42 por −0.0109.
Dev tenía a raw nominalmente adelante (+0.0037, p=1.000); test tiene a dual nominalmente adelante
(+0.0063, p=0.750). **Ambos son empates, y el signo se invierte entre ellos** — que es
exactamente cómo se ven dos modelos indistinguibles a través de dos conjuntos de evaluación.

No reportar esto como "dual gana en test". La dirección se invirtió respecto a dev, ninguno de los dos
resultados se acerca al piso de p=0.250, y la brecha está bien dentro de la sd de test de ~0.009.

**Lo que test confirma limpiamente (3/3 semillas, p=0.250 — el piso):**
- dual > percussive (+0.0145), raw > percussive (+0.0083 con 2/3)
- **los tres brazos fuertes > harmonic** (+0.0351 a +0.0496)

Así que el orden establecido en dev se reproduce en un conjunto genuinamente intacto. Que el
flujo armónico sea por mucho el más débil, y que la fusión supere a sus propias entradas, son los hallazgos
robustos; **dual vs raw es un empate en los tres: dev, val y test.**

**Nivel absoluto.** El mAP de test (~0.69–0.70) queda por debajo de dev (~0.72–0.73) y val
(~0.71) para cada brazo. Esperado: test es 3.5× más grande que val (31,187 vs 8,851
clips), abarca 538 grabaciones no vistas, y usa umbrales ajustados en dev. El
*orden* es lo que se transfiere, no el nivel.

**Estado de esta estimación.** Test se evaluó **una vez**, después de congelar la configuración
y después de que dev/val hubieran resuelto cada decisión. Los umbrales vinieron de dev. Es el
número más limpio del proyecto — y concuerda con dev y val.

### 7.5 Dos hallazgos que solo el conjunto de test podía revelar

**(a) El empate no es uniforme — son diferencias por clase que se compensan.**

AP por clase en test, dual vs raw (ensambles de 3 semillas, 34 clases de eval):

```
raw mejor en 21/34 clases, dual mejor en 13/34
media de |dif. por clase| 0.0287     mediana 0.0176

mayores victorias de dual       mayores victorias de raw
  PHYMAR  +0.1148 (n=117)        ADEDIP  -0.0921 (n=223)
  ELABIC  +0.1111 (n=491)        RHIICT  -0.0593 (n=180)
  PHYNAT  +0.0575 (n=65)         SCIFUV  -0.0535 (n=1395)
  ELAMAT  +0.0500 (n=293)        DENELE  -0.0485 (n=17)
  PHYDIS  +0.0246 (n=324)        SCIALT  -0.0447 (n=10)
```

Así que el empate agregado (+0.0063) oculta oscilaciones por clase de hasta ±0.11 que **se cancelan**.
Raw gana más clases; dual gana por márgenes mayores donde gana. Esta es una calificación genuina
de §7.3: los dos modelos coinciden en el 99.63% de las *decisiones binarias* pero sus
*rankings* difieren de forma significativa para especies individuales.

**Esto no rescata la descomposición, y no debe presentarse como si lo hiciera.**
Nada aquí identifica *cuáles* especies se benefician por adelantado — el patrón por clase es
visible solo después de evaluar test, y usarlo para justificar una arquitectura sería exactamente
el ajuste a val/test que la disciplina de splits prohíbe. Es una observación sobre
por qué los empates agregados pueden ser engañosos, no una ruta hacia un mejor modelo.

**(b) El ensamblado por semillas supera cualquier diferencia arquitectónica del proyecto.**

Promediando las probabilidades de las 3 semillas por brazo, evaluado en test:

```
arm          ensamble 3 semillas   media semilla única    ganancia
raw               0.7240           0.6925        +0.0315
harmonic          0.6785           0.6492        +0.0293
percussive        0.7122           0.6843        +0.0279
dual              0.7204           0.6988        +0.0216

dual + raw fusionados  0.7329    (mejor brazo individual 0.6988)   +0.0341
```

**Cada ganancia de ensamblado (+0.022 a +0.032) es mayor que cada diferencia
arquitectónica medida en cualquier parte de este proyecto** — mayor que dual−raw (+0.0063),
mayor que MixUp 0.7 (+0.0174), mayor que dual−percussive (+0.0145). Fusionar dual
y raw alcanza 0.7329, superando al mejor modelo individual por +0.0341.

La implicación práctica es contundente: en esta tarea, **tres semillas del modelo de flujo único
más barato, promediadas, superan a cualquier modelo individual de cualquiera de las dos arquitecturas** —
con un costo de entrenamiento de 3 × 7.98M params pero solo el costo de inferencia de flujo único por miembro,
frente a los 15.8M de dual. Si el objetivo es exactitud por unidad de esfuerzo de ingeniería, el
promedio entre semillas domina la elección arquitectónica aquí.

Advertencia: los ensambles se evaluaron en test con umbrales ajustados en dev, y la comparación de
ensamblado **no** fue preregistrada — se calculó después de que el resultado principal quedara
resuelto. Trátese como una observación bien respaldada para trabajo futuro, no como una
hipótesis que este proyecto probó.

---

## 8. CONCLUSIÓN

La réplica es fiel (arquitectura sin cambios: tipos de capa, orden de bloques,
anchos de canales, pares asimétricos 1×n / n×1) y el resultado negativo es robusto a través de
tres pruebas independientes.

**Lo que el mecanismo del paper no compra aquí:** exactitud. dual empata con raw en dev
(p=1.000), en val, y en el conjunto de test del origen (p=0.750, signo invertido) con 1.98x los
parámetros; empata de nuevo cuando raw se ensancha para igualarlo; es insensible al kernel de HPSS
que controla la descomposición; y — en 31,187 clips de test no vistos — **toma
la misma decisión binaria que raw en el 99.63% de las celdas (r=0.979).** Las dos arquitecturas
convergen a sustancialmente la misma función.

**Lo que test SÍ confirma, 3/3 semillas en el piso de p=0.250:** los tres brazos fuertes superan a
harmonic (+0.0351 a +0.0496), y dual supera a percussive (+0.0145). El orden
establecido en dev se reproduce en un conjunto intacto -- así que el resultado negativo es sobre
dual vs raw específicamente, no un fracaso en medir absolutamente nada.

**Lo que sí compra:** estabilidad. La sd entre semillas de dual es 4.2× menor que la de raw con
la menor brecha dev→val entre los brazos fuertes, y también es el brazo más
autoconsistente entre semillas en test (r=0.964 vs raw 0.954) — corroborado con
datos que no jugaron ningún papel en la selección. Una observación secundaria, no una
victoria en la métrica de selección.

**Contribuciones metodológicas:** (a) fusión por clase a nivel de logits, que elimina un
defecto real del diseño publicado; (b) un diseño reutilizable de control de capacidad para cualquier
afirmación de dos flujos cuyos flujos sumen la entrada original.

**Lo que no debe afirmarse:** que HPSS funciona en esta tarea, que la fusión por clase
mejora la exactitud, o que alguna diferencia aquí es estadísticamente significativa.
