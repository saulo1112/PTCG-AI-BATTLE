# M42 — los contextos de decisión sin modelo: dos mejoras reales, medidas (2026-08-09)

**Resumen: tras nueve palancas cerradas contra la fidelidad del ranking de MAIN, una auditoría de una
superficie distinta —qué contextos de decisión aprende el clon siquiera— encontró que el 12,2% de
todas las decisiones (23.354 de 190.731) no tienen modelo alguno y caen a greedy. Dos de ellas
mostraban una divergencia SATURADA AL 100% contra el maestro, verificada en nuestras propias
repeticiones del ladder. Ambas están arregladas, con puertas pre-registradas superadas y verificación
conductual sobre el artefacto embarcado. Es la primera mejora medida del clon desde M28.**

Continúa [m41_findings.md](m41_findings.md) (mazo, negativo) y [m40_findings.md](m40_findings.md)
(última hipótesis causal de política, falsificada).

---

## 1. Por qué esta superficie y no otra

M31/M36/M37/M38/M39 atacaron todas lo mismo: la fidelidad del scorer de ranking de MAIN (anchura,
datos, arquitectura, RL, ponderación). M34/M40 probaron las dos causas raíz de M33. Todas nulas o
negativas. Lo que nadie había mirado nunca es **el reparto de contextos**: el payload embarcado
aprende 10 contextos, y el juego tiene muchos más.

Auditoría sobre las 2.330 partidas de Yushin (**190.731 decisiones**, extractor verificado
`iter_player_decisions`):

| contexto | filas | /partida | ¿modelo? |
|---|---|---|---|
| MAIN | 114.048 | 48,9 | MLP ensemble |
| TO_HAND | 28.964 | 12,4 | lineal |
| **ACTIVATE** | **14.750** | **6,3** | **ninguno → greedy** |
| TO_ACTIVE / TO_BENCH / SWITCH | 19.380 | 8,3 | lineal |
| DISCARD_ENERGY, EVOLVE, IS_FIRST, **SETUP_BENCH_POKEMON**, SKILL_ORDER, DRAW_COUNT | 6.081 | 2,6 | **ninguno → greedy** |

Las decisiones sin modelo caen a `GreedyPolicy`, y las más frecuentes a `_safe_default`
([greedy.py:339-345](../src/ptcg_ai/decision/greedy.py#L339-L345)), que devuelve
`list(range(minCount))` — la opción 0, **sin mirar el estado**.

**Por qué nunca se entrenaron:** [`train.py:284`](../src/ptcg_ai/imitation/train.py#L284) descarta un
contexto salvo que supere a greedy por `_MIN_LIFT = 0,05`. Con greedy ya en 0,934 (ACTIVATE) la barra
exige 0,984 — **inalcanzable por aritmética**, no porque el contexto no importe. Es un artefacto de
umbral que llevaba desde M7 escondiendo el tercer contexto más frecuente del juego.

## 2. Las dos divergencias saturadas

Medidas con el mismo extractor sobre el maestro y sobre nuestras propias repeticiones del ladder
(`replays/55145833`):

| contexto | /partida | maestro rechaza | **clon rechaza** | mecanismo |
|---|---|---|---|---|
| ACTIVATE | 6,55 | 6,6% | **0 de 760** | `_safe_default` → siempre `[0]` = SÍ |
| SETUP_BENCH_POKEMON | 0,45 | 37,4% | **0 de 45** | [greedy.py:131-133](../src/ptcg_ai/decision/greedy.py#L131-L133) → siempre el tope |

P(0/760 si p=0,066) ≈ 3e-23. P(0/45 si p=0,374) ≈ 1e-9. No es ruido: es **imposibilidad estructural**.

> **Nota de método que casi cuesta el milestone.** Un primer barrido de repeticiones crudas leyó
> `cell['action']` del mismo paso y dio 75,8% de rechazos para el maestro — **el resultado contrario**.
> El extractor real ([kaggle_replay.py:97](../src/ptcg_ai/imitation/kaggle_replay.py#L97)) toma la
> acción de `nxt[seat]`, **el paso siguiente**: en la repetición cruda, la acción que responde a la
> observación del paso *i* está guardada en el paso *i+1*. Todas las cifras de este documento salen del
> extractor correcto, y se detectó porque contradecían el dataset de entrenamiento ya construido. Los
> logs defectuosos se borraron para que nadie los cite. **Regla:** al leer repeticiones crudas, cruzar
> siempre contra `build_decision_dataset` antes de creerse una cifra nueva.

## 3. Brazo A — cabeza ACTIVATE

`ACTIVATE` es "¿usas esta habilidad opcional?": Psychic Draw (Kadabra/Alakazam al evolucionar), Run
Away Draw (Dudunsparce — **se baraja de vuelta al mazo si robas**) y Flip the Script (Fezandipiti ex).

Un softmax de 2 opciones **es** una regresión logística sobre (x_SÍ − x_NO), así que la aprendibilidad
se puede comprobar directo: de las 60 features que difieren entre SÍ y NO, **58 varían con el estado**
— no es la trampa de M34, donde el bloque se cancelaba en el softmax.

**G-A1 (partición 3-vías por partida, semillas 0/1, TEST nunca visto, n=2.989):**

| | precisión |
|---|---|
| siempre-SÍ (lo embarcado) | 0,9408 |
| lineal | 0,9675 |
| **MLP ensemble (k=3, h=48)** | **0,9742** |
| | **lift +0,0335** (barra +0,020) |

**G-A2 conductual** — `analyze_activate_divergence.py`, la puerta que faltó en siete fracasos
anteriores:

| política | acuerdo | rechazos | tasa | recall | falsos-NO |
|---|---|---|---|---|---|
| campeón | 0,9408 | **0/2989** | 0,0000 | 0,000 | 0,0000 |
| candidato | 0,9742 | 146 | 0,0488 | **0,695** | 0,0082 |

El maestro rechaza 5,92%. El candidato rechaza 4,88%, atrapando el **69,5%** de sus rechazos con un
0,8% de falsos. `bc_failures = 0`.

**Cero cambios en `src/`**: [`policy.py:392`](../src/ptcg_ai/imitation/policy.py#L392) hace
`self._weights.get(ctx)`, así que cualquier clave presente en el payload se enruta sola. Mismo patrón
que M34 ya embarcó.

## 4. Brazo B — cabeza de conteo (el hallazgo más nítido)

`_top_k` toma `min(maxCount, n)` — la regla del maestro en todos los contextos **menos uno**. En
SETUP_BENCH_POKEMON el maestro rechaza el 37,4% de las veces, y **ningún modelo de ranking puede
expresarlo a ninguna capacidad**: el conteo no es una propiedad del orden.

**El mecanismo, medido antes de construir nada.** Todas estas decisiones ocurren en el turno 0, con
las cartas ofrecidas de una en una. La regla de Yushin es determinista:

| carta ofrecida | la puso | la dejó | tasa |
|---|---|---|---|
| **Abra** (arranque del combo) | 673 | 0 | **1,000** |
| **Dunsparce** (motor de robo) | 146 | 0 | **1,000** |
| **Fezandipiti ex** | 0 | 261 | **0,000** |
| **Shaymin** | 1 | 210 | **0,005** |

`Fezandipiti ex` es un **ex: vale 2 cartas de premio** al ser noqueado. Yushin nunca lo pone en banca
en el setup. **Nuestro campeón lo ponía siempre**, regalando un objetivo de 2 premios en un mazo cuyas
derrotas son 88,5% carreras de premios (M33). Eso explica por qué el número es grande: era una regla
determinista que violábamos el 100% de las veces.

**G-B1 (TEST n=204, extremo a extremo: rankear con la cabeza de ranking, tomar tantas como diga la de
conteo):**

| | precisión |
|---|---|
| greedy (lo embarcado) | 0,4706 |
| cabeza de ranking @ tope | 0,4706 |
| **ranking @ cabeza de conteo** | **0,9804** |
| | **lift +0,5098** (barra +0,050) |

Rechazos: maestro 37,7%, candidato 38,7%.

**Verificación conductual sobre 204 decisiones held-out, a través de la política real:**

| carta | campeón | candidato | maestro |
|---|---|---|---|
| Abra | 1,000 | 0,975 | 1,000 |
| Dunsparce | 1,000 | 1,000 | 1,000 |
| **Fezandipiti ex** | **1,000** | **0,000** | 0,000 |
| **Shaymin** | **1,000** | **0,000** | 0,005 |

**Único cambio en `src/`** ([policy.py](../src/ptcg_ai/imitation/policy.py)): `count_heads` en el
payload, **ausente por defecto** ⇒ todo agente existente byte-idéntico (patrón `recover_rule` de M27).
La predicción se **satura** a `[minCount, min(maxCount, n)]`, así que una cabeza que se equivoque
nunca puede emitir una jugada ilegal; `_legalize` sigue siendo la última red. 8 tests nuevos en
`tests/unit/test_imitation_count_head.py` fijan exactamente ese contrato.

## 5. Brazo C — MLP ensemble para TO_HAND

TO_HAND es el contexto #2 (12,4 decisiones/partida, 28.964 filas = 76% del volumen de MAIN) y seguía
siendo **lineal**: M34 le dio features nuevas (`ALAKAZAM_FETCH`, dim 750) pero nunca capacidad. M28
midió el salto lineal→MLP en MAIN en +0,224. Maquinaria de M13 reutilizada tal cual
(`train_mlp_v4._expand_stages`: un k-pick se factoriza en k etapas Plackett-Luce).

**G-C1 (TEST n=4.212):**

| | precisión |
|---|---|
| greedy | 0,5209 |
| lineal embarcado (baseline) | 0,7517 |
| **MLP ensemble (k=3, h=48)** | **0,7949** |
| | **lift +0,0432** (barra +0,020) |

Dos matices honestos: **(a)** el baseline embarcado se entrenó con el 80% de las partidas y el
candidato solo con el 60% (partición 3-vías), así que la comparación **perjudica** al candidato — sin
fuga, porque el TEST es el mismo 20% que el modelo embarcado nunca vio. **(b)** el lineal reentrenado
en esa partición más pequeña saca 0,7172, es decir la reducción de datos cuesta ~0,035; el MLP los
recupera y añade. **No hay firma de sobreajuste**: TEST (0,7949) > val (0,7726), lo contrario del
patrón M11/M14.

**G-C2 no se corrió como estaba planeado.** `cv_archetype_tohand.py` está cableado al mazo TR_650 de
M13 y adaptarlo no cabía en el plazo. En su lugar, el papel de "¿generaliza fuera de la distribución?"
lo hace el gauntlet de 72 mazos distintos de §6, que es un test OOD más duro. Queda anotado como
sustitución deliberada, no como puerta silenciada.

## 6. Puerta de daño (G-A3)

`head_to_head_par.py`, 72 mazos × 12 partidas × 2 brazos, campo pilotado por greedy:

| corte | candidato | campeón | delta |
|---|---|---|---|
| macro | 0,963 | 0,954 | +0,009 [90% −0,003, +0,021] empate |
| solo Grimmsnarl | 0,991 | 0,972 | +0,019 [−0,009, +0,046] empate |
| sin Grimmsnarl | 0,959 | 0,951 | +0,008 empate |
| ponderado por frecuencia | 0,978 | 0,954 | **+0,024** |

**0 intervenciones en 1.728 partidas.** El gauntlet está saturado (0,95-0,96) y **es un veto, no un
promotor**: "pasa" significa *sin evidencia de daño*. Los cuatro cortes apuntan en positivo, lo cual
es consistente pero no concluyente por sí solo.

**El mismo veto sobre A+B+C** (72 mazos × 12 × 2):

| corte | A+B+C | campeón | delta |
|---|---|---|---|
| macro | 0,964 | 0,948 | **+0,016 [90% +0,005, +0,028]** CI enteramente positivo |
| sin Grimmsnarl | 0,962 | 0,942 | **+0,020 [+0,007, +0,033]** positivo |
| solo Grimmsnarl | 0,981 | 0,991 | −0,009 (empate, 0W/1L sobre 9 mazos) |
| ponderado por frecuencia | 0,962 | 0,969 | **−0,006** |

**A+B+C gana en macro pero pierde en el corte ponderado por frecuencia, donde A+B ganaba (+0,024).**
Ese corte pesa por episodios reales, así que Grimmsnarl (30,8% del campo) lo domina — pero el corte
Grimmsnarl es de 9 mazos con 0W/1L, o sea prácticamente sin información. **La conclusión honesta es
que este instrumento no distingue A+B de A+B+C**: ninguno queda vetado, y ordenarlos con él sería
exactamente el error que M25 documentó (el gauntlet es un veto, no un ranker, en la banda ±0,05).

## 7. La arena directa (el instrumento que sí discrimina)

`scratchpad/test_ctx_heads.py` — los dos payloads juegan **entre sí** con el mismo mazo, así que lo
único que cambia en todo el experimento es el archivo de pesos, y el espejo vale 0,500 por
construcción. n=600 por la lección de M41.

| enfrentamiento | resultado | referencia | delta |
|---|---|---|---|
| **candidato (A+B) vs campeón** | **0,535** (321W-279L) | 0,500 | **+0,035** |
| campeón vs campeón (**control de ruido**) | **0,492** (295W-305L) | 0,500 | −0,008 |
| vs grimmsnarl: candidato | 0,640 | 0,655 (campeón) | −0,015 |

**Dos comprobaciones del instrumento, ambas pasan.** (1) El control da 0,492 contra un 0,500 teórico,
a 0,8 puntos — compárese con M41, donde el mismo control leyó 0,435 a n=200; a n=600 el suelo de ruido
sí está calibrado. (2) El campeón contra Grimmsnarl da **0,655** aquí y M41 midió **0,658** de forma
independiente: reproducibilidad a 0,3 puntos.

**Lectura honesta, incluido lo que va en contra:**

- El +0,035 del espejo directo: z = 1,71 contra la nula de 0,500, **p unilateral ≈ 0,043**. Sugerente
  al 90%, **NO concluyente al 95%**. No es prueba de una ganancia de elo.
- El −0,015 contra Grimmsnarl **va en contra del candidato**. No es significativo (SE de la diferencia
  ≈ 0,028, así que −0,015 ± 0,055 cruza el cero con holgura), y es la misma dirección que el corte
  ponderado por frecuencia del gauntlet A+B+C. Es el punto débil del resultado y queda anotado: si
  hubiera que apostar dónde puede fallar esto en el ladder real, es aquí.

## 8. Lo que este milestone NO afirma

- **No promete elo.** M34 movió TO_HAND +0,114 de fidelidad, verificó el cambio de comportamiento y no
  movió ni el ensamblaje ni el ladder. Estos cambios son más pequeños en masa de decisiones
  (~0,45/partida corregidas frente a las ~19,6 de desacuerdo que carga MAIN).
- **Lo que sí tienen que las nueve palancas anteriores no tenían:** no son un refinamiento de
  fidelidad, son la eliminación de una **ceguera estructural**. El campeón no podía rechazar *nunca*,
  en ningún estado. Y en el caso de SETUP_BENCH la jugada corregida tiene una justificación de reglas
  directa (no regalar un objetivo de 2 premios), no una estadística difusa.

## 9. Qué subir

Tres artefactos construidos y verificados extraídos, ninguno subido:

| bundle | contenido | evidencia |
|---|---|---|
| `build/imitation-ctx-a.tar.gz` | solo brazo A | G-A1 +0,0335, G-A2 pasa |
| **`build/imitation-ctx.tar.gz`** | **A + B** | + arena directa **0,535** (control 0,492), gauntlet: los 4 cortes positivos |
| `build/imitation-ctx-abc.tar.gz` | A + B + C | gauntlet macro +0,016 (CI positivo) pero ponderado −0,006; sin arena directa |

**Recomendación: `imitation-ctx.tar.gz` (A+B).** Razones, en orden:

1. Es el único con evidencia de **dos instrumentos independientes** apuntando igual (arena directa
   +0,035 con control calibrado; gauntlet positivo en los 4 cortes).
2. Sus dos arreglos son **eliminación de ceguera estructural**, no refinamiento de fidelidad — y el de
   SETUP_BENCH tiene justificación de reglas directa (no regalar un objetivo de 2 premios), no una
   estadística difusa. Las nueve palancas fallidas eran todas del otro tipo.
3. Superficie de cambio menor ⇒ menos riesgo. MAIN queda **byte-idéntico** (sha verificado).

A+B+C queda construido y gateado por si se quiere el brazo de mayor fidelidad; no se recomienda de
entrada porque su único instrumento discriminante disponible (el corte ponderado por frecuencia) va en
contra, y no dio tiempo a correrle la arena directa.

**El coste marginal de subir es cero.** Toda subida reinicia el rating a 600 y descarta las partidas
acumuladas, da igual qué archivo se suba; el plan previo ya contemplaba re-subir el MISMO archivo
apostando a un mejor sorteo de rivales. Subir A+B en vez de una copia del campeón cuesta exactamente
lo mismo y lleva una mejora conductual medida y ningún daño medido.

**Antes de subir:** el bundle ya está verificado extraído con `verify_ctx_bundle.py` (mazo 60/60
correcto, `_IMITATION_READY: True`, `bc_failures: 0`, y la cabeza nueva disparando de verdad sobre
observaciones reales). 250 tests pasan.

## 10. Reutilizable

- **`scratchpad/train_context_heads.py`** — entrenador para CUALQUIER contexto (lineal + MLP,
  single- y multi-pick vía expansión en etapas). Generaliza el `{"MAIN"}` fijo de
  `train_mlp_alakazam.py`. Su baseline es **el artefacto embarcado** para ese contexto (la regla de
  M37), y mide con el desempate real de `policy._rank_order`, no con `np.argsort` — los dos difieren
  materialmente en contextos con opciones duplicadas (TO_HAND: 0,7172 vs 0,6835).
- **`count_heads`** — mecanismo general para cualquier contexto donde el maestro elige *cuántas*, no
  solo *cuáles*. SETUP_BENCH_POKEMON era el único con divergencia medida, pero el mecanismo es
  reutilizable.
- **`scratchpad/build_fetch_weights.py`** — ahora injerta specs dict con perfil por contexto y
  cabezas de conteo, y el guardián sha256 cubre **todos** los contextos, no solo MAIN. Verificado:
  la ruta M34 sigue reproduciendo el campeón embarcado **byte a byte**.
- **`scratchpad/verify_ctx_bundle.py`** — conduce el entrypoint EMBARCADO con observaciones reales y
  falla si la cabeza nueva no se dispara. Existe porque en M37 `validate_submission` y
  `smoke_test_entrypoint` pasaron ambos sobre un bundle con el mazo equivocado, y costó una subida.
- **La lección del extractor** (§2): al leer repeticiones crudas, la acción está en el paso
  *siguiente*. Verificar contra el dataset construido antes de creerse cualquier cifra nueva.
