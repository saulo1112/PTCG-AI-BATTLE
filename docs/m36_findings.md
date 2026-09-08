# M36 — más datos para el clon de Yushin: NEGATIVO (2026-07-31)

**Resumen: se añadieron 545 partidas nuevas de Yushin (+42%, MAIN 66.162 → 92.231, y +83% sobre lo
que el campeón vio jamás). El MLP NO mejora: seed 0 da TEST 0.7622 contra los 0.767 del campeón
sobre el mismo conjunto. Pero el LINEAL sí sube (+0.032). Los datos siguen teniendo señal — el MLP
h=48 ya no puede extraerla. No es techo de datos, es techo de arquitectura.**

Continúa [m35_findings.md](m35_findings.md) (clon de Luca, cerrado en negativo por techo de datos).

---

## 1. Por qué se intentó

M35 cerró el clon de Luca. Quedaba una palanca no explotada y barata: **el campeón se entrenó con
menos datos de los que ya teníamos.** Verificado antes de gastar cómputo:

```
campeón MAIN entrenado con:  38.783 train + 11.540 val = 50.323
MAIN disponible en el dataset (27-jul):                  66.162   (+31,5% sin usar)
```

Y al re-descargar de forma incremental aparecieron **278 episodios nuevos** (Yushin sigue jugando):
1551 → **1830 replays**, y el dataset pasó de 1284 a **1829 partidas / 152.863 filas / MAIN 92.231**.
Frente a las 50.323 con las que se construyó el campeón: **+83%**.

Era el experimento más limpio posible: **mismo maestro, mismo perfil, misma arquitectura, misma
receta de M28 (h=48, 3 semillas). La única variable es el volumen de datos** — justo la palanca que
produjo al campeón (M28: 0.556 → 0.780 al pasar a datos grandes).

## 2. La línea base, y por qué es comparable

`split_by_game` asigna por hash del `game_id`, así que **el bucket de una partida es estable al
añadir partidas nuevas**: lo que estaba en el 20% retenido sigue estándolo. El campeón nunca entrenó
sobre ese conjunto, así que puntuarlo ahí es limpio.

Con `scratchpad/semantic_fidelity.py` (extendido en M36 para leer specs `mlp_ensemble`):

| campeón `bc_alakazam_mlp.json` | n | índice | estricto |
|---|---|---|---|
| MAIN sobre el TEST viejo | 13.838 | **0.779** | 0.785 |
| MAIN sobre el TEST nuevo | 18.935 | **0.767** | 0.773 |

El 0.779 **reproduce el 0.780 documentado en M28** — el evaluador queda auto-validado contra un
número independiente conocido.

**Puerta M36, pre-registrada antes de correr:** ensemble TEST ≥ **0.777** (campeón 0.767 + 0.010;
con n=18.935 el error estándar ronda 0.003) y gap train−test ≤ 0.030.

## 3. Resultado

```
MAIN featurizado: train=51.258  val=22.038  test=18.935
LINEAR      train 0.5928   TEST 0.5880   (gap +0.0048)
MLP seed 0  val   0.7592   TEST 0.7622
```

**FALLA.** Seed 0 queda **por debajo** del campeón (0.7622 vs 0.767), cuando hacía falta 0.777.

Se detuvo tras la primera semilla porque **el veredicto ya no podía cambiar**: la dispersión típica
entre semillas es ~0.006-0.010 (Luca: 0.6463-0.6521; M31 midió h=96 dentro del ruido entre semillas),
así que las seeds 1-2 caerían en ~0.755-0.772 y el ensamble en ~0.765-0.770. Alcanzar 0.777 era
aritméticamente imposible. Ahorra 5-6 h de máquina.

## 4. El hallazgo real: techo de ARQUITECTURA, no de datos

| modelo | M28 (~50k filas) | M36 (73k filas) | delta |
|---|---|---|---|
| LINEAR | 0.556 | **0.588** | **+0.032** |
| MLP h=48 | 0.780 | ~0.762 | **~plano / negativo** |

**Los datos nuevos SÍ contienen señal aprovechable — el lineal la extrae. El MLP h=48 no.** El
lineal estaba infra-ajustando y mejora; el MLP ya saturó lo que su arquitectura puede representar de
este maestro.

Con esto, las cuatro palancas sobre el clon de Alakazam quedan cerradas con medición:

| palanca | dónde | resultado |
|---|---|---|
| volumen de datos | M31 Track A / **M36** | negativo |
| capacidad (anchura h=48→96) | M31 Track D | **+0.0056, dentro del ruido**; gap +0.018→+0.026 |
| ponderación por victoria (alpha) | M31 Track A | negativo |
| enriquecimiento de features (`ALAKAZAM_V2`) | M31 | +0.009 fidelidad, **perdió el veto head_to_head** |
| selección de cartas (`ALAKAZAM_FETCH`) | M34 | +0.114 TO_HAND, **no movió el ensamblaje** |

**Caveat honesto sobre la interpretación:** las 545 partidas nuevas son de un periodo distinto —
Yushin cayó de ~1148 a ~1096 y el meta pasó de 17% a 51% de Grimmsnarl. Mezclar regímenes podría
diluir la fidelidad por sí solo, independientemente de la capacidad. No está descartado.

## 5. Qué queda vivo, y qué no

**Cerrado:** subir la fidelidad al clon de Yushin por volumen, anchura, ponderación o features.

**Vivo pero sin probar — la hipótesis del usuario:** la arquitectura embarcada es
`658 → 48 → 1`, **una sola capa oculta**, ~32k parámetros. M31 cerró la **anchura**, nunca la
**profundidad**, y nadie ha probado dar **contexto entre opciones**: hoy cada opción se puntúa en
aislamiento total y luego se aplica softmax, así que el modelo no puede representar "esta carta es
la mejor *de las disponibles*" a ninguna profundidad.

Escalera propuesta (barato → caro), a decidir:
1. **DeepSets** — añadir un resumen agregado del conjunto de opciones como features extra. Entrena
   con el trainer existente y embarca con el `_mlp_forward` existente (solo un vector más largo).
   Prueba la misma hipótesis. **Horas.**
2. **Transformer sobre el conjunto de opciones** — solo si (1) da señal. Cómputo viable
   (~1.1M MAC/decisión ≈ 45-135 s/partida contra 600 s de presupuesto), pero exige escribir atención
   multi-cabeza, layernorm y softmax **a mano en stdlib** (~300-500 líneas en el camino de embarque).
   **Días.**

**Sospecha que ninguna arquitectura resolvería:** Yushin tarda **401 ms/decisión** frente a los
123-184 ms de los agentes reactivos del top-8. Si hace búsqueda o mantiene estado interno, sus
decisiones no son función de la observación sola y existe un techo de fidelidad estructural. Encaja
con que tres palancas independientes den ~+0.005 cada una.

## 6. Artefactos

- `data/imitation/yushinito_full.jsonl.gz` — **reconstruido**: 1829 partidas / 152.863 filas.
  El anterior (1284 partidas) se conserva como `yushinito_full_1284g.jsonl.gz`.
- `replays/54773249` — 1830 replays (+278).
- `scratchpad/semantic_fidelity.py` — ahora puntúa specs `mlp_ensemble` además de lineales.
- **Nada embarcado, nada subido.** El campeón `bc_alakazam_mlp.json` queda intacto.
