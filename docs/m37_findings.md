# M37 — set transformer sobre las opciones: el kernel funciona, la ganancia no llega (2026-08-03)

> ## ⚠️ ADENDA (2026-08-04) — el envío 55203764 (~500 elo) es INVÁLIDO: mazo equivocado
>
> **El bundle se subió con el mazo de muestra del SDK (Snover / Mega Abomasnow ex) en vez del
> Alakazam de Yushin. Solape con el vocabulario del perfil: 0 de 9 cartas distintas — CERO.** Todas
> las cartas caían al bucket OOV, así que el scorer veía el mismo vector para cada opción y jugaba
> a ciegas. **Ese ~500 elo no mide nada sobre el transformer.**
>
> Causa: `build_submission` toma el mazo de `config.paths.deck_path`, cuyo valor por defecto en el
> perfil `benchmark` es `sample_submission/deck.csv`. Se construyó sin sobrescribirlo.
>
> **Por qué no lo detectó nada:** `validate_submission` pasa (el tarball es estructuralmente
> correcto), `smoke_test_entrypoint` pasa con `bc_failures: 0` (una carta desconocida no lanza
> excepción, se puntúa como OOV), y la verificación del tarball extraído confirmó perfil y
> `feature_dim` **correctos** — el perfil nunca fue el problema; el mazo sí. Un fallo invisible a
> toda comprobación estructural por construcción.
>
> **Huella en las repeticiones** (37 partidas reales, 55203764 vs el campeón 55145833):
>
> | | setxf2 (mazo malo) | imitation-fetch |
> |---|---|---|
> | Banca en turno 3 / 5 / 8 | **0,70 / 0,91 / 1,00** | 3,77 / 4,43 / 4,39 |
> | Derrotas con ≤1 Pokémon en juego | **15/37 (41%)** | 7/133 (5%) |
> | Derrotas con los 6 premios sin tomar | 5 | 0 |
> | Elo medio del rival | 508 | 857 |
>
> Perdía en el turno 4 sin noquear nada. **No era el transformer jugando mal: era un agente ciego.**
> Confirmado además que **NO hubo problema de tiempo** — máximo 32,8 s por partida contra 600 s de
> presupuesto, 333 ms/decisión, y el guardia nunca se activó. Toda la preocupación por timeouts de
> §2 era infundada en la práctica.
>
> **Arreglado:** `builder._assert_deck_matches_profile` aborta el build si el mazo comparte menos
> del 80% de sus cartas distintas con el `deck_ids` del perfil del payload, con
> `tests/unit/test_submission_deck_profile_match.py` fijando el caso exacto que se subió. Bundle
> correcto reconstruido: `build/imitation-setxf2-fixed.tar.gz` (mazo 22/22 = 100%, Abra/Kadabra/
> Alakazam ×4). **Sigue sin subirse.**
>
> **Lo que NO cambia:** la medición offline de §2 (+0.017 contra una barra de +0.030) y la
> saturación del gauntlet de §3 se hicieron con el perfil y el mazo correctos. El veredicto de no
> embarcar sigue en pie por sus propios motivos; lo que se invalida es únicamente la evidencia del
> ladder.

**Resumen: el transformer de conjunto se construyó entero y funciona — paridad con torch a 8.9e-16,
235 tests, bundle verificado. Pero contra el modelo que REALMENTE reemplazaría gana +0.017, no los
+0.048 del barrido de Colab, y la barra pre-registrada era +0.030. El barrido se comparaba con su
propio brazo `mlp` re-entrenado (0.7561), no con el MAIN embarcado en `bc_alakazam_fetch.json`, que
marca 0.768 con el mismo instrumento. Segundo hallazgo, independiente y más caro: el field gauntlet
está SATURADO (línea base 0.965 macro, 93% de mazos al 100%, Grimmsnarl 9/9 al 100%), así que no
puede vetar nada.**

Continúa [m36_findings.md](m36_findings.md) (más datos, negativo — "no es techo de datos, es techo
de arquitectura"). M37 pone a prueba justamente esa frase.

---

## 1. Qué se construyó

`SetTransformerV2`: dos flujos (estado codificado una vez por decisión → FiLM sobre las opciones),
d_model=128, 3 capas, 4 cabezas, 658k params por miembro, **sin codificación posicional** (dos copias
de la misma carta son vectores idénticos; la equivarianza a permutación garantiza que reciban el
mismo score, y una codificación posicional dejaría al modelo ajustar ruido puro entre copias).

Embarcado en `src/ptcg_ai/imitation/setnet.py`, stdlib puro: matmul, layernorm, softmax y atención
multi-cabeza a mano, porque el runtime de Kaggle no tiene numpy (ADR-0014).

**Verificaciones que sí pasaron:**

| qué | resultado |
|---|---|
| Paridad stdlib ↔ torch (float64) | **8.9e-16** (objetivo 1e-9) |
| Invariante de duplicados | exacto, 0.0 de diferencia |
| Equivarianza a permutación | pasa |
| Suite completa | **235 pasan**, 4 se saltan (solo los de torch) |
| Tarball extraído | `_IMITATION_READY: True`, `set_contexts: ["MAIN"]`, `bc_failures: 0` |
| 40 decisiones reales | 31/40 de acuerdo con el maestro, cero fallback silencioso |

---

## 2. El resultado: la barra no se alcanza

Instrumento: `scratchpad/pareto_k.py` — fidelidad **estricta por identidad de carta** (R21) sobre el
20% held-out, comparando contra el MAIN embarcado, no contra un brazo re-entrenado.

| scorer | index | strict | vs base | p50 s/partida | p99 s/partida |
|---|---|---|---|---|---|
| **mlp (embarcado)** | 0.762 | **0.768** | — | ~0 | ~0 |
| setxf2 k=1 | 0.747 | 0.763 | **−0.006** | 12 | 81 |
| setxf2 k=3 | 0.758 | 0.777 | +0.009 | 36 | 243 |
| setxf2 k=5 | 0.765 | 0.783 | +0.015 | 60 | 405 |
| setxf2 k=7 | 0.769 | 0.786 | **+0.017** | 83 | 567 |

**Barra pre-registrada: +0.030 y p99 < 200 s. Ninguna `k` cumple las dos; ninguna cumple la primera.**

### 2.1 Por qué Colab decía +0.048 — error de comparabilidad de línea base

El barrido comparaba `setxf2` (0.8044) contra **su propio brazo `mlp` re-entrenado** (0.7561). Pero
el modelo a reemplazar es el MAIN de `bc_alakazam_fetch.json`: también un ensemble de 3 MLPs h=48,
pero mejor entrenado, que marca **0.768** con este mismo instrumento. El transformer nunca le ganó
+0.048 al campeón — le ganaba +0.048 a una MLP *más débil que el campeón*.

Es la misma clase de error que ya se corrigió una vez en esta milestone (comparar contra el 0.7831
de M31 cuando el corpus había crecido), reaparecida en otra forma. **Regla para la próxima: la línea
base de un barrido es el artefacto embarcado, medido con el mismo instrumento, nunca un brazo
interno del barrido.**

### 2.2 Lo que sí es real: el transformer acierta la CARTA, no el índice

A k=3 el transformer es **peor en índice** (−0.004) y **mejor en estricta** (+0.009). El margen vive
entero en decisiones con cartas duplicadas — exactamente lo que compra la equivarianza a permutación.
El sesgo inductivo funciona como se predijo; simplemente no vale +0.030.

---

## 3. Hallazgo independiente: el field gauntlet está saturado

Antes de gastar ~3 h en el veto, se midió si el instrumento puede discriminar. No puede.

Campo reconstruido desde las repeticiones **reales** de `imitation-fetch` (submission 55145833,
~870-890 elo, 134 partidas descargadas), sustituyendo al cacheado, que era de `imitation v1/v2` del
6-10 de julio en la era ~650-700 elo:

| | campo viejo | campo fresco |
|---|---|---|
| Mazos distintos | 120 | 72 |
| Grimmsnarl por **episodios** | — | **30,8%** |
| Grimmsnarl por **mazos distintos** | 7,5% | 12,5% |

Y la línea base sobre el campo fresco:

```
macro WR        : 0.965
mazos al 100%   : 67/72  (93%)
mazos bajo 50%  : 0/72
franja Grimmsnarl: 1.000 sobre 9 mazos
```

**Solo hay 3,5% de margen.** Es el fallo de M11 ("gauntlet saturado, no puede verificar mejoras
sutiles") repetido con un campo nuevo.

Lo más informativo: **en el ladder real perdemos contra Grimmsnarl (~37%) y aquí le ganamos 9/9 al
100%.** La diferencia no está en el mazo sino en el **piloto** — Grimmsnarl pilotado por greedy es
trivial. El campo tiene los mazos correctos y los pilotos equivocados. Un campo greedy no puede
vetar nada a este nivel de juego, por fresco que sea.

### 3.1 Dos defectos del instrumento, uno no arreglado

1. **Frescura** — arreglado apuntando `LADDER_FOLDERS` a `replays/55145833`.
2. **Ponderación** — NO arreglado y estructural: `_extract_field_uncached` deduplica y el macro es
   una media **sin ponderar** sobre mazos distintos. Los 41 episodios de Grimmsnarl colapsan en 8
   mazos, así que el veto le da 12,5% de peso a lo que en el ladder es el 30,8% de las partidas.

---

## 4. Reutilizable pase lo que pase

- **`setnet.py`**: atención multi-cabeza + layernorm + softmax en stdlib puro, con paridad probada.
  Cualquier arquitectura de conjunto futura ya tiene camino de embarque.
- **`_score_options`**: punto de entrada a nivel de DECISIÓN en `policy.py`. `_score(spec, x)` era
  por-opción por firma y hacía inexpresable *"esta carta es la mejor de las que tengo"* a cualquier
  profundidad. Los dos sitios de llamada (`_decide` y `_decide_switch`) ya pasan por él.
- **Guardia de tiempo por episodio** con degradación escalonada, y `on_episode_start()` enganchado
  al único límite de episodio que expone Kaggle: la petición del mazo (`select is None`). El plan
  original afirmaba que no existía tal canal; sí existe.
- **`head_to_head_par.py`**: gauntlet repartido entre procesos, **verificado bit a bit idéntico** al
  serial. Se descartó deliberadamente un scorer numpy (7×) porque difiere 1.33e-15 y podría voltear
  un desempate; el paralelismo da lo mismo y es exacto por construcción.
- **`pareto_k.py`**: fidelidad estricta de varios scorers en **una sola pasada** sobre el held-out
  (el featurizado domina y es idéntico para todos), con cortes por arquetipo.
- El campo del gauntlet, ahora fresco y reproducible desde las repeticiones reales.

---

## 5. Tres bugs propios que valieron la revisión

1. **`time.monotonic()` dejaba el guardia ciego** — los deltas salían exactamente 0.0 y el
   presupuesto nunca crecía. `perf_counter` lo arregla.
2. **El chequeo de turno estaba fuera del `try`** de `_decide`: un `turn` inesperado habría sido un
   crash en vez de una degradación a greedy.
3. **`test_setnet.py` entero se saltaba bajo `uv run`** (sin torch), así que la paridad no protegía
   nada en CI. Reestructurado: solo los 4 tests de torch se saltan, y hay vectores dorados en stdlib
   puro que corren siempre.
