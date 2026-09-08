# M34 — TO_HAND: el contexto que nunca se midió, y la primera puerta positiva desde M28

**Fecha: 2026-07-29.** Continúa [m33_yushin_deep_dive.md](m33_yushin_deep_dive.md), que dejó dos
hipótesis vivas para el retraso de ~0.9 turnos en el ensamblaje del combo: (A) pérdida de piezas por
disrupción del rival, (B) mala priorización de jugadas. Este milestone mide (B) en el contexto que
nadie había mirado y encuentra un defecto grande, con mecanismo identificado y arreglo barato.

**Herramientas nuevas (read-only):** `scratchpad/analyze_tohand_divergence.py`,
`scratchpad/triage_tohand_profile.py`, `scratchpad/assembly_turn_gauntlet.py`.
**Cambio en `src/`:** un perfil nuevo y aditivo `ALAKAZAM_FETCH` (dim 750). Nada existente tocado.

---

## 0. Por qué TO_HAND, y por qué nadie lo había mirado

El payload embarcado `data/models/bc_alakazam_mlp.json` tiene **MAIN = `mlp_ensemble`** (M28 lo subió
de 0.603 a 0.780) y **todo lo demás lineal**. Entre ese "todo lo demás" está **TO_HAND en 0.606**
(greedy 0.512) — el contexto donde se resuelven los ~16 tutores del mazo (Dawn ×4 busca 3, Hilda ×4,
Poké Pad ×4, Buddy-Buddy Poffin ×4). Es decir: **qué pieza del combo traigo a la mano**, que es
literalmente la variable que M33 midió como tardía.

Cada intento de mejora desde M28 (M31 V2, M32 MATCHUP, M32 SPECIALIST) fue sobre MAIN. TO_HAND nunca
se midió ni se actualizó.

---

## 1. La medición: el clon busca la carta equivocada

`scratchpad/analyze_tohand_divergence.py` re-puntúa con el clon embarcado las decisiones TO_HAND del
propio maestro. **16.173 filas crudas → 10.393 puntuadas** (4.512 descartadas por ser elección de
premios — informativamente vacías, el entrenamiento también las descarta; 1.219 con una sola opción).

**Tres trampas de medición que el script evita** (las tres son reales en estos datos):
- **39% de las filas TO_HAND son elección de premios.** Contarlas infla todo.
- **55% de las decisiones tienen opciones duplicadas con el mismo card_id.** El acuerdo se compara
  por **multiconjunto de card_id**, no por conjunto de índices.
- **16.5% de las decisiones son triviales** (todas las opciones son la misma carta) y otro 10% tienen
  una sola opción. El 0.606 embarcado está inflado; todo se reporta dos veces.

| Corte | acuerdo |
|---|---|
| todas las decisiones | 63.9% |
| solo no triviales | **57.0%** |
| turnos 1-5 | 61.5% |
| **turnos 1-5, no triviales** | **54.2%** |
| turnos 6+, no triviales | 60.2% |

Y en los turnos 1-5, sobre **estados idénticos**:

| bucket buscado | maestro | clon | delta |
|---|---|---|---|
| **COMBO** (Abra/Kadabra/Alakazam/Rare Candy/energías) | **59.9%** | **41.9%** | **−17.9** |
| ENGINE (Dunsparce/Dudunsparce/Fezandipiti) | 38.6% | 58.0% | +19.4 |
| OTHER | 1.5% | 0.1% | −1.4 |

**El desacuerdo #1, con mucha diferencia: el maestro busca Kadabra, el clon buscaría Dudunsparce —
914 veces.** El clon cambia sistemáticamente pieza-de-combo por motor-de-robo.

### Puerta G-0 (pre-registrada antes de correr)

| | criterio | resultado |
|---|---|---|
| **G-0a** | acuerdo temprano no trivial < 0.75 | **54.2%** → PASS |
| **G-0b** | en el subconjunto de tradeoff real: `T − C ≥ 0.05` y McNemar p < 0.01 | n=3.506, **T=0.494 vs C=0.210**, gap **+0.284**, **p=6.3e-141** → PASS |
| **G-0c** | ≥1000 desacuerdos tempranos y ≥0.8 por partida | 2.128 y 1.73 → PASS |

El instrumento está calibrado: n y T se predijeron independientemente en 3.483 y 0.493 antes de
correr; salieron 3.506 y 0.494.

> **Nota de honestidad:** el prior escrito en el propio script era que **G-0b iba a fallar** (M33 §6
> había encontrado al clon *más* agresivo que el maestro sobre estados idénticos, y §4 su composición
> de mano indistinguible). Falló el prior, no la puerta.

---

## 2. El mecanismo: el modelo es ciego a la variable que gobierna la decisión

Dividiendo el mismo subconjunto por una variable que el modelo **no recibe**:

| Abra en mano | Kadabra en mano | n | maestro COMBO | clon COMBO |
|---|---|---|---|---|
| No | No | 2249 | **59.1%** | **19.7%** |
| No | Sí | 975 | 30.6% | 29.5% |
| Sí | No | 183 | 50.3% | 1.1% |
| Sí | Sí | 99 | 12.1% | 2.0% |

El maestro sigue una regla trivial y limpia: **no busques una pieza que ya tienes** (59% → 31% → 12%).
El clon no la sigue: *sube* cuando ya tiene Kadabra (dirección invertida) y en la celda más frecuente
(n=2249, donde la respuesta correcta es inequívoca) falla por **39 puntos**.

### Por qué está ciego — la parte que importa para el diseño

En una decisión TO_HAND **todas las opciones son `OptionKind.CARD` del mismo área**. Mirando
`features.featurize_option` (`return a + b + c + d + e + f + s + a_s + b_s`):

| bloque | ¿discrimina entre opciones de una decisión TO_HAND? |
|---|---|
| A (tipo de opción), C (ataque), E (combate), F (activo rival), **S (snapshot)**, **A⊗S** | **NO** — idénticos entre opciones ⇒ se cancelan en el softmax |
| B (one-hot de card_id) | sí, pero es un prior global por carta |
| D (área/target) | parcialmente |
| **B⊗R (card_id ⊗ reduced)** | **sí — es el ÚNICO canal de estado** |

`reduced_alakazam` son 9 escalares: hand_size, mano rival, turno, prize_diff, energías del activo,
tiene-{P}, Alakazam en juego, Rare Candy en mano, HP rival. **Ninguno dice si ya tienes un Abra o un
Kadabra en la mano.** El modelo no puede expresar la regla del maestro, así que ajusta un proxy
(probablemente hand_size/turno) que se invierte justo en la celda decisiva.

**Esto es una brecha de OBSERVABILIDAD, no de capacidad.** Más ancho de MLP sobre features que omiten
la variable discriminante no puede aprender la regla — y explica por qué M31 falló: `ALAKAZAM_V2`
(dim 1269) enriqueció **solo `snapshot`** y dejó `reduced_len=9` intacto. Verificado aritméticamente:
1269 − (74 + 76 + 12·76) = 207 = 23×9. **V2 nunca tocó el único bloque que podía ayudar aquí.**

---

## 3. El arreglo: `ALAKAZAM_FETCH` (dim 750)

Perfil **aditivo** — mismo deck/attacks/snapshot/damage/wants que `ALAKAZAM`; solo `reduced` crece
9 → 13 con exactamente las variables que la sonda nombró:

```python
_frac(_count_hand_a(me, _ABRA_ID), 2)       # ¿ya tengo Abra en mano?
_frac(_count_hand_a(me, _KADABRA_ID), 2)    # ¿ya tengo Kadabra en mano?
_frac(_count_hand_a(me, _ALAKAZAM_ID), 2)   # ¿ya tengo Alakazam en mano?
_frac(_count_in_play(me, _ABRA_ID), 3)      # ¿hay un Abra en juego para evolucionar?
```

Perfil separado para que `bc_alakazam_mlp.json` (fijado a 658) siga cargando sin cambios. Test de
dims añadido en `tests/unit/test_profile_snapshot_lengths.py`. **215 tests pasan.**

### Puerta G-1 — A/B lineal en TO_HAND (`scratchpad/triage_tohand_profile.py`)

Split 3-vías por partida con las mismas semillas que todos los experimentos ALAKAZAM previos
(0.2 seed=0 → test; luego 0.25 seed=1 → train/val), así que los números son comparables.

| arm | dim | l2 | val | TEST | TEST no-trivial | gap train−test |
|---|---|---|---|---|---|---|
| ALAKAZAM (baseline C0, reentrenado) | 658 | 1e-3 | 0.6272 | 0.6074 | 0.5256 | +0.0223 |
| **ALAKAZAM_FETCH** | 750 | 1e-4 | **0.7264** | **0.7104** | **0.6393** | **+0.0175** |

**delta: +0.1030 global, +0.1137 en el subconjunto no trivial.** G-1 exigía ≥ +0.02 sin que el gap de
sobreajuste creciera más de +0.02: pasa por 5.7×, **y el gap de sobreajuste BAJA** (−0.0048). Esa es
la firma de una feature informativa, no de capacidad abusada.

Sanity check: la baseline reentrenada (0.6074) reproduce el 0.606 del payload embarcado.

Detalles metodológicos del arnés:
- Desempate **estable por índice más bajo**, igual que `policy._rank_order` en vivo — `np.argsort` es
  inestable y con 55% de opciones duplicadas idénticas las dos convenciones difieren materialmente.
- Plackett-Luce por **expansión en etapas** (equivalente a `_train_multi`, pero vectorizado). TO_HAND
  es 95.3% single-pick, así que la parte multi-pick es casi ceremonial.

---

## 4. El mecanismo de embarque: perfil POR CONTEXTO (MAIN queda intacto)

El payload lleva históricamente **un solo `DeckProfile` para todos los contextos**, así que adoptar
`ALAKAZAM_FETCH` habría obligado a reentrenar también el MLP de MAIN en dim 750 — horas de cómputo y
riesgo sobre el mejor activo del campeón.

Pero `policy.py` ya resolvía un perfil distinto por contexto dentro del spec `mlp_switch` (M32).
Se generalizó ese mecanismo: **cualquier spec de contexto puede declarar su propio `profile`**, que
`_validate_spec` verifica contra la dim de *ese* perfil y `_decide` usa para featurizar. Resultado:

```
MAIN     = mlp_ensemble del campeón, copiado BYTE A BYTE, sigue en ALAKAZAM/658
TO_HAND  = vector lineal bajo ALAKAZAM_FETCH/750     <- el único cambio
resto    = vectores del campeón, intactos
```

`scratchpad/build_fetch_weights.py` **hashea `contexts["MAIN"]` antes y después y se niega a
escribir si cambió** — una regresión silenciosa de MAIN sería invisible en toda métrica offline, y
es exactamente el footgun que tiene `build_mlp_alakazam_weights.py:66`. Bundle + `validate_submission`
+ `smoke_test_entrypoint` verificados. 215 tests + 2 nuevos que fijan el override.

**A/B lineal en MAIN (3h42m):** `ALAKAZAM` 0.5818 vs `ALAKAZAM_FETCH` 0.5880 = **+0.0063**, muy por
debajo del listón de +0.02. Las 4 features son específicas de TO_HAND (+0.114) y casi inertes en MAIN
(+0.006) — confirma que dejar MAIN en 658 no cuesta nada, y que **no vale la pena reentrenar el MLP
de MAIN en dim 750** en el futuro.

---

## 5. G-3 — la puerta de comportamiento: **FALLA**

`scratchpad/assembly_turn_gauntlet.py`, 25 mazos de campo × 8 partidas = **200 partidas por brazo**.

| hito (turno medio) | campeón | candidato | delta | CI 90% (partidas) |
|---|---|---|---|---|
| Alakazam en juego | 5.21 | 5.08 | −0.12 | [−0.60, +0.38] |
| Alakazam activo | 5.26 | 5.19 | −0.07 | [−0.56, +0.43] |
| **Alakazam activo con {P}** | **5.43** | **5.41** | **−0.02** | **[−0.50, +0.49]** |
| nunca ensambló | 2.5% (5/200) | 3.0% (6/200) | | |

G-3 exigía delta ≤ −0.25 con CI excluyendo 0. **FALLA.** El ensamblaje no se acelera.

Sanity check del instrumento: el campeón simulado ensambla en 5.43, casi idéntico al 5.46 medido en
sus replays reales de ladder (M33 §3). El instrumento está calibrado.

### 5-bis. Re-corrida en el campo CORRECTO — sigue fallando, y ahora peor

El campo cacheado resultó ser **el campo equivocado para esta pregunta**: de sus 120 mazos solo **6
(5%) son Marnie's Grimmsnarl** y 1 es Team Rocket, porque se extrae de
`Logs/Submission {greedy v5, imitation v1, v2}` — nuestros rivales de cuando jugábamos a ~650 elo.
Pero la brecha de 0.9 turnos de M33 es **específica de Grimmsnarl** (58.1% del campo del maestro,
27% del nuestro). Se añadió `--marker` y se repitió restringiendo a esos 6 mazos, 24 partidas cada
uno = **144 partidas por brazo**:

| campo | campeón | candidato | delta | CI 90% (partidas) |
|---|---|---|---|---|
| general (200/brazo) | 5.43 | 5.41 | −0.02 | [−0.50, +0.49] |
| **solo Grimmsnarl (144/brazo)** | **5.86** | **6.09** | **+0.23** | [−0.56, +1.05] |

El instrumento capta correctamente que vs Grimmsnarl el ensamblaje se retrasa (5.86 vs 5.43), igual
que en los replays reales. Pero **en el campo donde vive el defecto el candidato es nominalmente más
LENTO.** Ninguno de los dos deltas es significativo, pero son **344 partidas por brazo** sin una sola
señal a favor, y el estimador puntual del campo relevante va en contra. **La excusa "no se probó
donde importa" queda eliminada.**

### Y esto NO es "otro proxy que mintió" — la intervención sí funciona

Re-corriendo `analyze_tohand_divergence.py --weights` sobre el candidato, en los mismos estados del
maestro:

| métrica (turnos 1-5) | campeón | **candidato** | maestro |
|---|---|---|---|
| acuerdo, no trivial | 54.2% | **68.5%** | — |
| acuerdo global | 63.9% | **72.5%** | — |
| busca pieza COMBO | 41.9% | **52.1%** | 59.9% |
| tradeoff subset, tasa C | 0.210 | **0.371** | T = 0.494 |

El gap T−C pasa de **+0.284 a +0.123: cierra el 57% de la brecha de comportamiento.** La cabeza nueva
está viva y hace exactamente lo que fue diseñada para hacer.

**Conclusión causal:** en las 7 mispredicciones anteriores (M11/M14/M31/M32) nunca se verificó si la
intervención siquiera cambiaba el comportamiento. Aquí sí, y el resultado es más fuerte que un
"negativo de proxy": **elegir la carta correcta al tutorizar NO controla la velocidad de ensamblaje
en este mazo.** Explicación más plausible: Dudunsparce roba 3 cartas, así que la ruta "equivocada"
del clon encuentra la pieza igual, solo que dando un rodeo — el motor de robo sustituye al tutor.
Esto **falsifica la hipótesis (B) de M33** (prioridad/secuenciación como causa del retraso) por la vía
de TO_HAND, y deja **(A) pérdida de piezas/resiliencia** como la única de las dos aún sin medir.

---

## 6. Disposición

La puerta pre-registrada falló, así que **no se embarca por la vía "arregla el ensamblaje"** —
promover tras una puerta fallida es mover la portería, el error que este repo ya documentó.

Se corrió `head_to_head` igualmente, como **comprobación exploratoria de una pregunta distinta**
("el candidato es más fiel al maestro, ¿gana más?"), con barra elevada: no basta "no pierde", tendría
que **BEATS** con CI limpio para justificar un slot.

**Resultado (120 mazos × 10 partidas × 2 brazos = 2400 partidas):**

```
fetch  macro WR 0.969   (interventions=0)
mlp    macro WR 0.959   (interventions=0)
paired delta: +0.010  90%CI [+0.000, +0.020]   mazos mejor/peor = 26/16 de 120
VERDICT: fetch BEATS mlp
```

**No hay veto, pero tampoco hay evidencia real, y el veredicto literal engaña:**
1. **El CI toca cero** (`lo = +0.000`). El veredicto "BEATS" sale por la regla `lo > 0` en el tercer
   decimal. No es un CI limpio; mi propia barra elevada **no se cumple**.
2. **Ambos brazos están saturados en 0.96-0.97.** `strong_gauntlet.py` se construyó en M12/M13
   justamente porque el campo greedy satura ~0.90. 78 de 120 mazos empatan (ambos en 1.000).
3. **M14 calibró este instrumento**: leyó una diferencia real de ladder de +0.13 como −0.003. Un
   +0.010 aquí está muy por debajo de su resolución.

Lo único sólido que aporta: **0 intervenciones en 1200 partidas** — el candidato es robusto y no va a
romperse en el ladder.

**Recomendación: NO promover.** No hay una sola medición a favor: G-3 falla en los dos campos (y en
el relevante va en contra), y el `head_to_head` no tiene resolución. Subirlo esperando elo no está
justificado.

**Subirlo como EXPERIMENTO sí tiene sentido, pero hay que llamarlo por su nombre.** Es el
experimento de fidelidad más limpio que el proyecto puede montar: maestro fuerte (~1230, sin el
confound de M11/M14 que imitaban a uno de 650), cambio de comportamiento **verificado** (57% de la
brecha), y **una sola variable** (MAIN byte a byte idéntico). La pregunta que responde —
*¿parecerse más a Yushin produce elo?* — es la que el proyecto lleva 6 milestones sin poder contestar
limpiamente, y **cualquiera de las dos respuestas redirige el tiempo restante**:
- **sube elo** → la fidelidad SÍ era el objetivo; seguir por contextos con fidelidad baja (SWITCH
  0.553, DISCARD 0.385, ambos sin tocar).
- **plano/baja** → cerrar bien una brecha grande hacia un maestro de 1230 no produce nada ⇒ lo que
  falta ya no está en la política por decisión, y todo el esfuerzo debe ir a los levers estructurales.

Valor añadido independiente del elo: **el corpus de replays**. Los 100 replays del campeón
produjeron M32, M33 y M34 enteros; un corpus del candidato mide el ensamblaje en el campo real
(~27% Grimmsnarl) en vez del gauntlet (5%).

Dado el resultado de 5-bis, la expectativa honesta es **plano o ligeramente negativo**. Si el objetivo
es elo y no información, el mismo slot rinde más en **Track E (planificación intra-turno)** — que M34
acaba de hacer más atractivo por eliminación: la ruta de selección por decisión está falsada en el
campo que importa.

**Activos que quedan construidos y son reutilizables pase lo que pase:**
- `ALAKAZAM_FETCH` — perfil aditivo, testeado, inerte si no se usa.
- **Override de `DeckProfile` por contexto en `policy.py`** — permite cambiar UN contexto sin tocar
  los demás ni reentrenar el MLP caro. Es el mecanismo que faltaba para iterar barato.
- `assembly_turn_gauntlet.py` — el primer instrumento del proyecto que mide **comportamiento en
  simulación** con una variable continua. Detectó de paso que `BattleRunner` nunca llama
  `on_battle_start/on_battle_end` (por eso `field_gauntlet` los llama a mano).
- `analyze_tohand_divergence.py --weights` — verificador de "¿la intervención cambió el
  comportamiento?", el chequeo que faltó en las 7 mispredicciones anteriores.
