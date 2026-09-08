# M48 — por qué el RL se cancela a sí mismo, un bug real en la búsqueda, y una hipótesis de
campo cerrada con datos (2026-08-13/14)

**Resumen: se encontró la causa aritmética exacta de por qué el RL de M47 no se movía (87,5% de
la señal de aprendizaje es "de qué partida vino", no "qué tan buena fue la decisión"); se corrigió
con TD/GAE, que funcionó mecánicamente (la deriva desapareció) pero produjo una política
confiadamente PEOR — y su reflejo exacto también fue peor, cerrando la línea de auto-juego con
tres confirmaciones. Se reabrió la búsqueda en árbol con un hallazgo que invierte la premisa de
M18 (el crítico ya estaba bien calibrado sobre el campo real, sin construir nada nuevo) y se
encontró + corrigió un bug real: la búsqueda pisaba las cuatro cabezas especializadas de M42/M43
en cualquier contexto que no fuera MAIN. Corregido, la búsqueda mejoró pero sigue perdiendo. Una
hipótesis nueva del usuario — que el campo del ladder está dominado por clones y la disrupción de
mano explota su fragilidad compartida — se probó con las 4.622 partidas del volcado diario de
Kaggle (cero partidas nuevas) y quedó REFUTADA con significancia. Nada se sube; `imitation-final`
se mantiene intacto en los dos slots.**

Continúa [m47_findings.md](m47_findings.md) (recorte PPO, `tau`, mezcla de rivales, crítico
reajustado — cinco candidatos, ninguno pasó la puerta).

---

## 1. Por qué el RL se cancelaba a sí mismo — la aritmética, no una corazonada

`m47_step_coherence.py` había medido que las seis iteraciones del brazo A de M47 tenían eficiencia
de recorrido **0,214** (contra 0,408 de un paseo aleatorio) y coseno consecutivo **−0,283** — los
pasos se cancelaban. Faltaba el *por qué*.

`scratchpad/m48_advantage_variance.py` descompone `Var(A)` por partida: cuánto de la varianza de la
ventaja es **entre partidas** (idéntico para las ~45 decisiones MAIN de una partida, sin información
sobre cuál fue buena) contra **dentro de partida** (lo único que puede discriminar una decisión de
otra).

| estimador | std(A) | entre partidas | dentro de partida |
|---|---|---|---|
| sin crítico (`A = R − media`) | 0,4998 | 100,0% | 0,0% |
| crítico viejo (`v_alakazam`, AUC 0,65 en auto-juego) | 0,4779 | 92,5% | 7,5% |
| **crítico nuevo (`v_alakazam_v2`, AUC 0,79)** | 0,4360 | **87,5%** | **12,5%** |

Triplicar la calidad del crítico movió la parte que discrimina de 7,5% a 12,5%. El 87,5% restante
es ruido de resultado — el gradiente dice "haz más de lo que hiciste en las partidas que ganaste",
y qué partidas se ganan es casi una moneda que se vuelve a tirar en cada lote. Eso, combinado con
que Adam da un paso de **longitud fija** haya señal o no (medido en M47: 0,138-0,149 por
iteración), produce exactamente la anti-correlación medida.

## 2. El remedio: TD/GAE, calculado sobre datos ya recolectados

`_featurize_traj` en `rl_selfplay.py` gana un parámetro `gae_lambda` (por defecto `None` = el
retorno Monte Carlo de siempre, camino byte-idéntico — verificado con 5 tests nuevos, incluida la
identidad telescópica: con `gamma=1, lambda=1` las ventajas TD deben colapsar exactamente al
retorno Monte Carlo, y así lo hacen). Con `gae_lambda` puesto:

```
delta_t = gamma*V(s_{t+1}) − V(s_t)      (último paso de la partida: R − V(s_T))
A_t = Σ_k (gamma*gae_lambda)^k · delta_{t+k}
```

Dentro de partida por construcción — no puede heredar el problema medido en §1.

## 3. Gate mecánico — pasado, y con una escalera perfecta

`scratchpad/m48_gae_chain.py` repite EXACTAMENTE el brazo A de M47 (mismos 6 lotes, mismos
hiperparámetros) cambiando sólo el estimador de ventaja, barriendo `gae_lambda ∈ {0, 0,5, 0,95}`.
Puerta pre-registrada: coseno consecutivo de −0,283 a **≥ 0,00**, eficiencia de 0,214 a **≥ 0,40**.

| λ | entre partidas | eficiencia | coseno medio |
|---|---|---|---|
| **0,0** | **2,5%** | **0,661** | **+0,234** |
| 0,5 | 7,5% | 0,452 | −0,081 |
| 0,95 | 60,2% | 0,227 | −0,283 |
| MC (M47) | 87,5% | 0,214 | −0,283 |

Monótono en las tres columnas, y `λ=0,95` reproduce el Monte Carlo casi exacto — coherente con la
teoría. **Gate pasado por λ=0**: la asignación de crédito era real y arreglarla elimina la
cancelación.

## 4. La arena — el candidato pierde con confianza, y su reflejo exacto también

`m47_arena.py` sobre el checkpoint λ=0 de la iteración 6, n=600 + control:

| | resultado | IC 95% |
|---|---|---|
| candidato vs campeón | 0,448 | [0,409, 0,488] |
| control (campeón vs sí mismo) | 0,487 | [0,448, 0,527] |

**Confiadamente peor**, con el control dentro de la banda válida. El arreglo mecánico funcionó — la
política viajó de forma coherente — y el sitio al que viajó es peor que el punto de partida.

**La comprobación adicional, y por qué se hizo:** el desplazamiento medido (`|d| = 0,4697`, sólo
**1,03%** de la norma del campeón en un espacio de 95.043 dimensiones) es lo bastante pequeño para
que la aproximación lineal sea razonable, así que `scratchpad/m48_reflect.py` construyó el reflejo
exacto (`W_champion − (W_candidate − W_champion)`, coseno verificado en **−1,0000**) y se midió
igual: **0,427** [0,388, 0,467] — también confiadamente peor.

**Que ambos signos pierdan cierra la pregunta, no la deja abierta.** El campeón está en un óptimo
local a lo largo de ese eje específico del espacio de pesos: moverse en cualquier dirección por ahí
empeora. No es un error de signo del gradiente; es que el objetivo de auto-juego (ganarle a tres
clones fijos) apunta a algo que no es "jugar mejor en el ladder real".

**Tercera confirmación independiente de la línea de auto-juego, cerrada:** plano (MC), peor con
confianza (TD), peor con confianza (reflejo exacto). No se prueba una cuarta variación sobre el
mismo mecanismo.

## 5. La búsqueda, reabierta — y el motivo de M18 ya no aplica aquí

### 5.1 El crítico ya estaba calibrado sobre el campo real — sin construir nada

M18 cerró la búsqueda porque su valor de hoja se ajustó sobre auto-juego de TR-650 y estaba mal
calibrado fuera de esa distribución. `scratchpad/m48_fit_ladder_critic.py` probó si haría falta
ampliar `v_alakazam.json` con nuestras propias partidas de ladder (135 partidas held-out de
`imitation-final`, nunca tocadas en el ajuste):

| crítico | AUC held-out (ladder real) |
|---|---|
| `v_alakazam.json`, ajustado SÓLO sobre el corpus de Yushin | **0,7550** |
| candidato ampliado con nuestras 288 partidas propias | 0,7472 (−0,0078) |

**El crítico de M47 (AUC 0,6526 sobre auto-juego) da 0,755 sobre partidas de ladder reales** —
porque el corpus de Yushin (2.330 partidas, todo el campo) nunca fue la distribución estrecha que
hundió a M18; el auto-juego de 3 arquetipos sí lo era. Ampliar con datos propios lo empeoró
ligeramente (piloto más ruidoso sobre la misma distribución). **No se construyó ningún crítico
nuevo — se usó `v_alakazam.json` tal cual.**

### 5.2 Trampa cerrada antes de que importara

`LearnedEvaluator.value` puntuaba con `zip(pesos, features, media, std)` — y `zip` **para en el más
corto sin lanzar error**. Con `v_alakazam_v2.json` (10 pesos) contra un evaluador de 7 u 8
features, se habría truncado en silencio y jugado con un evaluador corrupto sin ningún síntoma — la
misma forma de fallo que costó la submission 55203764. Se añadió una aserción de ancho en
`__init__` y `n_features` declarado por subclase; 6 tests nuevos en
`tests/unit/test_search_evaluator_guard.py`.

### 5.3 Sonda de coste — y por qué el protocolo completo era inviable tal cual

Con `SearchConfig` por defecto (`determinizations=4, max_candidates=16`): **74,3 s/partida** —
cómodo contra el presupuesto de Kaggle (600s/episodio) pero 8-15× más caro que las arenas normales
con las que se decide todo en este proyecto. El protocolo completo (barrido + dos puertas a n=600)
habría tomado **12+ horas por brazo**. Se recortó la config
(`determinizations=2, max_candidates=8, max_candidate_sets=4`) y se añadió paralelismo por partida
(`scratchpad/m48_search_arena.py`, mismo patrón de sharding que `rl_selfplay.collect`, medido
~2,68× en 4 núcleos): **11,73 s/partida equivalente**, arena n=600 en ~2h por brazo.

### 5.4 El bug real — la búsqueda pisaba las cabezas especializadas de M42/M43

Con la config recortada, dos sondas cortas (n=10, n=20) acumularon **4W-26L (13,3%)** — señal
demasiado mala para ignorar antes de gastar horas. Se amplió a n=100 en tres valores de
`greedy_bias`:

| `greedy_bias` | resultado | IC 95% |
|---|---|---|
| 0,02 (más búsqueda) | 0,175 | [0,113, 0,261] |
| 0,05 | 0,280 | [0,201, 0,375] |
| 0,10 (más confianza en BC) | 0,270 | [0,193, 0,364] |

Monótono en el extremo bajo, luego se estanca alrededor de 0,27-0,28 — nunca cerca de 0,50 aunque
`greedy_bias` casi no deje que la búsqueda anule nada.

**El mecanismo, encontrado leyendo `search.py`:** `SearchPolicy.choose()` no distingue contexto —
corre para CUALQUIER decisión con más de una opción: MAIN, pero también TO_HAND, ACTIVATE,
SETUP_BENCH, SWITCH. Genera candidatos alternativos alrededor de la respuesta del campeón y, si el
evaluador de 7 features (diseñado y validado sólo para tableros de MAIN) puntúa una alternativa por
encima del margen `greedy_bias`, **descarta la cabeza especializada** — exactamente las cuatro
cabezas de M42/M43 que son la mayor mejora medida del proyecto (+0,096 a +0,117 en esta misma
arena). Esto nunca fue un problema en M10/M17/M18 porque el campeón de esa época era un scorer
lineal de un solo contexto, sin nada que pisar fuera de MAIN.

**Corrección:** `ImitationSearchPolicy.choose()` (en `search_bc.py`, no se tocó el módulo base
compartido) ahora enruta directo a `self._greedy.choose(ctx)` — el campeón completo, con sus
cabezas — para cualquier contexto que no sea MAIN, y sólo entra en la maquinaria de búsqueda para
MAIN. 20 tests nuevos en `tests/unit/test_search_context_routing.py`. 263 tests totales pasan.

**Medido, con la corrección, mismo `gb=0,05`, n=100:**

| | sin corregir | con la corrección |
|---|---|---|
| candidato vs campeón | 0,280 | **0,330** [0,246, 0,427] |
| control (campeón vs sí mismo) | — | 0,560 [0,462, 0,653] (ruidoso a n=100, M41 ya documentó que este motor necesita n~600) |

**El bug era real y la corrección ayudó (+0,05), pero no alcanza.** Incluso ajustando
conservadoramente por el ruido del control, el candidato queda ~0,27-0,33 — muy por debajo de
0,50, y el límite superior de su IC (0,427) ya está por debajo de 0,50 sin ningún ajuste. Con el
barrido sin corregir ya se había visto que `greedy_bias` no sigue cerrando la brecha más allá de
~0,28. **Diagnóstico distinto al de M18: aquí el crítico SÍ está bien calibrado (AUC 0,755 en
datos reales), pero es demasiado tosco para discriminar entre 2-3 alternativas parecidas dentro
del mismo turno** — un límite de resolución, no de calibración.

**No se completó el protocolo de n=600 (Gates A/B) dado el tiempo restante y esta señal.** La
búsqueda queda con un bug real corregido (reutilizable si se revisita) pero sin evidencia de que
cierre la brecha a 0,50 con el valor de hoja actual.

## 6. La hipótesis del campo dominado por clones — probada con datos existentes, refutada

Planteamiento del usuario, dicho llanamente: si la mayoría de quienes nos ganan también son clones
(no expertos humanos de TCG), entonces "jugar más parecido a un jugador fuerte" —la premisa de
M7 a M48 completo— podría estar optimizando la variable equivocada. El muro de Grimmsnarl encaja:
M32 lo confirmó como defecto del MAZO (persiste con tres pilotos reales distintos), y gana vía
disrupción de mano — el mecanismo exacto que rompe a un clon (M38: nuestro propio agente se desvía
de la línea del maestro cada ~4,7 decisiones y a partir de ahí juega sin señal de entrenamiento).

**Prueba, con datos ya en disco (el volcado diario de Kaggle de M46, 4.622 partidas reales, 377
jugadores, cero partidas nuevas):** `scratchpad/m48_ladder_disruption_check.py`.

**Lista de cartas construida por TEXTO real, no por memoria** — y esto encontró un error propio a
tiempo: la memoria del proyecto citaba "Petrel/Spikemuth Gym atacan tamaño de mano", pero su texto
real es "busca una carta y ponla en TU mano" — ventaja de cartas, no disrupción. Se escaneó todo el
texto de habilidades/ataques por `"opponent"` + `"hand"` juntos (91 coincidencias,
`_m48_disruption_scan.txt`) y se clasificó cada una a mano contra el efecto real, separando
disrupción genuina (reduce el tamaño de mano del rival) de reveal-only (información, no
disrupción), locks (impide USAR cartas que ya tiene, mecanismo distinto) y daño-escalado-con-mano
(premia una mano pequeña, no la crea).

**Método:** comparación **emparejada** — filtrar a partidas donde EXACTAMENTE un lado lleva
disrupción y el otro no, medir la tasa de victoria de ese lado contra 0,50. Controla razonablemente
el "me tocó un rival débil" (misma partida), no controla "los jugadores con disrupción son
simplemente mejores constructores en promedio" — señalado explícitamente, no escondido.

| comparación | resultado | IC 95% |
|---|---|---|
| **Emparejada** (decisiva) | **0,425** | [0,394, 0,457] |
| No emparejada (todas las partidas) | 0,491 | [0,480, 0,502] |
| Control: cartas de bloqueo (impiden JUGAR Ítems/Apoyo, mecanismo distinto) | **0,605** | [0,577, 0,631] |

**El lado con disrupción de mano PIERDE con significancia en todo el ladder (960 emparejamientos
mixtos).** La hipótesis específica —disrupción de tamaño de mano como estrategia ganadora general
porque explota la fragilidad compartida de los clones— queda **refutada con datos**, no confirmada.
El muro de Grimmsnarl contra nosotros es más probablemente una interacción de mazo puntual
(Munkidori mueve exactamente los contadores de daño que pone nuestro propio Powerful Hand) que una
ley general del campo.

**Hallazgo no buscado, con la misma medición:** las cartas de **bloqueo** (mecanismo distinto —
impiden jugar ciertos tipos de carta, no reducen el conteo) ganan con significancia real: 0,605.
Mismo candado que arriba: correlación, no causalidad probada; podría ser que mazos ya fuertes por
otras razones incluyan más tech de bloqueo, no que el bloqueo cause la victoria. **No se actuó
sobre esto** — construir un mazo nuevo alrededor de un hallazgo correlacional, sin maestro que
clonar y sin tiempo para probarlo en la arena decisiva, no es una apuesta defendible tan cerca del
cierre.

**Nota operativa:** el volcado pesa 21,5 GB (4,6 MB/partida en promedio, historial completo turno a
turno, no un resumen ligero) — el análisis tardó bastante más de lo estimado inicialmente. Se dejó
correr hasta el final en vez de cambiar de método a mitad de camino, para no arriesgar una segunda
extracción improvisada bajo presión de tiempo.

## 7. Recomendación

**No subir nada.** `imitation-final` se mantiene intacto en los dos slots. Nueve hipótesis de
auto-juego/búsqueda falsificadas en total (M31-M48), la línea de auto-juego cerrada con tres
confirmaciones independientes y explicación mecánica completa, la búsqueda con un bug real
corregido pero sin evidencia de cierre de brecha, y la hipótesis de disrupción de mano cerrada con
datos de todo el ladder, no sólo con nuestra propia experiencia.

## 8. Reutilizable

- **`m48_advantage_variance.py`** — descompone cualquier estimador de ventaja en señal entre/dentro
  de episodio antes de gastar tiempo de entrenamiento en arreglarlo.
- **`gae_lambda` en `rl_selfplay._featurize_traj`** — TD/GAE completo, con `--tag` para no pisar
  arms de un mismo setup. Camino MC intacto por defecto.
- **`m48_gae_chain.py`** — el patrón "repetir el mismo experimento cambiando una sola variable,
  featurizar una sola vez, barrer el resto" para cualquier ablación futura de RL.
- **`m48_reflect.py`** — reflejar una dirección de pesos medida como mala; método general para
  comprobar si un óptimo es direccional o si el punto de partida ya es un mínimo local en ese eje.
- **La corrección de contexto en `ImitationSearchPolicy`** — cualquier futuro trabajo de búsqueda
  hereda automáticamente el respeto a las cabezas especializadas.
- **`m48_search_arena.py`** — arena de búsqueda con sharding por partida (mismo patrón que
  `rl_selfplay.collect`), reduce el coste de prueba ~2,7× en 4 núcleos.
- **`m48_fit_ladder_critic.py`** — patrón para validar cualquier crítico futuro sobre partidas de
  ladder reales held-out en vez de auto-juego, con el extractor vetado.
- **`m48_ladder_disruption_check.py`** — plantilla para probar cualquier hipótesis de "esta
  mecánica de carta predice victoria" en todo el campo, usando el volcado diario ya descargado,
  con la comparación emparejada como patrón reutilizable de control.
