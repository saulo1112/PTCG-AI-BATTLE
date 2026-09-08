# M49 — +32% de corpus del maestro, un renombre de equipo que casi corrompe todo en
silencio, y el techo de datos medido cabeza por cabeza (2026-08-14/15)

**Resumen: el usuario señaló que cada día trae más datos de Yushin, y tenía razón — su
submission sigue viva y se bajaron 555 episodios nuevos (2.330 → 3.074 partidas, +32%),
con el mazo verificado idéntico en 392/392. Pero el reentrenamiento de las cabezas de
contexto sobre ese corpus ampliado falló en las CUATRO: SWITCH +0,0000, SETUP_BENCH
+0,0000 (saturada en 1,000), ACTIVATE −0,0005 y TO_HAND −0,0437. Además se midió por fin
el agente público de la comunidad que el usuario tenía en el repo (M50): fuerte de verdad
(0,980 contra greedy) pero pierde 6-94 contra nuestro campeón. Por el camino se encontró y corrigió un fallo
silencioso grave: el equipo del maestro se RENOMBRÓ dos veces, y la extracción del corpus
—que filtra por nombre— habría descartado un tercio de las partidas sin error ni aviso.
Nada se sube; los slots del usuario (812,2 y 801,5, resubidos por él) quedan intactos.**

Continúa [m48_findings.md](m48_findings.md) (auto-juego cerrado, búsqueda con bug corregido
pero insuficiente, hipótesis de disrupción refutada).

---

## 1. La premisa del usuario era correcta, y mi primera lectura estaba mal

El usuario planteó que el corpus del maestro crece cada día. Mi primera verificación
concluyó lo contrario —que Yushin había dejado de jugar— porque **no aparecía en el
volcado diario del 11-ago**. Esa inferencia era inválida: el volcado es una **muestra**
(377 jugadores contra los 6.725+ del leaderboard, seleccionada por rating medio), y la
ausencia en una muestra no prueba inactividad.

Verificado contra la fuente autoritativa (`rating_trajectory.py 54773249 --by-day`), la
submission **sigue viva y jugó 33 episodios ese mismo día**:

| día | eps | WR | rival medio | score |
|---|---|---|---|---|
| 02-ago | 262 | 0,56 | 1143 | 1196,8 |
| 08-ago | 69 | 0,65 | 1113 | 1180,6 |
| 11-ago | 31 | 0,39 | 1137 | 1139,8 |
| 13-ago | 31 | **0,26** | 1069 | 1046,3 |
| 14-ago | 33 | 0,52 | 1006 | **1043,4** |

**Lección de método:** una muestra no refuta existencia. La API por `submissionId` es la
fuente; el volcado diario sirve para censar el campo, no para decidir si un agente concreto
sigue corriendo.

## 2. El fallo silencioso: el equipo se renombró DOS veces

Descarga incremental: **555 episodios nuevos, 0 fallos**, 3.075 replays en disco. Al
verificar, ninguna partida nueva contenía a "Yushin Ito" — y la primera lectura ("la
descarga trae partidas equivocadas") **también era incorrecta**: los 189 archivos revisados
estaban todos en la lista de episodios de la API para esa submission.

Resolviendo el asiento por `submissionId` en vez de por nombre
(`scratchpad/m49_resolve_teacher_names.py`), aparecen **cuatro nombres para la misma
submission**:

| nombre | episodios |
|---|---|
| `Yushin Ito` | 256 |
| `AlphaStarmie` | 179 |
| `AlphaTCG` | 58 |
| `AlphaTcg` | 13 |

Verificación de colisión: ninguno de esos nombres aparece jamás en el asiento del rival, así
que el conjunto es seguro como filtro.

**Por qué esto era grave.** Todo el pipeline de imitación filtra por nombre:
`build_decision_dataset(log_dir, player_name, out)` → `player_seats(data, player_name)`
compara `info.Agents[].Name`. Reconstruir el corpus con "Yushin Ito" habría casado **cero
asientos** en cada replay nuevo y escrito un corpus del mismo tamaño que el viejo — **sin
excepción, sin advertencia**. Las cabezas se habrían reentrenado creyendo tener +32% de
datos y teniendo 0%. Es la forma exacta del fallo de M37 (mazo equivocado embarcado,
superando todas las validaciones estructurales).

**Corrección** (`src/ptcg_ai/imitation/kaggle_replay.py`, `dataset.py`): `player_seats`
acepta ahora un **conjunto** de nombres; una cadena simple sigue comportándose igual que
antes. 6 tests nuevos (`tests/unit/test_replay_player_names.py`), incluido uno que fija la
regresión misma (el nombre viejo NO casa las eras renombradas). **269 tests pasan.**

**Y un fallo propio del mismo tipo, en mi verificador:** su primera versión parseó 125
archivos, **saltó los 125** por el nombre obsoleto, y aun así imprimió *"OK — mismo mazo,
seguro para fusionar"*, porque con 0 parseados `mismatched == 0` es trivialmente cierto.
Ahora falla ruidosamente con cero evidencia.

## 3. Las dos puertas previas al entrenamiento

**3.1 Identidad de mazo — PASA.** 392 de 392 partidas nuevas someten el multiconjunto de
60 cartas **idéntico** a `decks/yushinito.csv`. No repitió el patrón de M45/M46 (Majkel:
Alakazam→Lucario; やる気元気ミワハルキ: Grimmsnarl→Lopunny). Mismo agente, mismo mazo.

**3.2 El declive es del PILOTO, no del campo.**

| arquetipo | n nuevo | WR nuevo | WR M33 | Δ |
|---|---|---|---|---|
| marnie_grimmsnarl | 58 | 0,414 | 0,519 | **−0,105** |
| alakazam (espejo) | 52 | 0,712 | 0,875 | **−0,163** |
| mega_kangaskhan | 38 | 0,500 | 0,725 | **−0,225** |
| **global** | 390 | **0,526** | 0,569 | **−0,043** |

Todos los arquetipos con línea base caen, y su rival medio bajó de 1143 a 1006 — enfrenta
gente **más débil** y gana **menos**. Si fuera composición de campo, el WR por arquetipo
estaría plano. El campo también se movió mucho (Grimmsnarl 58,1% → 14,9% de su campo;
Dragapult ahora 14,9% con WR 0,276; "other" 46,2%), pero eso no explica la caída *dentro*
de cada arquetipo. **Se procedió igualmente** —el mazo es idéntico y los datos nuevos son
~24% del corpus final— pero registrando que la fidelidad se estaría dirigiendo hacia un
piloto en declive.

Corpus reconstruido: **3.074 partidas / 248.163 filas**, `illegal_actions=0`. El de 2.330
se preserva como `yushinito_full_2330g.jsonl.gz`.

## 4. El reentrenamiento: cero en toda la línea

Barra pre-registrada +0,020 sobre la cabeza **embarcada** (la regla de M37: la línea base
es el artefacto que se despliega, medido con el mismo instrumento).

| cabeza | filas | base embarcada | candidato (+32% datos) | lift | veredicto |
|---|---|---|---|---|---|
| **SWITCH** | 5.438 | 0,7724 | 0,7724 | **+0,0000** | falla |
| **SETUP_BENCH** | 1.344 | 1,0000 | 1,0000 | +0,0000 | falla (saturada) |
| **ACTIVATE** | 19.159 | 0,9735 | 0,9729 | **−0,0005** | falla |
| **TO_HAND** | 37.979 | 0,8536 | 0,8100 | **−0,0437** | falla |

**Cuatro de cuatro.** Y TO_HAND, la única con margen real (14,6% de techo disponible, la
más grande y la de perfil propio `ALAKAZAM_FETCH`/750), es la que **más empeoró**: −0,0437
con +31% de datos. Eso encaja con el declive medido en §3.2 y no es casualidad de contexto —
TO_HAND es la resolución de tutores ("qué pieza del combo busco"), exactamente la decisión
donde la degradación de un piloto en caída se manifiesta. Los datos nuevos no sólo no
aportan: en el contexto de más volumen, **restan**.

**SWITCH no es un artefacto.** Las tres semillas entrenaron y salieron distintas entre sí
(test solo: 0,7221 / 0,7761 / 0,7596); el ensemble de las tres aterriza en 0,7724, que
coincide con la base al cuarto decimal. Con n=1094, ambos aciertan 845 filas exactas.

**SETUP_BENCH está saturada**: la cabeza de ranking embarcada ya acierta **el 100%** del
held-out. Un lift de +0,020 es aritméticamente imposible. (La ganancia de M42 en ese
contexto vino de la dimensión de **conteo**, que entrena por separado y no entra aquí.)

**ACTIVATE tenía 2,65% de margen total** — la barra de +0,020 exigía corregir tres cuartas
partes de los errores restantes.

### 4.1 La premisa del plan, falsificada por su propio experimento

El plan argumentaba que las cabezas de contexto son "modelos pequeños" y por eso
aprovecharían datos extra donde la MLP de MAIN no (M36 midió: +83% de datos → la MLP h=48
de MAIN **empeoró**, pero el LINEAL subió +0,032).

**La clasificación estaba mal.** SWITCH, ACTIVATE y TO_HAND son `mlp_ensemble(k=3, h=48)` —
**arquitectónicamente idénticas** a la cabeza de MAIN, sólo entrenadas con menos datos. El
techo de arquitectura que M36 midió para MAIN aplica igual a ellas, y estos resultados son
exactamente eso. La única cabeza genuinamente "pequeña" (SETUP_BENCH, lineal) resultó estar
saturada al 100%, así que no podía probar la hipótesis en ningún sentido.

## 5. M50 — el agente público de la comunidad, medido (lo que estaba fuera de la caja)

El usuario señaló, con razón, que llevábamos 19 milestones optimizando **una sola familia**
(clonar un maestro con una red) y que un notebook suyo llevaba días en la raíz del repo
**sin abrir**: `a-better-hand-alakazam-rising-tide-v21.ipynb`, de jazivxt, publicado
explícitamente para la comunidad.

No es una arquitectura de red. Es un agente de familia distinta:

- ~1.200 líneas de **heurística escrita a mano** con pesos de prioridad afinados (`WEIGHTS`,
  `heuristic_scores`) — conocimiento de dominio que este proyecto nunca tuvo;
- **modelado de creencias del rival** (`_match_archetype`, `_TEMPLATES`,
  `_visible_opponent_line`): infiere el arquetipo contrario por lo revelado y cambia de
  plan. Nuestro clon no tiene ninguna noción de contra quién juega;
- **búsqueda determinizada propia** (`_sample_hidden`, `_leaf_eval`, `_search_decide`) sobre
  un evaluador de hoja hecho a mano, en vez de la V aprendida que M10/M18/M48 nunca
  lograron hacer generalizar;
- negación explícita de Rocket Energy (`_rocket_energy_hammer_scores`).

Mismo arquetipo Alakazam, **build distinto** (8 cartas: ellos Night Stretcher ×2, Lillie's
Determination, Neutralization Zone y otra impresión de Dunsparce/Dudunsparce; nosotros
Fezandipiti ex, Shaymin, Enhanced Hammer, Nighttime Mine ×2). Su entrypoint
`agent(obs_dict)` tiene el **mismo contrato** que `BasePolicy`, así que bastó un adaptador
fino (`scratchpad/m50_notebook_arena.py`) — cero reimplementación.

| duelo | resultado | IC 95% |
|---|---|---|
| notebook vs **greedy** (control) | **0,980** (49W-1L) | [0,895, 0,996] |
| campeón vs greedy (referencia) | 1,000 (50W-0L) | [0,929, 1,000] |
| **notebook vs nuestro campeón** | **0,060** (6W-94L) | [0,028, 0,125] |

**El control es lo que hace legible el resultado.** Su agente aplasta a greedy al 98%, con
0 errores y 0 intervenciones de seguridad en el adaptador — **no está mutilado**, juega a
alto nivel. Y aun así pierde 94-6 contra el nuestro. Es el patrón que M11/M37 ya
documentaron: un benchmark saturado (greedy) no distingue entre dos agentes fuertes; el
enfrentamiento directo sí.

**Mecanismo probable, y es concreto:** sus plantillas de creencias cubren Grimmsnarl y
Great Tusk/Crustle — **no el espejo Alakazam**. Contra nuestro campeón, que *es* Alakazam,
su sistema de inferencia de arquetipo está en su punto ciego. Es su peor emparejamiento
posible.

**Salvedad que hay que registrar:** medir contra UN solo rival —y encima el que explota su
punto ciego— no es una evaluación de ladder. No es evidencia de que su agente sea débil en
el campo real; es evidencia de que **no es un upgrade sobre el nuestro** según el único
instrumento que ha predicho el ladder en este proyecto.

## 6. Contexto operativo: la barra de subida subió sola

Durante el milestone el usuario **resubió `imitation-final` a los dos slots**. El mismo
binario, sin cambiar nada del agente, sacó **812,2 y 801,5** contra los 757,4 y 776,7 de
tres días antes — **+35 y +25 puramente por volver a tirar el sorteo del pool de rivales del
día 1**, confirmando en vivo la aritmética de M29/M43 sobre el ruido de campo. Es, con
diferencia, la mayor ganancia obtenida en toda la sesión, y no vino de ningún algoritmo.

Eso cambia el cálculo de cualquier subida futura: un candidato nuevo evictaría un **801,5 ya
convergido y aún subiendo** (M43: el puntaje se fija el día 1 y luego suma ~+37) a cambio de
una tirada que **reinicia en 600**. Históricamente el cierre del día 1 de una subida fresca
ha ido de 684 a 862, así que a 812 ya se está por encima de la mitad de ese rango:
**volver a tirar los dados ahora tiene peores probabilidades que hace 17 h.**

## 7. Recomendación

**No subir nada.** Ninguna cabeza pasó la barra y el agente de la comunidad pierde 94-6
contra el campeón; no hay candidato que llevar a la arena. Los 812,2 / 801,5 siguen
corriendo hasta el domingo y deberían sumar ~+20-30 más por convergencia
(objetivo del usuario: 850 — alcanzable por convergencia sola, sin garantía).

## 8. Reutilizable

- **`player_seats` con conjunto de nombres** (`src/ptcg_ai/imitation/kaggle_replay.py`) —
  cualquier maestro futuro que se renombre queda cubierto; el nombre **no** es una clave
  estable y ahora hay un test que lo fija.
- **`m49_resolve_teacher_names.py`** — deriva los nombres de un equipo desde la API por
  `submissionId` y verifica que ninguno colisione con un rival antes de usarlos como filtro.
- **`m49_verify_teacher_deck.py`** — puerta previa al entrenamiento: identidad de mazo por
  multiconjunto + descomposición piloto-vs-campo del rendimiento, y **falla ruidosamente si
  no parsea nada** en vez de aprobar sobre una muestra vacía.
- El corpus ampliado (`yushinito_full.jsonl.gz`, 3.074 partidas) queda en disco: sirve para
  cualquier experimento futuro aunque no diera nada aquí, y el anterior se preserva.
- **`scratchpad/m50_notebook_arena.py`** — adaptador que corre CUALQUIER agente estilo
  Kaggle (`agent(obs_dict) -> índices`) dentro de nuestra arena, con reset de estado global
  por partida y conteo de excepciones. Cualquier notebook público de la competencia se puede
  medir contra el campeón en minutos, sin reimplementar nada.
- **La disciplina del control en M50**: el 0,060 contra el campeón sólo es interpretable
  porque el 0,980 contra greedy demuestra que el adaptador no mutila al agente. Sin ese
  control, la lectura obvia habría sido "su agente es malo", que es falsa.
