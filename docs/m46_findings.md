# M46 — el cuello de botella de MAIN falsificado, un dataset externo con datos de Grimmsnarl, y k=7 (2026-08-12/13)

**Resumen: el enriquecimiento de features para MAIN (la aplicación del patrón M34/TO_HAND a MAIN) se
midió con rigor y falló — 7ª hipótesis de conducta falsificada. El diagnóstico con +24h de ladder real
no reveló nada nuevo: salud limpia, Grimmsnarl sigue el muro exacto de siempre. Se descubrió un dataset
público de Kaggle con volcados diarios de replays de todo el ladder, que resuelve el techo de datos que
hundió el intento de clonar Grimmsnarl en M35. Se probó un ensemble k=7 de MAIN — entrenado en GPU vía
Colab tras confirmar que el entrenamiento local en CPU era inviable (>10h proyectadas) — y **también
falló** en la arena decisiva (8ª hipótesis falsificada), pese a una mejora offline real de +0,0076.
`imitation-final` se mantiene sin cambios.**

Continúa [m45_findings.md](m45_findings.md) (cambio de maestro, cerrado).

---

## 1. El cuello de botella de MAIN — hipótesis y por qué parecía prometedora

Auditoría de features (`features.py:208`): en un softmax sobre opciones de MAIN, todo lo constante
entre opciones se cancela. Para dos opciones del **mismo tipo** (jugar Poffin vs jugar Dawn, atachar al
activo vs a la banca), toda la discriminación colapsa a **B⊗R: 9 escalares** — el mismo cuello de
botella que M34 encontró en TO_HAND (y arregló, +0,114). Y M33 midió que PLAY→PLAY es el desacuerdo #1
con el maestro (29,2%) — exactamente las decisiones que pasan por ese cuello.

Dos agujeros concretos citados en el propio código:
- `active_dies_if_pass` solo llega a opciones ATTACK (`features.py:184`) — invisible para RETIRAR,
  ATACHAR, EVOLUCIONAR, JUGAR, PASAR.
- `reduced` (9 escalares) no tiene Abra/Kadabra en mano (las variables que valieron +0,114 en TO_HAND),
  premios absolutos (solo la diferencia), ni reloj intra-turno.

Perfil nuevo `ALAKAZAM_MAIN2` (dim 658→796, +6 escalares en `reduced`, 9→15), aditivo, registrado junto
a los existentes. `tests/unit/test_profile_snapshot_lengths.py` fija los dims.

## 2. G-0 — el triaje que lo cerró

A/B lineal-y-MLP contra el control (mismo split, misma baseline embarcada 0,7712 en ambos):

| perfil | lineal TEST | **MLP x1 TEST** |
|---|---|---|
| MAIN2 (dim 796, features nuevas) | 0,6374 | **0,7632** |
| control (dim 658) | 0,6249 | **0,7558** |
| **delta** | **+0,0125** | **+0,0074** |

**Barra: +0,020. Falla.** Reprodujo exactamente el patrón de M31: lineal +0,031→MLP +0,009 en su caso
(atenuación 3,4×); aquí lineal +0,0125→MLP +0,0074 (atenuación 1,7×). Las features nuevas llevan
información real (delta positivo y consistente en val y TEST; `active_dies_if_pass` es cierto en el
28,1% de las decisiones MAIN, verificado antes de entrenar), pero el MLP ya extraía casi toda esa
señal indirectamente. **Séptima hipótesis de conducta falsificada.** Vía abandonada sin gastar las 6-8h
del entrenamiento grande — que es justo para lo que sirve la puerta.

**Fallo operativo durante G-0** (nota para el futuro): correr los dos brazos EN PARALELO agotó la RAM
(cada `bigX` a dim 796 ≈ 2 GB, con solo 4,6 GB libres) y el sistema mató ambos procesos sin dejar
traza de error — la firma típica de OOM. Arreglado corriendo en secuencia y fijando el L2 del lineal
al ya conocido (1e-4), lo que cortó ese brazo a un tercio.

## 3. Lo que sí sobrevivió: dos contextos que hoy deciden a ciegas

`DISCARD_ENERGY` y `EVOLVE` caían a `_safe_default` (opción 0, sin mirar el estado) por no alcanzar la
barra de +0,020 en M43. Contra esa baseline ciega, con barra ajustada a +0,005:

| contexto | baseline ciega | MLP | lift |
|---|---|---|---|
| DISCARD_ENERGY | 0,9498 | **0,9615** | +0,0117 |
| EVOLVE | 0,9812 | **0,9875** | +0,0063 |

Compuesto en `bc_alakazam_v2.json` (MAIN byte-idéntico, sha verificado, 11 contextos sin cambio).
**Sometido a G-1 (arena n=600 + control) y también falló**: candidato 0,522 vs control 0,512 — ventaja
real de solo +0,010, IC cruzando 0,500. Indistinguible de `imitation-final`. No se sube.

## 4. Diagnóstico con +24h de ladder real (133 partidas combinadas)

Con más del doble de datos que la última vez, el diagnóstico completo (`diagnose_mlp.py`) **no reveló
ningún bug nuevo**:

- Paridad de bundle 100,0% (8.380/8.380), `bc_failures=0`.
- Letalidad 96,3% — **idéntica** a la del maestro (96,3%), así que es el techo del instrumento, no un
  déficit propio.
- **Grimmsnarl sigue el muro exacto de siempre:** 7W-12L = 37% WR, 11 de 12 derrotas por carrera de
  premios. Sin moverse un punto respecto a M32.
- Ensamblaje: Alakazam en derrotas llega en 91,5% de las partidas (clon) vs 96,2% (maestro) — la misma
  brecha de M33/M38, ya atacada con cinco hipótesis distintas, todas falsificadas antes de hoy.

**Conclusión honesta: más datos confirmaron lo ya sabido, no abrieron una puerta nueva.**

## 5. El dataset externo — resuelve el techo de datos de M35

Descubierto por el usuario: Kaggle publica **volcados diarios de replays de todo el ladder**
(`kaggle/pokemon-tcg-ai-battle-episodes-2026-08-11`, formato `kagglehub`, CC0, sin necesitar
credenciales), seleccionados por rating medio del agente. Un solo día trae **4.622 episodios** de
cientos de jugadores distintos — rompe el sesgo de "solo podemos investigar a quien hemos enfrentado".

**Formato 100% compatible** con nuestras herramientas (mismas claves `info.Agents`/`steps`/`rewards`,
mismas celdas `status`/`observation`/`action`) — cero código nuevo necesario para leerlo.

**Hallazgo concreto:** やる気元気ミワハルキ, **138 partidas de Grimmsnarl en un solo día**, WR 61,6%
dentro del volcado, y confirmado en el leaderboard externo del 21-jul: **#82 del mundo, ~1000 elo**.
Más del doble del volumen que hundió a `luca_full.jsonl.gz` (539 partidas totales acumuladas, techo de
datos que M35 documentó como causa de fallo). Mejor maestro que Luca y con más datos — la primera vía
genuinamente nueva contra Grimmsnarl desde M35.

**Pero con una salvedad medida, no solo teórica:** la submission ACTIVA de ese mismo equipo hoy
(`55465468`, hace 4h, implícito **1142**) **ya no juega Grimmsnarl** — cambió a Mega Lopunny ex / Mega
Froslass ex (verificado, cero cartas de la línea Grimmsnarl). Es el **tercer caso** de un jugador top
abandonando su mazo cerca del cierre (Majkel: Alakazam→Lucario; este equipo: Grimmsnarl→Lopunny).
Patrón real del meta cerca del cierre, no casualidad.

**Decisión:** dado el poco tiempo que queda y que no se puede saber de antemano cuántos jugadores
tienen volumen suficiente en el mazo objetivo sin invertir tiempo en buscarlo, se prioriza la palanca
de menor riesgo (§6) sobre abrir una investigación de clonabilidad nueva. El dataset queda documentado
como vía disponible si hubiera más plazo.

## 6. k=7 — probado en GPU (Colab), y también falló

Única palanca sin probar con mecanismo real y sin riesgo teórico de empeorar: ensemble de MAIN de k=3
a k=7, mismo perfil `ALAKAZAM` embarcado (dim 658, sin tocar el que falló en G-0). M37 midió el
barrido de k **monótono (+0,023, k=1→7)** en el set transformer; nunca se barrió para la MLP.

**El entrenamiento local en CPU resultó inviable** — el brazo lineal quedó atascado por *swapping*
(>2h sin terminar un solo valor de L2; la matriz de opciones de MAIN, no solo `dim`, es grande: todas
las opciones de las 62.559 decisiones apiladas). Se añadió `--skip-linear` a
`scratchpad/train_context_heads.py` (aditivo, sin efecto si no se pasa). Aun así, el brazo del MLP en
CPU proyectaba **10-14 h** para las 7 semillas.

**Pivote a Colab (GPU T4).** Ya existía la infraestructura de M31 (`export_for_colab.py` +
`colab_train_mlp.py`, puerto fiel a PyTorch, nunca antes usado en este proyecto para una decisión
real). Export (34 MB) subido manualmente por el usuario; entrenamiento en GPU:

| corrida | tiempo | TEST |
|---|---|---|
| featurización | 171 s | — |
| validación k=3 (GPU) | 96 s | 0,7736 (ref. local 0,7685, dentro de ±0,006 — puerto válido) |
| **k=7 real (GPU)** | **205 s** | **0,7788** |

**Lo que en CPU proyectaba 10-14 h se resolvió en ~5 minutos.** Delta offline sobre el embarcado
(0,7712): **+0,0076**. Señal de alerta que ya apuntaba a un resultado débil: la brecha train-test se
**ensanchó** con más miembros (0,0278→0,0312) en vez de mantenerse o cerrarse, lo contrario de lo que
predice una reducción de varianza real.

**Injerto y bundle.** `colab_train_mlp.py --emit` compone sobre `bc_alakazam_full.json` (la base
antigua, solo lineal) — le faltan ACTIVATE/SWITCH/TO_HAND/conteo de M42-M43. Nuevo
`scratchpad/graft_main_k7.py`: extrae solo el bloque `MAIN` del payload de Colab y lo injerta sobre
`bc_alakazam_final.json`, con verificación sha de que los otros 10 contextos quedan intactos (el mismo
guardián que `build_fetch_weights.py` aplica, pero invertido — aquí SÍ queremos tocar MAIN, a
propósito, con G-1 como puerta). Compuesto con DISCARD_ENERGY + EVOLVE (§3) vía
`build_fetch_weights.py --heads`. 252 tests pasan.

**G-1 — FALLA:**

| | observado | ref | delta |
|---|---|---|---|
| candidato (k=7) vs campeón | 0,488 (292W-306L-2D) | 0,500 | −0,012 |
| control (campeón vs sí mismo) | 0,481 (287W-310L-3D) | 0,500 | −0,019 |
| vs Grimmsnarl: candidato | 0,685 | 0,693 (campeón) | −0,008 |

**Ajustado por el control, el candidato ≈ campeón +0,007 — indistinguible del ruido**, y en Grimmsnarl
específicamente negativo. Reproduce exactamente el patrón de M31 (offline positivo, arena
plana/negativa): el ensemble no reducía varianza real, memorizaba un poco más sin generalizar mejor —
justo lo que la brecha train-test ensanchada ya anticipaba. **Octava hipótesis (contando M31-M45)
falsificada. No se sube. `imitation-final` se mantiene intacto en los dos slots.**

## 7. Reutilizable

- `ALAKAZAM_MAIN2` — perfil descartado pero el patrón de auditoría (qué escalares llegan a qué bloques
  vía A⊗S/B⊗R) es reutilizable para auditar cualquier otro contexto.
- `train_context_heads.py --skip-linear` y `--l2` — recortan el coste de cualquier A/B futuro sobre
  contextos de mucho volumen.
- **El pivote a Colab es la lección operativa más grande del día**: `export_for_colab.py` +
  `colab_train_mlp.py` ya existían desde M31 sin haberse usado nunca para una decisión real de
  producción; convirtieron un entrenamiento de 10-14 h en ~5 minutos. Cualquier reentrenamiento futuro
  de MAIN debería pasar por ahí por defecto, no por CPU local.
- `scratchpad/graft_main_k7.py` — patrón reutilizable para injertar un MAIN nuevo (entrenado donde
  sea) sobre el campeón sin perder las demás cabezas, con el mismo guardián sha de siempre.
- El volcado diario de Kaggle — vía de datos nueva, sin rate-limit, para cualquier candidato futuro.
- La lección operativa sobre RAM local: no correr dos entrenamientos de MAIN en paralelo en esta
  máquina — motivo real por el que Colab es preferible incluso para corridas más chicas.
