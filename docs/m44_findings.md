# M44 — imitation-final en el ladder: la mejora SÍ se materializó, y la deflación no existe

**Resumen: `imitation-final` es medible y sustancialmente mejor en el ladder real — WR 58,6% (51W-36L)
contra 53,6% del par anterior, rating implícito 785-797 contra 743-762. La ganancia contra Grimmsnarl
que se midió en simulación (+0,062) se materializó: de ~30% a 47%. La hipótesis de deflación del
ladder queda REFUTADA con el experimento natural correcto (Yushin, sin tocar su bot, no cae). Y cinco
hipótesis mecánicas sobre la brecha de ensamblaje quedaron falsificadas en una sola sesión.**

---

## 1. La mejora es real y se ve en el ladder

| submission | n | W-L | WR | rival medio | implícito | pendiente |
|---|---|---|---|---|---|---|
| 55369527 mlp-fetch | 77 | 40-37 | 0,519 | 748,2 | 761,7 | +0,65 |
| 55369569 ctx (M42) | 76 | 42-34 | 0,553 | 706,0 | 742,8 | −0,01 |
| **55438687 final** | 44 | 26-18 | **0,591** | 720,8 | **784,6** | **+1,40 SUBIENDO** |
| **55438655 final** | 43 | 25-18 | **0,581** | 739,4 | **796,5** | +0,06 |

**+5 a +7 puntos de WR y +30 a +40 de implícito.** Es la primera vez en el proyecto que una mejora
medida en simulación se confirma en el ladder real con esta claridad.

### 1.1 Por arquetipo — el muro de Grimmsnarl cedió

| arquetipo | WR final | WR ctx | WR mlp-fetch | % campo |
|---|---|---|---|---|
| **Marnie's Grimmsnarl** | **0,471** | 0,353 | 0,250 | 19,5% |
| Alakazam (espejo) | 0,737 | 0,562 | 0,524 | 21,8% |
| Archaludon | 0,800 | 0,833 | 1,000 | 11,5% |
| Mega Lucario | 0,500 | 0,727 | 0,750 | 11,5% |
| Mega Kangaskhan | 0,286 | 1,000 | 0,500 | 8,0% |
| other | 0,167 | 0,000 | 0,600 | 6,9% |

Grimmsnarl era el muro desde M32 (37%, luego 35%, 25%). **Ahora 47%.** Coincide en dirección y
magnitud con el +0,062 medido en la arena n=600 contra el clon de Luca.

### 1.2 Comportamiento vs el maestro — casi todas las brechas cerradas

| eje | ctx (M43) | **final (M44)** | maestro |
|---|---|---|---|
| ataques por partida | 4,313 | **5,191** | 5,640 |
| turno mediano del 1er ataque | 4 | **3** | **3** |
| turnos MAIN por partida | 6,075 | **7,045** | 7,333 |
| Alakazam alcanzado | 0,955 | 0,944 | 0,987 |
| Kadabra turno mediano | 4 | **3** | 4 |
| **Alakazam turno mediano** | 5 | **5** | **4** |

El clon ya **iguala al maestro** en turno del primer ataque, y cerró la mayor parte de la brecha de
agresión y de longitud de partida. **Lo único que queda es el turno de ensamblaje de Alakazam: 5 vs 4.**

### 1.3 Salud del agente embarcado — LIMPIA (tercera confirmación)

- **Paridad de bundle 100,0% (5517/5517)**, en victorias y en derrotas por separado.
- `bc_failures = 0`; enrutado 5844 aprendido / 267 greedy.
- **Disciplina letal 97,6%** vs **96,3% del maestro** — mejor que él.

No hay fallo estructural. La rama con techo de +100 (el precedente M37) está cerrada.

---

## 2. La hipótesis de deflación del ladder — REFUTADA

El experimento natural correcto: **Yushin Ito (54773249) no ha tocado su bot en semanas.** Si el
ladder deflacionara, su rating caería solo.

| | n | WR | rival medio | implícito | pendiente |
|---|---|---|---|---|---|
| Yushin Ito | 1000 | 0,561 | 1112,6 | 1155,2 | **+0,08 (PLANO)** |

Serie diaria: 1187 → 1197 → 1129 → 1148 → 1132 → 1083 → 1116 → 1181 → 1187 → 1175 → 1140.
**Oscila ±55 alrededor de ~1150 y no cae sistemáticamente.** Lo que se ve como "declive" en agentes
ajenos es la oscilación normal de este sistema de rating, no deflación.

### 2.1 El hallazgo que reencuadra el problema

**Nuestro WR (0,586) es MAYOR que el de Yushin (0,561).** La diferencia de 370 elo entre él y nosotros
no está en ganar más partidas: está en **contra quién nos emparejan** (rivales de 730 vs 1112). El
sistema empareja por rating, así que el equilibrio se alcanza cuando WR≈50% contra la propia piscina.
Con WR 58,6% contra una piscina de ~730, el equilibrio matemático cae en ~790 — exactamente donde
estamos. **Para subir hay que ganar más contra rivales fuertes, no simplemente ganar más.**

Por banda de rating del rival (n=87, submuestras pequeñas):

| banda | n | WR |
|---|---|---|
| 650-725 | 16 | 0,812 |
| 725-800 | 38 | 0,526 |
| 800-875 | 14 | 0,286 |

*Advertencia honesta:* con n=14-16 por banda el error estándar es ~0,13, así que el −0,150 de la banda
800-875 respecto a lo que predice Elo es solo ~1,1σ. **No está establecido** que el agente sea
"frágil" contra rivales fuertes; no se construyó nada sobre esa lectura.

---

## 3. Cinco hipótesis de ensamblaje, todas falsificadas

La brecha del turno de Alakazam (5 vs 4) lleva viva desde M33 y ha resistido:

| # | hipótesis | resultado | milestone |
|---|---|---|---|
| 1 | elige mal QUÉ carta buscar | cambio de conducta verificado, 0 efecto | M34 |
| 2 | pierde piezas / recupera peor | recupera igual o mejor que el maestro | M40 |
| 3 | **no usa Rare Candy para saltarse Kadabra** | **clon 28,5% vs maestro 26,0% — la usa MÁS** | M44 |
| 4 | **consume el mazo más rápido (deck-out)** | **curvas idénticas; el maestro llega a ≤5 cartas MÁS a menudo (31,3% vs 25,8%)** | M44 |
| 5 | **evoluciona a Kadabra temprano y destruye el atajo de Rare Candy** | **con ambas legales, el maestro evoluciona MÁS (33,3% vs 23,1%)** | M44 |

El deck-out (26% de nuestras derrotas vs 3,7% de las suyas) resultó ser **síntoma, no causa**: no
quemamos el mazo más rápido, simplemente no cerramos la partida a tiempo y el mazo se acaba.

Lo único que queda medido y sin explicar: con Kadabra y Rare Candy ambos legales, el maestro toma una
de las dos el **66%** de las veces y el clon solo el **48,7%** — se compromete menos con el combo. Es
la misma brecha difusa PLAY→PLAY que M33 midió (29,2% de los desacuerdos) y que ninguna intervención
ha cerrado nunca.

---

## 4. Reutilizable

- **`scratchpad/rating_trajectory.py`** — rating implícito (`opp_mean + 400·log10(WR/(1-WR))`),
  pendiente de convergencia, cuartiles y serie diaria para cualquier submission, propia o ajena.
  Convierte en herramienta lo que en M29/M43 se hizo con scripts desechables. `--by-day` sobre un
  competidor que no ha tocado su bot mide la deriva del SISTEMA, no del agente.
- El patrón de **falsificar barato antes de construir**: cinco hipótesis descartadas en una sesión sin
  escribir una línea de `src/`. Las tres de M44 costaron ~2 minutos cada una.

## 5. Estado operativo

`imitation-final` está en los dos slots (55438687, 55438655), 7 h de ladder, sin fallos, y **una de las
dos sigue subiendo (+1,40/episodio)**. No hay ninguna palanca identificada y verificada pendiente de
construir: las siete de M43 se agotaron (1 pasó, 6 fallaron) y las cinco hipótesis de ensamblaje están
falsificadas.
