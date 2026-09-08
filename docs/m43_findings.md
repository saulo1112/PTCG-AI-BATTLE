# M43 — por qué el ladder se estancó, y la mayor mejora medida del proyecto (2026-08-11)

**Resumen: el clon NO se rompió y M42 no lo empeoró. El diagnóstico de las 76 partidas reales de
`imitation-ctx` sale con auditoría de salud LIMPIA (paridad de bundle 100,0%, 0 fallos). El
estancamiento es de sorteo de rivales y de episodios, no de calidad. Completar el programa de M42
sobre los contextos que faltaban (TO_HAND con capacidad + SWITCH) produce un candidato que le gana al
campeón **0,596 en espejo directo a n=600, IC 95% [0,556-0,634]** — el primer candidato del proyecto
que gana con significancia al 95%, y equivalente a ~+67 elo.**

---

## 1. El diagnóstico del estancamiento

Rango real: **1437 / 6725**; el bronce es top 10% ≈ rango 672.

### 1.1 Rating implícito de cada subida (`opp_mean + 400·log10(WR/(1-WR))`)

| submission | n | W-L | WR | rival medio | implícito | score |
|---|---|---|---|---|---|---|
| 55145833 (31-jul) | **164** | 90-74 | 0,549 | 854,1 | **888,1** | 899,5 |
| 55225391 (4-ago) | 156 | 78-78 | 0,500 | 866,5 | 866,5 | 864,9 |
| 55334177 (7-ago) | 63 | 33-30 | 0,524 | 801,8 | 818,3 | 818,4 |
| 55334194 (7-ago) | 80 | 43-37 | 0,537 | 786,3 | 812,4 | 812,0 |
| 55369527 mlp-fetch | 77 | 40-37 | 0,519 | 748,2 | 761,7 | 750,9 |
| 55369569 ctx (M42) | 76 | 42-34 | **0,553** | 706,0 | 742,8 | 743,3 |

**M42 no empeoró nada.** `ctx` ganó MÁS partidas que el campeón concurrente (55,3% vs 51,9%); su
implícito es menor solo porque le tocaron rivales más débiles (706 vs 748). Compartieron **1 rival de
~76** — piscinas casi disjuntas, exactamente M29.

### 1.2 Lo que sí explica la caída

- **El puntaje se decide el día 1.** `55145833` cerró el día 1 en **862,9** y solo ganó **+36,6** en
  los 4 días siguientes. Esperar vale ~+37, no +140.
- **Lo que cambia es el pool de siembra del día 1**: rival medio 794,8 (la que llegó a 899) vs ~684
  (ctx). 111 puntos de diferencia que se propagan al score final.
- **Cada subida mata el flujo de la anterior.** `55145833` jugaba 58/34/36/33 por día y **se detuvo en
  seco el 4-ago**, justo al subir la siguiente, con sus últimos 10 scores en 875→899 (subiendo).

### 1.3 Auditoría de salud — LIMPIA, no hay fallo estructural

Sobre los 67 replays reales de 55369569 (`diagnose_mlp.py`, re-apuntado vía `PTCG_LIVE`/`PTCG_WEIGHTS`):

- **Paridad de bundle 100,0% (4008/4008)** — el agente del ladder ES el clon entrenado.
- `bc_failures = 0`; enrutado 4241 aprendido / 197 greedy.
- Disciplina letal **95,9%** vs **96,3%** del maestro.

**Esto cierra la única rama con techo de +100.** No hay nada roto que arreglar.

### 1.4 El campo actual

| arquetipo | W-L | WR | % del campo |
|---|---|---|---|
| **Marnie's Grimmsnarl** | 6-11 | **35%** | 25% |
| Alakazam (espejo) | 9-7 | 56% | 24% |
| Mega Lucario | 8-3 | 73% | 16% |
| Archaludon | 5-1 | 83% | 9% |
| Dragapult ex | 2-3 | 40% | 7% |

Grimmsnarl sigue siendo el muro (M32 midió 37%, hoy 35%). Ganarlo del 35% al 60% valdría ~+54 elo.

### 1.5 Hipótesis Rare Candy — FALSIFICADA

El maestro llega a Alakazam el mismo turno que a Kadabra (mediana 4 y 4) y el clon tarda uno más
(4 y 5), lo que sugería que el clon no usaba Rare Candy para saltarse Kadabra. **Falso:** el clon la
juega en **28,5%** de las oportunidades y el maestro en **26,0%**; turno mediano de la primera, 4 en
ambos. Tercera hipótesis de ensamblaje que cae (tras M34 y M40).

---

## 2. Completar el programa de M42

M42 sólo había cubierto ACTIVATE y SETUP_BENCH. Se entrenaron cabezas para los 7 contextos restantes
con `train_context_heads.py` (barra pre-registrada +0,020 sobre el artefacto EMBARCADO):

| contexto | /partida | baseline | candidato | lift | veredicto |
|---|---|---|---|---|---|
| **SWITCH** | 1,81 | lineal 0,6107 | **MLP 0,7611** | **+0,1503** | **PASA** |
| DISCARD_ENERGY | 1,19 | greedy 0,9498 | 0,9615 | +0,0117 | falla |
| EVOLVE | 1,04 | greedy 0,9812 | 0,9875 | +0,0063 | falla |
| TO_ACTIVE | 3,59 | lineal 0,9232 | 0,9272 | +0,0041 | falla |
| TO_BENCH | 2,91 | lineal 0,9129 | 0,9005 | −0,0123 | falla |
| DISCARD | 0,56 | lineal 0,3579 | 0,3653 | +0,0074 | falla |
| TO_DECK | 0,58 | lineal 0,8030 | 0,7695 | −0,0335 | falla |

**1 de 7.** SWITCH es el segundo mayor salto de contexto del proyecto tras MAIN en M28, y es
tácticamente crítico: es qué Pokémon promueves cuando te noquean el activo.

## 3. El bundle final

`build/imitation-final.tar.gz` = campeón + 4 cabezas, **MAIN byte-idéntico** (sha `a206d7ed9c8e2181`):

| contexto | /partida | antes | ahora | lift |
|---|---|---|---|---|
| TO_HAND | 12,4 | lineal 0,7517 | **MLP 0,7949** | +0,0432 |
| ACTIVATE | 6,55 | sin modelo 0,9408 | MLP 0,9742 | +0,0335 |
| SWITCH | 1,81 | lineal 0,6107 | MLP 0,7611 | +0,1503 |
| SETUP_BENCH | 0,45 | greedy 0,4706 | linear+count 0,9804 | +0,5098 |

≈ **1,26 decisiones corregidas por partida** (casi 3× el bundle A+B de M42).

### La puerta que decide

| | resultado | referencia | delta |
|---|---|---|---|
| **final vs campeón (espejo, n=600)** | **0,596** (357W-242L) | 0,500 | **+0,096** |
| campeón vs campeón (control de ruido) | 0,490 | 0,500 | −0,010 |

**IC 95% [0,556 — 0,634], enteramente por encima de 0,500.** Primer candidato del proyecto que gana
con significancia al 95%; ≈ **+67 elo**. El control valida el instrumento (0,490 aquí, 0,492 en la
corrida de M42).

`head_to_head_par` (campo pilotado por greedy, saturado 0,94-0,96): macro −0,009 [−0,027, +0,008]
empate, ponderado por frecuencia +0,009. **No veta**, pero no discrimina — es el instrumento que M25
demostró que no ordena en la banda ±0,05.

### 3.1 Réplica independiente — más fuerte todavía, y el muro de Grimmsnarl cede

Segunda corrida completa, motor estocástico (no reseedable), mismo `build/imitation-final.tar.gz`:

| | resultado | referencia | delta |
|---|---|---|---|
| final vs campeón (espejo, n=600) | **0,617** (370W-230L) [95% 0,577-0,655] | 0,500 | **+0,117** |
| campeón vs campeón (control) | 0,516 [0,476-0,556] | 0,500 | +0,016 |
| **final vs Grimmsnarl** (n=600) | **0,743** [0,707-0,777] | — | — |
| campeón vs Grimmsnarl (n=600) | 0,682 [0,643-0,718] | — | **+0,062** |

**Dos corridas independientes, ambas con el espejo enteramente sobre 0,500** (0,596 y 0,617) — no es
un pico de ruido, es una señal que se repite. Y por primera vez en el proyecto el candidato mejora
**contra el emparejamiento que ha sido el muro desde M32** (0,743 vs el 0,682 del campeón, IC casi sin
solape: el techo del campeón, 0,718, queda debajo de la media del candidato). El `head_to_head_par`
también se corrió dos veces con resultados distintos (−0,009 tie / +0,017 positivo) — ambos dentro del
ruido esperado de un instrumento saturado (M25), ninguno veta, ninguno cambia la conclusión.

`head_to_head_par` réplica (bundle final completo, campo greedy): macro **+0,017 [+0,005, +0,030]**
positivo, sin Grimmsnarl **+0,019 [+0,004, +0,033]** positivo — 0 intervenciones en 3.456 partidas.

## 4. Recomendación operativa

1. **Subir `build/imitation-final.tar.gz` a los DOS slots.** Con el leaderboard tomando tu mejor
   submission y un ruido de sorteo de ±60-100, dos slots del mismo archivo son dos tiradas
   independientes y te quedas con el máximo (E[máx de 2] ≈ media + 0,56·SD ≈ +35 gratis).
2. **Y después no subir absolutamente nada más.** Cada subida corta el flujo de episodios de la
   anterior; el score se fija el día 1 y luego gana ~+37 si se le deja correr.

**Honestidad sobre el bronce:** +67 (mejora) +37 (dejarlo correr) parte de ~750 y llega a ~850. El
bronce exigía superar el rango 672, y el mejor histórico (899) hoy vale rango 1437. Esto es la mayor
mejora medida del proyecto y **aun así probablemente no alcanza el bronce**. Es lo mejor que la vía
del clon puede dar con la evidencia disponible.
