# M33 — Análisis profundo del bot de Yushin Ito (2026-07-29)

Primer estudio dedicado al **maestro**, no al clon. Se hizo porque dos intentos consecutivos de
mejorar el clon (M32 `ALAKAZAM_MATCHUP` y `ALAKAZAM_SPECIALIST`) fallaron construidos sobre una
premisa que **nunca verificamos**. Este documento reemplaza suposiciones por mediciones.

**Datos:** `replays/54773249`, **1551 episodios en disco → 1313 partidas no-espejo** (récord
**747-566 = 56.9%**). **59,136 decisiones MAIN** del maestro parseadas.
**Herramientas nuevas (read-only):** `scratchpad/analyze_yushin.py`,
`scratchpad/analyze_ph_availability.py`, `scratchpad/analyze_yushin_divergence.py`.

---

## 0. Resumen ejecutivo — las tres cosas que invalidan el trabajo previo

1. **No existe una "decisión de cuándo atacar".** Yushin usa Powerful Hand en el **99.6% de los
   turnos en que es legal** (6343/6368), y el retraso entre "se vuelve legal" y "lo dispara" es
   **+0.00 turnos exacto**. Su regla es trivial: *ataca en cuanto puedas*. Los dos intentos
   fallidos trataron de enseñarle al modelo un criterio de disparo que **no existe**.
2. **En los estados propios del maestro, el clon ya es MÁS agresivo que él** — dispararía Powerful
   Hand en 22.7% de las decisiones donde es legal, contra 21.3% del maestro. El clon no es tímido.
3. **Por lo tanto, la brecha real es de ENSAMBLAJE, no de decisión.** El clon tarda ~1 turno más en
   dejar Powerful Hand disponible, y como ambos disparan al instante, ese turno *es* toda la brecha.

---

## 1. El mazo: 60 cartas, y la economía que nadie había analizado

Las docs previas describían ~12 de las 22 cartas únicas. Lista completa (verificada contra
`pokemon-tcg-ai-battle/EN_Card_Data.csv`):

| n | Carta | Rol |
|---|---|---|
| 4 | **Alakazam** (743) | Win condition. *Powerful Hand* {P} = **20 × cartas en mano**. Habilidad *Psychic Draw*: roba **3** al evolucionar. |
| 4 | Kadabra (742) | *Psychic Draw*: roba **2** al evolucionar. |
| 4 | Abra (741) | Base de la línea. |
| 3 | Rare Candy (1079) | Salta Kadabra: Abra → Alakazam directo. |
| 3 | Dunsparce (305) / 2 Dudunsparce (66) | *Run Away Draw*: roba **3** y se baraja a sí mismo al mazo. |
| 1 | Fezandipiti ex (140) | *Flip the Script*: roba **3** si te noquearon algo el turno anterior. *Cruel Arrow* = 100 fijo. |
| 1 | Shaymin (343) | Protege la banca sin Rule Box del daño de ataques. |
| 4 | **Dawn** (1231) | Busca **3 cartas** (básico + fase 1 + fase 2) a la mano. |
| 4 | Hilda (1225) | Busca evolución + energía (**2 cartas**). |
| 4 | Poké Pad (1152) | Busca 1 Pokémon sin Rule Box. |
| 4 | Buddy-Buddy Poffin (1086) | Busca 2 básicos **a la banca** — *no* suma a la mano. |
| 3 | Xerosic's Machinations (1197) | El rival descarta hasta quedarse en **3 cartas**. |
| 4 | Enhanced Hammer (1081) | Descarta una energía especial del rival. |
| 3 | Boss's Orders (1182) | Arrastra un Pokémon de la banca rival al activo. |
| 1 | Night Stretcher (1097) / 1 Sacred Ash (1129) / 1 Lana's Aid (1184) | Recuperación desde el descarte. |
| 2 | Nighttime Mine (1266) | Estadio: ataques de Tera cuestan {C} más. |
| 4 | **Telepath Psychic Energy** (19) | Da {P}; al adjuntarla busca 2 básicos {P} a la banca. |
| 1 | Enriching Energy (13) | Da {C}; al adjuntarla **roba 4 cartas**. |
| 2 | Basic {P} Energy (5) | — |

**Dos rasgos estructurales que importan:**

- **Solo 7 energías en 60 cartas.** Extremadamente bajo. Y Powerful Hand necesita {P} en el
  Alakazam **activo**, así que la energía es un recurso crítico y escaso.
- **La economía de la mano es contradictoria consigo misma:** el arma hace 20 × mano, pero cada
  carta que juegas para desarrollar el tablero **le resta 20 de daño a tu propio ataque**. Todo el
  mazo está construido alrededor de esa tensión (por eso 12 cartas son motores de robo o tutores).

---

## 2. Qué dispara Powerful Hand: **nada — lo usa siempre que puede**

> ⚠️ **Trampa metodológica (lección M20, que este análisis volvió a pisar y luego corrigió).** Medido
> **por decisión**, el uso de Powerful Hand parece 20.4% (6343/31064). Es un artefacto de dilución:
> un turno contiene muchas decisiones de preparación y **como máximo un ataque**. Medido **por
> turno** — la unidad correcta — el número real es **99.6%**.

| Unidad de medida | Tasa de uso de Powerful Hand |
|---|---|
| Por decisión (**diluido, engañoso**) | 6343 / 31064 = **20.4%** |
| **Por turno (correcto)** | 6343 / 6368 = **99.6%** |
| Por turno, cuando es **letal** | 5345 / 5366 = **99.6%** |
| Por turno, cuando **no** es letal | 998 / 1002 = **99.6%** |

**No hay condición.** Ni letalidad, ni tamaño de mano, ni turno, ni matchup cambian nada: si
Powerful Hand es legal, lo usa. El retraso medio entre legalidad y disparo es **+0.00 turnos** en
todos los cortes medidos (maestro global, maestro vs Grimmsnarl, maestro en victorias, en derrotas,
y también el clon).

**Corolario:** el hallazgo previo de que Munkidori no cambia su tasa (18.0% vs 18.1%) era correcto
pero se quedó corto — **nada** cambia su tasa. La pregunta "¿cuándo ataca?" no tiene respuesta
interesante. La pregunta correcta es **"¿cuándo puede atacar?"**.

---

## 3. El cuello de botella real: velocidad de ensamblaje

Powerful Hand requiere **Alakazam en el puesto activo con una energía {P} adjunta**. Medición de
cada hito (`analyze_ph_availability.py`):

| Hito (turno medio) | Maestro (1313) | Maestro vs Grimm (763) | **Clon vs Grimm (27)** |
|---|---|---|---|
| Alakazam en juego | 4.59 | 4.42 | **5.16** |
| Alakazam en el activo | 4.69 | 4.48 | **5.16** |
| Alakazam activo **con {P}** | 4.77 | 4.56 | **5.46** |
| Powerful Hand LEGAL | 4.77 | 4.56 | **5.46** (mediana **6**) |
| Powerful Hand DISPARADO | 4.77 | 4.56 | **5.46** |
| **Retraso legal → disparo** | **+0.00** | **+0.00** | **+0.00** |

**El clon llega ~0.9 turnos tarde, y ya viene atrasado desde el primer hito** (Alakazam en juego:
5.16 vs 4.42). No pierde tiempo dudando — pierde tiempo **montando el combo**.

Y esto es exactamente lo que separa las victorias de las derrotas del propio maestro:

| Maestro vs Grimmsnarl | Alakazam en juego | PH legal |
|---|---|---|
| **Victorias** (396) | 4.13 | **4.21** |
| **Derrotas** (367) | 4.74 | **4.95** |

Cuando el maestro monta el combo en el turno 4 gana; cuando se le va al 5, pierde. **El clon está
sistemáticamente en el lado perdedor de esa línea** (5.46).

---

## 4. La economía de la mano: el clon está muy cerca

Tamaño de mano al inicio de cada turno, maestro vs clon (todos los matchups):

| Turno | Maestro | Clon | Δ |
|---|---|---|---|
| 1-4 | 6.98 / 6.92 / 6.68 / 7.16 | 6.67 / 6.95 / 6.47 / 7.02 | ≈0 |
| 5-8 | 9.02 / 9.18 / 9.81 / 9.76 | 8.65 / 9.14 / 10.11 / 9.05 | ≈0 |
| 9-12 | 11.71 / 11.64 / 12.80 / 13.25 | 12.13 / 11.46 / 12.50 / 12.19 | ≈0 |

Las diferencias (−1.06 a +0.42) son ruido. **El clon construye la mano igual de bien que el
maestro.** Esto descarta "sub-roba / sobre-juega" como explicación.

Al momento de disparar: maestro mano **13.57** (271 daño), clon **12.62** (252 daño). El crecimiento
*dentro* del turno de disparo es maestro **+1.63** cartas vs clon **+1.22** — una diferencia real
pero pequeña (~8 de daño).

**Lo que sí separa victorias de derrotas del maestro vs Grimmsnarl** no es la mano por turno, sino
**cuántas veces logra atacar**:

| | Victorias (396) | Derrotas (367) |
|---|---|---|
| Powerful Hands por partida | **4.61** | **2.86** |
| Mano al disparar | 14.43 | 12.21 |
| Daño por disparo | 288.5 | 244.1 |
| Duración de la partida | 11.55 turnos | 10.04 turnos |

Gana cuando **encadena más ataques, cada uno más fuerte**, en partidas más largas. Pierde cuando lo
atropellan antes de poder encadenar.

---

## 5. Por qué pierde Yushin (566 derrotas, nunca analizado)

| Causa | n | % |
|---|---|---|
| Carrera de premios | 501 | **88.5%** |
| Bench-out | 44 | 7.8% |
| Deck-out | 21 | 3.7% |

Por arquetipo:

| Arquetipo | Derrotas | Desglose |
|---|---|---|
| Marnie's Grimmsnarl | 367 | prize-race 336, bench-out 29, deck-out 2 |
| **Team Rocket** | 93 | prize-race 84, **deck-out 7**, bench-out 2 |
| Mega Kangaskhan | 36 | prize-race 21, **deck-out 9 (25%)**, bench-out 6 |
| Dragapult ex | 14 | prize-race 14 |

**Su peor matchup NO es Grimmsnarl.** Récord actualizado sobre 1313 partidas:

| Arquetipo | Récord | WR | % del campo |
|---|---|---|---|
| Marnie's Grimmsnarl | 396-367 | **51.9%** | **58.1%** |
| **Team Rocket** | 33-93 | **26.2%** | 9.6% |
| Cynthia's Garchomp | 3-8 | 27.3% | 0.8% |
| Mega Kangaskhan | 95-36 | 72.5% | 10.0% |
| Alakazam (espejo) | 56-8 | 87.5% | 4.9% |
| Mega Lucario | 21-5 | 80.8% | 2.0% |

**Team Rocket es un matchup de otra naturaleza**: partidas de **45.5 turnos** en sus victorias
(vs 27.2 en derrotas), con **19.6 Powerful Hands por partida** y el mazo bajando a **11.9 cartas** —
una guerra de desgaste en la que casi se auto-mila. Es la única franja donde el deck-out es un
riesgo estructural real (7 de 93 derrotas allí, y 9 de 36 contra Mega Kangaskhan).

Margen de premios en sus derrotas: 111 a un solo premio de ganar, 48 sin haber tomado ninguno —
distribución amplia, sin un patrón de "colapso" concentrado.

---

## 6. Dónde difiere el clon del maestro, decisión por decisión

`analyze_yushin_divergence.py`, **17,709 decisiones MAIN del maestro** re-puntuadas con el clon
embarcado. Acuerdo global **78.6%** (coincide con la fidelidad conocida de ~0.78 — el instrumento
está bien calibrado).

**Acuerdo según lo que eligió el maestro:**

| El maestro eligió | n | acuerdo |
|---|---|---|
| EVOLVE | 3997 | **92.1%** |
| ABILITY | 2618 | 88.5% |
| ATTACK | 1937 | 83.2% |
| RETREAT | 148 | 70.9% |
| ATTACH | 1694 | **70.4%** |
| END | 660 | 70.2% |
| PLAY | 6655 | **68.4%** |

**Los desacuerdos más frecuentes:**

| Maestro → Clon | n | % de desacuerdos |
|---|---|---|
| **PLAY → PLAY** | **1106** | **29.2%** |
| PLAY → ATTACK | 305 | 8.1% |
| PLAY → ABILITY | 282 | 7.5% |
| ATTACK → PLAY | 243 | 6.4% |
| PLAY → ATTACH | 188 | 5.0% |
| ATTACH → PLAY | 186 | 4.9% |

**El desacuerdo dominante es "ambos quieren jugar una carta, pero cartas distintas" (29.2%)** — no
es sobre atacar o no. Sumando todo lo que involucra PLAY o ATTACH, más del 60% de los desacuerdos
son sobre **qué recurso gastar y en qué orden**.

Los desacuerdos ocurren en las decisiones **más complejas**: mano media 12.15 vs 10.59 cuando
coinciden, y 12.38 opciones disponibles vs 10.65. El acuerdo es prácticamente igual en partidas
ganadas (77.8%) y perdidas (79.5%) — **los desacuerdos no se concentran en las derrotas**.

Y el dato que cierra el caso de M32:

> En las 8,124 decisiones del maestro donde Powerful Hand era legal, **el clon lo dispararía el
> 22.7% de las veces contra el 21.3% del maestro**. Sobre estados idénticos, el clon ya es *más*
> agresivo. El "clon dispara tarde" de sus propias partidas no es una preferencia — es consecuencia
> de llegar tarde al estado.

---

## 7. Síntesis: qué queda invalidado y qué queda abierto

**Invalidado (no volver a intentar):**

- ❌ *"El clon duda / dispara Powerful Hand demasiado tarde."* Ambos disparan con retraso +0.00
  turnos, y sobre los mismos estados el clon es más agresivo (22.7% vs 21.3%).
- ❌ *"El maestro reacciona a Munkidori."* No reacciona a nada: 99.6% siempre.
- ❌ *"Hay que enseñarle un criterio de disparo."* No hay criterio que aprender.
- ❌ *"El clon roba peor / gestiona peor la mano."* Su curva de mano por turno es indistinguible.

**Lo que el dato sí sostiene:**

1. **La brecha es de velocidad de ensamblaje del combo** (~0.9-1.0 turnos), y la línea entre ganar y
   perder para el propio maestro está exactamente ahí (4.21 en victorias vs 4.95 en derrotas).
2. **El déficit está en la selección de recursos**: PLAY→PLAY es el 29.2% de los desacuerdos, y el
   acuerdo en PLAY (68.4%) y ATTACH (70.4%) es el más bajo de todas las categorías, mientras que
   EVOLVE (92.1%) está casi resuelto. Es decir: el clon sabe *cuándo* evolucionar, pero no siempre
   *qué carta jugar* para llegar ahí antes.
3. **Team Rocket (26.2%) es un agujero más profundo que Grimmsnarl (51.9%)** y nunca lo miramos —
   aunque pesa menos en el campo (9.6% vs 58.1%).

**Advertencias de muestra:** el lado del clon es chico (99 partidas, 27 vs Grimmsnarl) — sus
números son direccionales, no definitivos. Los del maestro (1313 partidas, 59k decisiones) son
sólidos. La comparación de trayectoria de mano mezcla campos distintos (el clon juega a ~900 elo,
el maestro a ~1230).

---

## 8. Reproducir

```
uv run --group dev python scratchpad/analyze_yushin.py             # §2, §4, §5
uv run --group dev python scratchpad/analyze_ph_availability.py    # §3
uv run --group dev python scratchpad/analyze_yushin_divergence.py --sample 18000   # §6
```

Chequeos de cordura que deben cuadrar: acuerdo global **78.6%** ≈ fidelidad conocida 0.78; tasa de
PH vs Grimmsnarl por decisión **18.0%** (reproduce el número previo); retraso legal→disparo
**+0.00** en todos los cortes. Nada se construyó, cambió ni subió: los tres scripts son read-only y
no hay cambios en `src/`.
