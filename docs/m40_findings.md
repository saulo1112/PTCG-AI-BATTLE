# M40 — hypothesis A (piece loss/resilience) measured and closed: negative (2026-08-08)

**Resumen: la segunda y última hipótesis abierta de M33 sobre por qué el clon arma su combo más
lento que Yushin (turno 5,46 vs 4,56) queda falsificada con medición directa. Perder una pieza del
combo SÍ cuesta caro (+1,71 turnos, confirmado sobre 2.330 partidas del maestro), pero el clon NO
recupera peor que Yushin cuando eso pasa — al contrario, recupera con la misma frecuencia o algo
más (28,4% vs 21,8%, intervalos solapados). La puerta de tres condiciones fijada antes de correr
nada falla en la condición central. No se construye nada. Con esto, las dos hipótesis que M33 dejó
abiertas (B: mala priorización — cerrada en M34; A: pérdida de piezas — cerrada aquí) están
falsificadas con evidencia directa, y no queda ninguna hipótesis identificada sobre la política de
decisión sin probar.**

Continúa [m39_findings.md](m39_findings.md), que cerró la línea de mejora de fidelidad/RL con cinco
palancas nulas o negativas. Este milestone es distinto: no prueba una arquitectura o un
entrenamiento, prueba **la última causa raíz plausible** que quedaba de M33/M34.

---

## 1. Por qué esta pregunta y no otra

M33 dejó dos hipótesis abiertas para el retraso de ~0,9 turnos en el ensamblaje de Alakazam:

- **(B) mala priorización/secuenciación de jugadas** — el clon elige mal qué carta jugar dado el
  mismo estado que vio Yushin.
- **(A) pérdida de piezas por disrupción del rival** — el clon pierde copias de Abra/Kadabra al
  ataque o descarte del rival con más frecuencia, o se recupera peor cuando eso pasa.

M34 (2026-07-29) atacó (B) con el instrumento más riguroso del proyecto hasta la fecha: construyó
un perfil de features que sí distingue "¿ya tengo un Abra/Kadabra en mano?" (`ALAKAZAM_FETCH`),
**verificó que la intervención cambiaba el comportamiento** (cerró 57% de la brecha de elección de
carta) y aun así **no aceleró el ensamblaje ni ganó más partidas** en el campo real (Grimmsnarl:
candidato +0,23 turnos más lento, no significativo pero en la dirección contraria). Eso **falsificó
(B)**.

Quedaba (A) sin medir. Es la razón de este milestone: no es una palanca nueva sobre MAIN (las seis
anteriores fracasaron todas ahí), es la última causa raíz sin probar.

### Por qué era viable medirlo rápido

- El mazo de Yushin **sí lleva recuperación**: Night Stretcher (1097) ×1, Sacred Ash (1129) ×1
  (`decks/yushinito.csv`). Sin esto la hipótesis sería imposible de arreglar en este mazo.
- Ya existe un mecanismo de recuperación forzada en `ImitationPolicy._forced_recovery`
  (`src/ptcg_ai/imitation/policy.py:255-284`, M27): fuerza jugar un ítem de recuperación cuando la
  carta "reloj" está en el descarte y no en juego/mano. M27 lo probó en el mazo de Kanga y encontró
  que el clon congelado casi nunca lo usaba (7-9%) aunque lo tuviera en mano — pero nunca se probó
  en Alakazam.

---

## 2. El montaje

`scratchpad/analyze_piece_loss.py` (solo lectura, sin cambios en `src/`), sobre los replays crudos
reales de ambos:

- **maestro**: `replays/54773249/` (Yushin Ito), **2.330 partidas**
- **clon**: `replays/55145833/` (nuestro agente en el ladder), **133 partidas**

Definiciones, reutilizando instrumentos ya calibrados en vez de inventar nuevos:

- **turno de ensamblaje** = primer turno en que Powerful Hand (id de ataque 1072) es una opción
  MAIN legal — la misma métrica que M33/M34 usaron y validaron contra replays reales.
- **evento de pérdida** = primer turno en que una copia de Abra (741) o Kadabra (742) aparece por
  primera vez en el descarte, antes del turno de ensamblaje. Alakazam (743) se excluye a propósito:
  es la condición de victoria cambiando de zona (banca↔activo), no una pérdida de material.
- **ítem de recuperación** = Night Stretcher o Sacred Ash — los dos que de verdad están en el mazo
  de 60 cartas embarcado.

## 3. El resultado

| | Maestro (Yushin) | Clon |
|---|---|---|
| Partidas analizadas | 2.330 | 133 |
| Con evento de pérdida | 20,8% (485) | 25,6% (34) |
| Tasa de ensamblaje | 97,8% | 95,5% |
| Turno de ensamblaje **con** pérdida | **6,11** (n=463) | 5,78 (n=32) |
| Turno de ensamblaje **sin** pérdida | **4,40** (n=1.815) | 4,69 (n=95) |
| Oportunidades de recuperación | 1.286 | 67 |
| **Tasa de recuperación** | **21,8%** [19,6–24,1%] | **28,4%** [19,0–40,1%] |

### La puerta pre-registrada (fijada antes de correr nada)

| condición | criterio | resultado |
|---|---|---|
| **[1]** | tasa de recuperación del clon ≪ la del maestro (brecha > 0,30) | **FALLA** — clon 28,4% vs maestro 21,8%, en la dirección contraria |
| **[2]** | la pérdida sí retrasa el ensamblaje del maestro (≥0,5 turnos) | PASA — delta 1,71 |
| **[3]** | suficientes casos (≥20 oportunidades cada uno) | PASA — 67 y 1.286 |

**Veredicto: NO construir.** Basta con que falle una condición; falla la central.

## 4. Interpretación

**El mecanismo es real: perder una pieza cuesta 1,71 turnos.** Eso confirma que (A) es una fuerza
genuina en el juego — la hipótesis no estaba mal planteada. Pero **el clon no está fallando en
reaccionar a ella**: recupera con la misma frecuencia que Yushin, si acaso un poco más (aunque los
intervalos se solapan lo suficiente para no afirmar una diferencia real en ningún sentido).

**Efecto colateral útil, no anticipado:** como Yushin tampoco juega el ítem de recuperación cada
vez que puede (22% de las veces, no 100%), forzar su uso automático —el arreglo que estaba medio
construido reutilizando el mecanismo de M27— habría hecho que el clon se pareciera **menos** a
Yushin, no más. La regla de M27 asume una decisión binaria y clara ("la pieza reloj desapareció,
recupérala ya"); aquí la decisión real es más matizada — Yushin guarda el ítem para el momento
táctico correcto, no lo dispara automáticamente. Bueno que se midiera antes de construir: habría
sido una octava intervención con la misma forma que las siete anteriores, "cambia algo, mide
después", y esta vez la medición previa la evitó por completo.

## 5. Balance final: las dos hipótesis de M33, ambas falsificadas

| hipótesis | mecanismo | ¿el clon falla ahí? | milestone |
|---|---|---|---|
| (B) mala priorización de jugadas | verificado — la intervención cambió el comportamiento | **NO** — el cambio no aceleró el ensamblaje | M34 |
| (A) pérdida de piezas / recuperación | verificado — perder una pieza cuesta 1,71 turnos | **NO** — recupera igual o mejor que el maestro | M40 (este) |

**No queda ninguna hipótesis identificada, en todo el proyecto, sin probar sobre la política de
decisión.** Sumado a las cinco palancas de fidelidad/arquitectura/RL ya cerradas (M31, M36, M37,
M38, M39), son siete intentos de intervención más dos pruebas de causa raíz, todos nulos o
negativos. La recomendación operativa (ver `docs/handoff.md`) es consolidar el envío final en vez
de seguir buscando una novena palanca sobre el mismo cuello de botella ya diagnosticado.

## 6. Reutilizable pase lo que pase

- `analyze_piece_loss.py` — patrón de medición directa sobre replays crudos (maestro y clon) sin
  pasar por el dataset de entrenamiento, reutilizable para cualquier pregunta futura de "¿el clon
  reacciona igual que el maestro ante la condición X?".
- La puerta pre-registrada de tres condiciones evitó construir una intervención sobre una hipótesis
  que, mirada de cerca, apuntaba en la dirección contraria (el clon recupera *más*, no menos).
