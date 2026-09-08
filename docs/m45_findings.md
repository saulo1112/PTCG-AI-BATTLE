# M45 — cambiar de maestro, cerrado con medición (2026-08-12)

**Resumen: se investigaron dos candidatos a maestro distintos de Yushin — Majkel1337 (#1 del mundo) y
Yan (clonabilidad más alta jamás medida en el proyecto). Ninguno mejora sobre `imitation-final`. La vía
de "cambiar de maestro" queda cerrada con evidencia, no por descarte.**

Continúa [m44_findings.md](m44_findings.md) (diagnóstico en vivo de `imitation-final`, deflación del
ladder refutada).

---

## 1. Majkel1337 — descartado antes de construir nada

M23 había identificado a Majkel1337 (#1, 1231 elo en su día) jugando el mazo **byte-idéntico** al
nuestro (`decks/yushinito.csv`), verificado sobre 40 partidas reales de su submission `54909711`.
Nunca se clonó — M23 lo descartó por la puerta de *greedy pilotability*, la misma puerta que M22
degradó a triaje por no predecir nada.

**Rating implícito de las dos submissions suyas con datos completos:**

| maestro | episodios | WR | rival medio | implícito |
|---|---|---|---|---|
| Yushin (el que clonamos) | 1000 | 0,561 | 1112,6 | **1155** |
| Majkel `54909711` (jul, Alakazam) | 1000 | 0,595 | 1058,1 | 1125 |
| Majkel `55333348` (ago, ACTIVA) | 273 | 0,560 | 1168,0 | 1210 |

**El hallazgo que lo cierra:** `55333348` — su submission activa, la de mayor implícito — **ya no
juega Alakazam**. Verificado sobre 40 partidas reales: se pasó a **Mega Lucario ex** (4x Lucario ex,
3x Riolu, Solrock/Lunatone/Makuhita/Hariyama), cero cartas de la línea Alakazam. Y con n=1000 cada uno,
el error estándar del WR es ~28 elo — **1125 vs 1155 es ~1σ, estadísticamente equivalentes** como
maestros de Alakazam. No hay upgrade que clonar.

Sobre su Lucario: mejor piloto de Lucario que hemos enfrentado en todo el proyecto está en 831 elo
(barrido de 33 pilotos rivales reales), y `55333348` solo tiene 273 episodios — la mitad de los 539 con
los que M35 ya falló por techo de datos. Sin corpus alternativo. Descartado sin construir nada.

## 2. Yan (Teal Mask Ogerpon ex) — la clonabilidad más alta jamás medida, y aun así pierde

Candidato encontrado buscando en la franja 830-1000 (la banda que M23 identificó como dulce para
clonar, no el top absoluto). `55235071`: implícito **997,2** (293 episodios), pico 1032,9, mazo de un
solo atacante (4x Ogerpon ex + 1 Tapu Bulu, 19 cartas distintas — mucho más simple que el combo de 3
etapas de Alakazam).

**Cribado (`quick_screen.py`), con datos completos (sin submuestreo):**

| | fuerza de mazo | clonabilidad MAIN (perfil genérico, lineal) |
|---|---|---|
| **Yan** | **0,722** (barra 0,40) | **0,813** (barra 0,75) |
| Yushin (perfil a mano + MLP) | — | 0,780 |

**Primer candidato del proyecto que pasa las dos puertas del screen.** Su lineal con perfil genérico
saca más que el lineal a mano de Yushin (0,556 en M28) y casi iguala el MLP completo de Yushin.

**La puerta que decide, jugando (arena n=600, con control de ruido):**

| | resultado | ref | delta |
|---|---|---|---|
| **Yan vs imitation-final** | **0,275** (110W-290L) | 0,500 | **−0,225** |
| control (imitation-final vs sí mismo) | 0,505 | 0,500 | +0,005 |

El control confirma que el instrumento está calibrado — la derrota es real, no ruido. **0,813 de
clonabilidad offline, la más alta jamás medida en este proyecto, se traduce en perder 27,5%-72,5%.**

## 3. Conclusión

Dos maestros investigados desde ángulos opuestos — el mejor rating disponible (Majkel) y la mejor
clonabilidad jamás medida (Yan) — y ninguno mejora `imitation-final`. Confirma otra vez la lección de
M8/M22/M23: la clonabilidad offline no predice el resultado jugando. **La vía de "cambiar de maestro"
queda cerrada con medición, no por falta de intentarlo.**

## 4. Reutilizable

- `scratchpad/test_yan_vs_final.py` (luego generalizado a `test_ctx_heads.py`) — arena directa +
  control de ruido contra un candidato con perfil genérico registrado en caliente
  (`dp.build_generic_profile`).
- El patrón de verificación de mazo (extraer la acción de 60 cartas del propio replay, comparar contra
  `decks/yushinito.csv`) — reutilizado en M45 y M46 para confirmar/descartar candidatos antes de
  invertir tiempo.
