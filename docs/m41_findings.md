# M41 — cambio de composición del mazo: NEGATIVO (2026-08-08)

**Resumen: sustituir Enhanced Hammer por Night Stretcher en el mazo de Yushin, con los pesos
congelados, NO mejora nada. Con n=200 el cambio parecía ganar (+0,115 en el espejo, POOLED +0,012);
al repetirlo con n=600 el efecto se evaporó por completo (−0,023 en el espejo, POOLED −0,013). Nada
adoptado, nada subido. El valor duradero del milestone es metodológico: es la segunda vez en la misma
sesión que una señal a n pequeño resulta ser ruido, y esta vez se atrapó ANTES de gastar un envío.**

Continúa [m40_findings.md](m40_findings.md), que cerró la última hipótesis causal sobre la política de
decisión. Agotadas las palancas de política, esta es la única otra superficie que quedaba: **el mazo**.

---

## 1. La hipótesis

M40 confirmó un mecanismo real: perder una pieza del combo (Abra/Kadabra) antes del ensamblaje le
cuesta al maestro **+1,71 turnos**. M40 también mostró que el clon **no** se recupera peor que él, así
que no había nada que arreglar en la política. Pero quedaba la otra mitad: si perder una pieza es tan
caro, **¿más cartas de recuperación ayudarían?**

El mazo de Yushin lleva Night Stretcher ×1 y Sacred Ash ×1. La ranura más blanda es **Enhanced Hammer
×4** ("descarta una Energía Especial del rival") — situacional, y muerta contra los mazos del campo
que no juegan Energía Especial.

Dos variantes probadas:

| variante | cambio | archivo |
|---|---|---|
| v2 | −2 Enhanced Hammer, +2 Night Stretcher | `decks/yushinito_v2stretcher.csv` |
| v3 | conservadora, 1 por 1 | `decks/yushinito_v3stretcher.csv` |

## 2. El montaje — partidas reales, no un proxy offline

`scratchpad/test_deck_variant.py`. **Deliberadamente NO es otro proxy de fidelidad offline**: ocho de
esos ya han fallado en predecir el ladder en este proyecto. Motor real, partidas reales, **los mismos
pesos congelados** (`bc_alakazam_fetch.json`) en ambos brazos; lo único que cambia es `deck.csv`.
Oponentes: espejo (`yushinito`), Grimmsnarl (`luca`) y al principio Kangaskhan.

## 3. El resultado — y por qué n=200 mintió

**Primera corrida, n=200 por oponente y por mazo** (`scratchpad/_m41_v3.log`):

| oponente | mazo viejo | mazo nuevo | delta |
|---|---|---|---|
| espejo | 0,435 | 0,550 | **+0,115** |
| grimmsnarl | 0,670 | 0,655 | −0,015 |
| kangaskhan | 0,255 | 0,190 | −0,065 |
| **POOLED** | | | **+0,012** |

El +0,115 del espejo es enorme y tentador. **Es ruido.**

**Segunda corrida, n=600** (`scratchpad/_m41_v3_n600.log`; Kangaskhan retirado por ser ~6% del campo
real):

| oponente | mazo viejo | mazo nuevo | delta |
|---|---|---|---|
| espejo | 0,503 | 0,480 | **−0,023** |
| grimmsnarl | 0,658 | 0,655 | −0,003 |
| **POOLED** | | | **−0,013** |

El signo se invirtió y la magnitud se desplomó a cero. Fíjate además en el **mazo viejo contra sí
mismo**: marcó **0,435** con n=200 y **0,503** con n=600. Un espejo tiene que valer 0,500 por
construcción — así que ese 0,435 mide directamente el ruido del instrumento a n=200, y es de la misma
magnitud que el "efecto" que estábamos a punto de creernos.

## 4. Conclusión

**No hay mejora real. Ninguna variante adoptada; el mazo embarcado sigue siendo `decks/yushinito.csv`.**
Los CSV se dejan en el repo como registro, marcados como NO adoptados.

Nota adicional que refuerza el resultado: M42 verificó después que Yushin usó **una sola lista** en sus
2.330 partidas, byte-idéntica a la nuestra. Un jugador de 1179 elo con ese volumen de partidas ya
exploró su propio espacio de mazo y se quedó con este; que nuestras variantes no mejoren es coherente.

## 5. Lo reutilizable

- `scratchpad/test_deck_variant.py` — patrón de A/B de mazo con pesos congelados. Sin argumentos CLI a
  propósito: se editan las constantes, para que quede en el log exactamente qué se comparó.
- **La regla operativa, ahora con dos confirmaciones en la misma sesión** (RL en M38 y el mazo aquí):
  una señal a muestra pequeña en este motor casi siempre es ruido. Señal chica primero para decidir si
  vale la pena mirar; **confirmar con n grande antes de recomendar subir**. El espejo contra sí mismo
  es el mejor calibrador de ruido disponible y es gratis: debe dar 0,500.
