# M39 — reweighting by decision-importance: also a wash (2026-08-05)

**Resumen: reponderar las decisiones de armado del combo (peso ×3 en las jugadas previas a
que Powerful Hand se vuelva legal, el hueco que M33 midió como decisivo: turno 4,56 vs
5,46) no movió la fidelidad estricta. Control 0,7670, con peso 0,7659 — una diferencia de
0,11 puntos, más pequeña que la varianza entre semillas del mismo entrenamiento (~0,8
puntos). Es el QUINTO intento consecutivo de mover la fidelidad de imitación (tras M31
anchura, M36 datos, M37 arquitectura, M38 RL) que no produce una ganancia medible.**

Continúa [m38_findings.md](m38_findings.md), que cierra además la segunda mitad del
experimento de RL: la reponderación de oponentes (30/30/40 en vez de 40/35/25) para
corregir el colapso contra Kangaskhan **también empeoró**, con una medición mucho más
confiable que las anteriores.

---

## 1. Cierre de M38: la reponderación de oponentes en el RL fue negativa, con confianza

Tras el veredicto ambiguo de M38 (0,550 contra el modelo congelado, dentro del margen de
error, con Kangaskhan colapsado de 27,5% a ~19%), se identificó la causa: `rl_update`
mezcla las decisiones de todos los oponentes en un solo gradiente, ponderado solo por
cuántas partidas juega cada uno. Kangaskhan tenía el peso más bajo (25%), así que su señal
quedaba sistemáticamente superada por Grimmsnarl y el espejo.

Se subió su peso a 40% (bajando los otros dos a 30% cada uno) y se retomó desde el mejor
punto ya alcanzado (iteración 8), con evaluaciones a 300 partidas en vez de 200 para
estrechar el margen de error.

**Resultado, con el margen de error ahora sí excluyendo el empate:**

| | Grimmsnarl | Espejo | Kangaskhan | **contra el modelo original** |
|---|---|---|---|---|
| antes de reponderar (it8, n=200) | 64,2% | 51,7% | 19,2% | 55,0% [48,1%-61,9%] empate |
| **con la reponderación (it10, n=300)** | **55,7%** | **45,7%** | 21,0% | **43,8% [38,2%-49,4%] PEOR** |

**Kangaskhan casi no se movió** (19,2% → 21,0%, dentro del ruido) y **los otros dos
rivales empeoraron de forma clara**. La corrección no arregló el problema que buscaba
arreglar y de paso destruyó las ganancias que sí existían. Con 300 partidas el intervalo
de confianza ya no cruza el 50%: es un resultado negativo con certeza, no ambiguo.

**Balance final de las 10 iteraciones de RL, las cuatro mediciones contra el modelo
original:** 52,0% (empate) → 41,2% (peor, con certeza) → 55,0% (empate) → 43,8% (peor, con
certeza). **En ningún punto el resultado fue mejor con confianza estadística.** Es la
misma conclusión de M16/M19 (plateau de RL), ahora confirmada también sobre una base ~250
elo más fuerte y con un mix de oponentes representativo del meta real — descartando que la
base débil o el mix desactualizado fueran la causa de los fracasos anteriores.

Proceso terminado y detenido limpiamente (`data/rl_alakazam/loop_manifest.json`,
checkpoints `data/models/rl_alakazam_iter1..10.json`, no embarcados).

---

## 2. M39: ponderar por importancia de la decisión, no por oponente

### 2.1 La hipótesis

M33 midió que Yushin dispara Powerful Hand el 99,6% de los turnos en que es legal, con
retraso medio de +0,00 turnos — **no hay ninguna decisión de "cuándo atacar" que
aprender**. Toda la brecha está en la velocidad de armado: el turno en que Powerful Hand
se vuelve legal es 4,56 para Yushin y 5,46 para el clon, y su propio corte victoria/derrota
es 4,21 vs 4,95.

El entrenamiento actual trata cada una de las 114.048 decisiones MAIN por igual. La
hipótesis: si las decisiones de armado (las anteriores a que Powerful Hand se vuelva legal
en esa partida) pesan más en la función de pérdida, el modelo podría acertar más justo ahí
— aunque siga desviándose en el resto — y eso podría importar más que la fidelidad
promedio, porque son las que deciden la velocidad de la partida.

### 2.2 El montaje

`scratchpad/train_weighted_main.py`: reutiliza exactamente el entrenador del campeón
(`train_mlp_alakazam.py`, misma partición 3-vías, mismo `_train_mlp`), solo añade un
multiplicador de peso a las decisiones anteriores al turno en que Powerful Hand se vuelve
legal por primera vez en cada partida (detectado igual que `analyze_ph_availability.py`:
`option.type is ATTACK and option.attackId == POWERFUL_HAND`, id **1072**). Validación y
prueba **nunca** llevan peso — si lo llevaran, el experimento se autoevaluaría con la vara
torcida.

**17,8% de las decisiones de entrenamiento cayeron en la fase de armado** (11.166 de
62.559) — una fracción sustancial pero minoritaria, el perfil correcto para que ponderar
tenga sentido.

### 2.3 El resultado

| | control (peso ×1) | con peso ×3 |
|---|---|---|
| lineal (prueba) | 0,5837 | 0,5796 |
| red, semilla 0 (validación) | 0,7497 | 0,7613 |
| red, semilla 1 (validación) | 0,7561 | 0,7529 |
| red, semilla 2 (validación) | 0,7481 | 0,7555 |
| **conjunto de 3, prueba** | **0,7670** | **0,7659** |

**Diferencia: −0,0011. Menor que la varianza entre semillas del mismo entrenamiento
(~0,008).** Ninguna mejora real, ni tampoco un daño real — es ruido.

### 2.4 Lo que esta medición NO responde

El promedio general plano no descarta que el modelo haya mejorado específicamente en las
decisiones de armado a costa de empeorar un poco en el resto (un resultado parcialmente
positivo que un promedio agregado no vería). **No se comprobó** porque los pesos
entrenados no se guardaron a disco — el script solo imprimió métricas — así que no hay
forma de re-analizar sin repetir el entrenamiento completo (~4 h por corrida). Dado que ya
son cinco intentos negativos seguidos y el plazo apremia, no se considera la relación
coste/beneficio favorable para cerrar ese hueco.

### 2.5 Incidente operativo: el equipo se suspendió a mitad de la corrida de control

La corrida de control quedó pausada ~4,5 horas (05:28–09:51) por hibernación de batería
baja mientras el usuario dormía; el proceso sobrevivió la suspensión y continuó solo al
reactivarse la máquina, pero infló el tiempo total de ~2 h estimadas a >12 h reales.
Lección para cualquier corrida larga de esta serie: **verificar que el equipo esté
conectado a la corriente antes de lanzar un trabajo de varias horas** — `powercfg
standby-timeout` solo evita la suspensión por inactividad, no la de batería crítica ni la
de tapa cerrada.

---

## 3. El patrón que ya es innegable

Cinco intentos, cinco resultados nulos o negativos, atacando la fidelidad de imitación
desde ángulos completamente distintos:

| intento | palanca | resultado |
|---|---|---|
| M31 | anchura de la red (h=48→96) | +0,006 (dentro del ruido) |
| M36 | +42% más datos de Yushin | negativo (0,767→0,762 con la MLP) |
| M37 | arquitectura (transformer de conjunto) | +0,017 (bajo la barra de +0,030) |
| M38 | aprendizaje por refuerzo (self-play) | nunca mejor con confianza; dos veces peor |
| M39 | ponderar por importancia de decisión | −0,0011 (ruido) |

Ninguna de estas palancas ataca la causa raíz que M38 midió directamente: el clon se
desvía de la línea de Yushin cada ~4,7 decisiones y a partir de ahí juega en estados sin
datos de entrenamiento. La única familia que ataca eso de raíz (DAgger, corrección
interactiva) requiere un oráculo consultable que no tenemos — solo repeticiones grabadas,
no un Yushin en vivo.

**Recomendación:** cerrar la búsqueda de mejoras de fidelidad/política. Los dos agentes ya
desplegados (`imitation-setxf2-fixed` y `imitation-mlp-fetch`, ambos convergiendo cerca de
885-890 tras un día completo de partidas) son el resultado consolidado de esta línea de
trabajo. El tiempo restante antes del cierre rinde más asegurando su envío final que
intentando una sexta palanca sobre el mismo cuello de botella ya diagnosticado tres veces
de forma independiente.
