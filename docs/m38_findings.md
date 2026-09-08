# M38 — por qué la imitación no puede cerrar el hueco, y qué queda (2026-08-04)

**Resumen: dos análisis independientes, por caminos distintos, dan la misma respuesta. (1) El techo
de Bayes de nuestras features es 0,968 y estamos en 0,786 — las features NO son el cuello de
botella. (2) Convertidos a rating implícito, NO existe ningún agujero de matchup: nuestro perfil
relativo tiene la misma forma que el de Yushin, somos uniformemente ~250 puntos peores. La causa
común es el desplazamiento de distribución: nos desviamos de su línea cada ~4,7 decisiones y a
partir de ahí jugamos donde no hay datos. Eso explica de una sola vez por qué fracasaron más datos
(M36), más anchura (M31) y mejor arquitectura (M37): las tres optimizan la variable equivocada.
También corrige un error propio: el motor es ESTOCÁSTICO y mi verificación de `head_to_head_par`
como "bit a bit idéntica" era inválida.**

---

## 1. El techo no está en las features

Medido con `scratchpad/ceiling_decomposition.py` sobre las 114.048 decisiones MAIN del corpus.

La idea: si dos decisiones producen el **mismo vector de features X**, nuestro modelo tiene que dar
el mismo score en ambas. Si Yushin jugó cosas distintas, esa discrepancia es **irreducible para
nosotros** — ninguna arquitectura la arregla, porque la información ya se perdió al featurizar.

```
vectores X que se repiten: 552  (cubren 12.206 decisiones = 10,7% del total)
de esas, con accion INCONSISTENTE: 388 = 3,2%
-> techo de Bayes de ESTA representacion: ~0,968
```

**Estamos en 0,786. Hay ~18 puntos que son aprendibles en principio.**

Comprobación de sesgo, porque el subconjunto podía ser trivial: las decisiones con X repetido
tienen **9,18 opciones de media** (mediana 8) contra 11,16 del conjunto completo, y solo el **2,9%**
son forzadas de una sola opción. No es un subconjunto vacío de contenido.

### 1.1 El test de estado oculto quedó SIN RESOLVER, no refutado

También se buscaron observaciones **crudas** idénticas con acciones distintas — la firma de un
agente con estado interno o búsqueda. **Cero repeticiones exactas en 114.048 decisiones**, así que
el experimento no pudo correr. La hipótesis de los 401 ms/decisión de Yushin sigue abierta.

## 2. El error se compone, y la métrica no lo ve

Con 49 decisiones MAIN por partida:

| fidelidad por decisión | P(reproducir una partida entera) |
|---|---|
| 0,768 (MLP embarcada) | 2,5 × 10⁻⁶ |
| 0,786 (transformer) | 7,6 × 10⁻⁶ |
| 0,95 (hipotético) | 8,1 × 10⁻² |

Nos desviamos de la línea de Yushin cada **~4,7 decisiones**, unas 10 veces por partida. Tras la
primera desviación estamos en estados que él nunca visitó, donde el entrenamiento no dice nada.

**El clon no es "Yushin con un 21% de ruido". Es un agente distinto que coincide con él el 79% de
las veces en los estados de él.** Toda nuestra medición asumía lo primero.

Lo que arregla esto es imitación interactiva (DAgger): preguntar al maestro qué haría en los
estados a los que llega el ALUMNO. **No podemos** — solo tenemos repeticiones, no un oráculo.

## 3. No hay agujero de matchup (y esto corrige el plan que casi ejecuto)

El WR por arquetipo engaña porque el emparejamiento da rivales de tu propio nivel: Yushin gana
58,6% contra un campo de **1096** y nosotros 55,6% contra uno de **857**. Comparar esos
porcentajes directamente no mide nada.

Convertido a **rating implícito relativo a la base de cada uno**:

| rival | CLON (base 888) | YUSHIN (base 1139) |
|---|---|---|
| Marnie's Grimmsnarl | **−42** (n=41) | **−58** (n=370) |
| Alakazam (espejo) | **−3** (n=33) | **−44** (n=45) |
| Mega Kangaskhan | +73 (n=8) | +232 (n=56) |
| Archaludon | +232 (n=18) | — (0,3% de su campo) |
| TR Tarountula | — (0,8%) | **−196** (n=29) |

- **Grimmsnarl es debilidad del MAZO, no del clon.** Yushin está −58 contra él; nosotros −42.
  Relativamente lo llevamos *mejor*. Atacar ese matchup era perseguir algo que el maestro tampoco
  consigue.
- **El "hueco de 19,5 puntos en el espejo" era un artefacto** de la fuerza del rival. Clon −3,
  Yushin −44.
- Nuestro campo es incluso **más fácil** en composición (13,5% Archaludon que ganamos 83%, contra
  0,3% del suyo). Aun así estamos 250 puntos por debajo.

**El déficit es uniforme**, exactamente lo que predice §2: el desplazamiento de distribución degrada
en todas partes, no en un matchup.

### 3.1 Lo que hace falta para 1000

El modelo elo encaja: en 899 contra un campo de 857, esperado 56,0%, observado 55,6%. Para estar en
1000 contra ese mismo campo hace falta un **69,5% de WR**. Estamos en ~62%. Nada en 37 milestones ha
movido el WR más de ~4 puntos de golpe.

## 4. Corrección: el motor es ESTOCÁSTICO

`scratchpad/_check_determinism.py`: dos corridas idénticas de `collect` en el **mismo proceso** dan
resultados distintos (1033 vs 1115 filas; WR Grimmsnarl 0,2 vs 0,6). El motor baraja desde entropía.

**Esto invalida una verificación que di por buena en M37.** Afirmé que el gauntlet paralelo era "bit
a bit idéntico" al serial, pero lo medí sobre una porción donde la línea base ganaba **1.000 en los
8 mazos**: dos resultados saturados coinciden trivialmente. La afirmación ya está corregida en
`head_to_head_par.py`.

Consecuencias:
- La paralelización sigue siendo válida, pero el criterio correcto es **distribucional**, no bit a
  bit: mismas cuotas por oponente y WR dentro del ruido de muestreo.
- Cualquier comparación pareada en el gauntlet necesita más partidas de las que suponía — la misma
  razón por la que M25 vio señales a n=6 evaporarse a n=12.
- El hallazgo de saturación de M37 **no** queda afectado: 0,965 macro es un nivel medido, no una
  comparación entre dos corridas.

## 5. Lo que se montó para el intento de RL

Único camino que ataca §2, porque entrena sobre los estados que la política **realmente visita**.
M19 cerró esta línea con cinco confirmaciones, pero **ninguna tuvo estas cuatro cosas**:

1. **Base ~923** (clon de Alakazam) en vez de ~686 (`bc_650_v2`).
2. **Mix de oponentes del meta real** — el de M16 no contenía **ningún** Grimmsnarl, que es el 30,8%
   del campo que enfrentamos. `bc_luca_full` (descartado como candidato en M35 por techo de datos)
   sirve aquí como *oponente*: un Grimmsnarl pilotado de verdad en vez de por greedy.
3. **Evaluación que no satura**: WR contra el **init congelado** (0,500 por construcción, sin techo)
   en vez del gauntlet. M16 leyó "+0,027 contra una barra de +0,05" en un instrumento con 3,5% de
   margen.
4. **Free-roll de ladder**, que la propia conclusión de M16 recomendaba y nunca se hizo.

Sin cambiar: λ=0,10 fijo (M19 aisló el schedule y demostró que bajarlo desestabiliza), batch,
epochs, tau y los seis kill criteria. Experimento de variable aislada.

**Barra pre-registrada:** ≥55% contra el init congelado con n≥300 y el IC del 95% excluyendo 0,500.

### 5.1 Limitación conocida del montaje

El oponente Grimmsnarl es el clon de Luca, entrenado con solo 539 episodios. En pruebas le ganamos
50-79% mientras en el ladder real perdemos ese matchup a 43,9%. Es mucho mejor que greedy, pero
**sigue siendo más débil que un Grimmsnarl real**; si el RL sobreajusta a batirlo, puede no
transferir. Por eso Archaludon se reserva como arquetipo **held-out**.

## 6. Infraestructura reutilizable

- `rl_selfplay.py` con **setups** (`tr650` por defecto, reproducible; `alakazam` nuevo) y rutas de
  salida separadas, para no pisar los checkpoints de M16/M19.
- `collect` paralelo por **sharding de partidas** (no por oponente: con 3 oponentes de pesos
  desiguales el reparto por oponente medía 1,86×; por partida da **2,68×**).
- `train_value.py` parametrizable → `v_alakazam.json`, reajustado sobre el corpus de Yushin porque
  M18 ya documentó que `v_650` está mal calibrado fuera de su distribución.
- `run_rl_detached.ps1`: `nohup` bajo Git-Bash **no** sobrevive al teardown de sesión en este
  entorno — se perdieron dos trabajos largos así, en silencio (log vacío, sin traza, sin código de
  salida). `Start-Process` sí desprende el proceso del árbol del shell.

## 6.1 El colapso de la primera iteración: los hiperparámetros de M16 no portan

La primera corrida murió en `[it1]` con `bc_acc` cayendo de **0,733 a 0,168** — el modelo
destruido en una sola actualización. Pero los pesos apenas se movieron: `|ΔW1|/|W1|` = 5%,
`|Δw2|/|w2|` = 2%. Un cambio del 5% no destruye un modelo... salvo que sus decisiones estén
muy reñidas.

Medido:

| | tr650 (M16) | alakazam | ratio |
|---|---|---|---|
| margen medio 1º-2º entre opciones | 4,66 | 1,98 | **2,4× menor** |
| desviación de scores por decisión | 6,99 | 3,42 | **2,0× menor** |
| exploración con `tau=1.0` | ~16% | **32,9%** | 2× |

**La escala de scores del modelo Alakazam es la mitad.** Con `tau` fijo eso duplica la
exploración, y con el mismo paso de gradiente voltea el doble de argmax. Los
hiperparámetros de M16 estaban afinados contra un modelo de escala distinta; **copiar los
números en crudo en vez de escalar el paso efectivo es lo que rompió la corrida.**

Barrido sobre una trayectoria real (`scratchpad/_sweep_update.py`, sin volver a jugar):

| lr | epochs | coincidencia con la base | bc_acc | Δ |
|---|---|---|---|---|
| **1e-3** | **25** | 0,172 | 0,168 | **−0,564** ← el valor de M16 |
| 3e-4 | 10 | 0,782 | 0,662 | −0,071 |
| 1e-4 | 10 | 0,920 | 0,722 | −0,011 |
| **3e-5** | **10** | **0,965** | **0,732** | **−0,001** |

`3e-5`/10 mueve el 3,5% de las decisiones conservando la precisión intacta. Las iteraciones
sanas de M16 movían ~1% y hacía 30; ocho vueltas al 3,5% dan un desplazamiento total
comparable.

**Regla general:** al portar un experimento de RL a otra política base, hay que escalar los
hiperparámetros a la **escala de scores** del modelo nuevo. El paso efectivo, no el número.

## 7. Cuatro bugs propios cazados en el montaje

1. **Los workers volvían al setup por defecto.** Windows crea los procesos por *spawn*, que
   reimporta el módulo y reinicia los globales: buscaban oponentes de `tr650`, no encontraban
   ninguno y devolvían **cero filas**, mientras el padre reportaba un "speedup de 24×". Habría
   entrenado con un dataset vacío. Ahora el setup viaja explícito y hay un guardia que convierte un
   worker perdido en error duro.
2. **Los checkpoints iban a pisar los de M16/M19** (`data/models/rl_ckpt_iter{t}.json`). Aislados
   por setup.
3. **El umbral de muerte K2 estaba fijado en 0,93 absoluto**, heredado del modelo TR-650. El
   modelo base de Alakazam marca **0,733** en esa misma sonda, así que el kill habría disparado en
   la iteración 1 de una corrida perfectamente sana — y lo hizo. Ahora el suelo es **relativo**
   (base − 0,03).
4. **`--workers` no llegaba a `iterate`** (sí a `collect`), y **`lr` estaba hardcodeado a 1e-3**
   dentro del bucle, precisamente el valor que destruye este modelo. Ambos expuestos.

Los cuatro comparten una forma: **una constante calibrada para otro contexto, heredada sin
revisar.** Es el mismo patrón que el mazo equivocado de M37 y que la línea base del barrido de
Colab. Cuando se reapunta maquinaria existente a un problema nuevo, las constantes son el sitio
donde mirar primero.
