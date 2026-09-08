# M47 — el RL de auto-juego, corrido con sus cuatro entradas corregidas (2026-08-13)

**Estado: EN CURSO.** Las secciones 1-4 están medidas y cerradas. La sección 5 (la corrida real)
y la 6 (la puerta decisiva) se completan cuando terminen.

Continúa [m46_findings.md](m46_findings.md) (enriquecimiento de MAIN y k=7, ambos cerrados
negativos).

---

## 0. Por qué se reabre una línea que M39 declaró cerrada

M39 escribió "no reabrir el RL de auto-juego sin una palanca fundamentalmente distinta". Hay
cuatro, y ninguna es una corazonada: las cuatro son defectos que estaban **medidos y escritos
en este mismo repo**, y que nadie había conectado con el resultado.

1. **Falta el recorte de razón de probabilidad de PPO.** `rl_update` reutiliza el mismo lote 10
   épocas sin corregir que la política se movió entre ellas. La deriva medida en M38
   (`probe_agree` 0,966 → 0,8835 en 10 iteraciones, sin ganancia de victorias) es exactamente
   la inestabilidad que el recorte existe para evitar. Prueba retroactiva en GPU sobre 3
   transiciones ya recolectadas: 3 de 3 a favor del recorte.
2. **`tau` nunca se reescaló.** M38 §6.1 midió que la escala de scores del modelo Alakazam es
   la mitad que la de TR-650 y que con `tau=1.0` la exploración se duplica (32,9% contra el
   ~16% de M16). Corrigió `lr` por esa razón y **dejó `tau` en 1,0**. Los 10 manifiestos lo
   confirman: `pct_nonargmax` ≈ 0,33 en todas las iteraciones.
3. **Los pesos de rival del archivo son los de M39**, 0,30/0,30/0,40 — la configuración que el
   propio M39 midió como **confiadamente peor** (0,438 [0,382-0,494] contra el init congelado a
   n=300) y que quedó sin revertir.
4. **La evaluación de M38 no medía nada.** Sus cuatro lecturas fueron a **n=200**, y M41
   demostró después que a n=200 este motor lee el espejo consigo mismo como 0,435. Sólo el
   reweight de M39 (n=300) quedó cerrado con confianza. El RL con la configuración de M38
   **nunca fue evaluado de forma concluyente.**

A eso se suma un quinto defecto, encontrado hoy y el mayor de todos (§1).

## 1. El crítico estaba casi ciego — AUC 0,6526

`A = R − V(s)` multiplica **cada gradiente** del bucle, así que la calibración del crítico acota
lo que el RL entero puede hacer. `data/models/v_alakazam.json` se ajustó sobre el corpus de
repeticiones de Yushin, antes de que existiera una sola partida de auto-juego de Alakazam. En
disco había **10 iteraciones de auto-juego ya jugadas** (`data/rl_alakazam/iter1..10_traj.jsonl.gz`)
que nunca se usaron para reajustarlo.

Reajuste con `refresh_critic` (entrenar iter1-8, validar iter9-10, **282.242 / 70.980**
decisiones), con la puerta que ya traía escrita — sólo adopta si el AUC held-out supera al
incumbente sobre las mismas filas:

| crítico | features | AUC held-out |
|---|---|---|
| `v_alakazam.json` (incumbente, M38/M39) | 7 | **0,6526** |
| candidato base-7 | 7 | 0,7846 (+0,1320) |
| **candidato vf2 (ADOPTADO)** | 10 | **0,7902 (+0,1377)** |

**Las ventajas de las tres corridas de RL anteriores se calcularon con una función de valor de
AUC 0,65.** No es un ajuste fino: es la diferencia entre una señal de crédito informativa y una
casi aleatoria.

**Salvedad honesta:** el candidato se ajusta sobre estados de auto-juego y se valida sobre
estados de auto-juego, así que el +0,138 es "mejor crítico **para esta distribución**", que es
justo la que el RL visita — no un crítico universalmente mejor. Y las 70.980 filas de validación
salen de ~800 partidas, así que el tamaño de muestra efectivo es el de partidas, no el de filas;
la brecha es demasiado grande para ser ruido, pero su IC no es el que sugieren 71k filas.

`refresh_critic` quedó parametrizado (`--train/--val/--out/--features`) y acepta varios conjuntos
de features en **una sola** pasada de featurización, que es la parte cara (301 s).

## 2. `tau` — la curva que nadie había medido

`collect` de 100 partidas por punto, con el campeón embarcado como base:

| tau | % de decisiones MAIN fuera del argmax |
|---|---|
| 1,0 | **0,3334** ← el valor de M38 (sus manifiestos: 0,3285-0,3334) |
| 0,6 | 0,2563 |
| 0,5 | 0,2167 |
| 0,4 | 0,2006 |
| **0,3** | **0,1690** ← el perfil de exploración de M16 (~16%) |
| 0,25 | 0,1505 |

El punto a `tau=1,0` **reproduce exactamente** el valor histórico, lo que valida el instrumento
antes de usarlo. Se elige `tau=0,3`.

**Confirmación independiente de que la corrección importa:** en la iteración 1 de la corrida real
el espejo de recolección (política muestreadora contra el campeón determinista) da **0,500**. A
`tau=1,0`, M38 leía **0,457** en ese mismo espejo. Es decir: con la temperatura vieja, la política
que generaba los datos de entrenamiento jugaba ~4 puntos por debajo de la que se despliega. Ahora
juega a la misma fuerza.

## 3. El barrido del update — qué hace el recorte, medido

Lote fresco de 300 partidas a `tau=0,3` recolectado por el propio campeón, ventajas calculadas con
el crítico nuevo, un update completo (10 épocas) por configuración, sobre 4.000 decisiones MAIN
held-out (`scratchpad/m47_sweep_update.py`).

| clip | adv-norm | lr | movido | bc_acc | Δbc |
|---|---|---|---|---|---|
| — | — | **0** (control nulo) | 0,0000 | 0,7325 | +0,0000 |
| — | — | 1e-5 | 0,0120 | 0,7315 | −0,0010 |
| 0,2 | — | 1e-5 | 0,0122 | 0,7305 | −0,0020 |
| — | — | 3e-5 | 0,0175 | 0,7318 | −0,0008 |
| 0,2 | — | 3e-5 | 0,0138 | 0,7312 | −0,0013 |
| — | — | 1e-4 | 0,0282 | 0,7278 | −0,0048 |
| **0,2** | — | **1e-4** | **0,0185** | **0,7285** | −0,0040 |
| — | — | 3e-4 | **0,0483** | 0,7255 | −0,0070 |
| 0,2 | — | 3e-4 | **0,0198** | 0,7292 | −0,0033 |
| 0,2 | ✓ | 3e-5 | 0,0170 | 0,7305 | −0,0020 |
| 0,2 | ✓ | 1e-4 | 0,0170 | 0,7298 | −0,0028 |

**(1) El recorte funciona y se ve en el desplazamiento, no en la precisión.** Sin recorte, subir
`lr` de 1e-4 a 3e-4 mueve de 2,82% a **4,83%**; con recorte se queda en 1,85% → **1,98%**. Eso es
la región de confianza acotando cuánto puede empujar un solo lote, sobre datos reales.

**(2) A un solo update, la ventaja del recorte en `bc_acc` NO es detectable.** Todas las
diferencias son ≤0,002 contra un error estándar de ~0,007 en 4.000 decisiones. Esto es lo
esperado y hay que decirlo: el efecto del recorte es **acumulativo entre iteraciones** — es
justo donde M46 lo midió (transiciones encadenadas iter4→5, 6→7, 9→10) y donde no puede verse
desde la base en un paso. **Este barrido elige `lr`; no es evidencia de que el recorte ayude.**

**(3) El control nulo era imprescindible.** Sin la fila `lr=0` las dos columnas no distinguen
"el recorte aprende mejor" de "el recorte aprende menos": un update que no hace nada saca
`movido=0` y `bc_acc` máximo. La prueba retroactiva de M46 no lo tenía.

**(4) `--adv-norm` es inerte en el tamaño del paso** — Adam es invariante a la escala del
gradiente (`m/√v` cancela cualquier factor global), y las filas lo confirman (0,0170 movido a
3e-5 **y** a 1e-4). Se activa igual, por una razón distinta y medida: el crítico nuevo deja un
sesgo de **+0,093** en la ventaja media (se ajustó sobre partidas de `tau=1,0`, donde la política
ganaba 40%; a `tau=0,3` gana más y V queda bajo). Un desplazamiento constante del baseline no
puede sesgar el gradiente de política, sólo quitar varianza.

**(5) El crítico nuevo estrecha la ventaja un 8,8%** (std 0,4779 → 0,4360) — lo que se espera de
un V que explica más del resultado.

Configuración elegida: **`clip=0,2`, `lr=1e-4`, `epochs=10`, `tau=0,3`, `lambda=0,1`, `adv-norm`**.
`lr=1e-4` se elige porque es el punto donde el brazo **sin recorte** seguiría siendo seguro
(2,82% movido), así que la corrida no depende de que la máscara del recorte sea perfecta.

## 4. Lo que se construyó, y su red de seguridad

Todo aditivo y apagado por defecto — el mismo patrón de M27 (`recover_rule`), M34 (perfiles por
contexto) y M42 (`count_heads`):

- `rl_update(..., clip=, adv_norm=, entropy=)`. Con las tres ausentes el camino de M16/M19/M38
  queda **bit a bit idéntico**, y eso lo fija un test que compara contra una copia literal del
  cuerpo anterior (`tests/unit/test_rl_update_regression.py`, 7 tests).
- **Un test encontró un error de premisa mío, no del código:** con banda tan ancha que nunca
  ata, el recorte **no** se reduce a REINFORCE plano — se reduce a REINFORCE con peso de
  importancia, porque el factor `ratio` sigue en el gradiente. La diferencia aparece desde la
  segunda época y **es precisamente la corrección de lote rancio que M47 mide**. El test que
  exigía igualdad ahí habría estado afirmando el bug de vuelta; se reescribió para fijar la
  propiedad correcta (igualdad exacta en la **primera** época, donde `ratio ≡ 1`).
- SETUP `alakazam_final`: base = campeón embarcado, mezcla de rivales revertida a M38, y un
  arquetipo held-out **con clon real** (Mega Lucario, 11,5% del campo) — `alakazam` nombraba
  Archaludon y luego tenía que resolverlo "en tiempo de evaluación" porque no existe clon, es
  decir, no tenía chequeo de sobreajuste.
- `scratchpad/m47_arena.py`: la puerta de M43 generalizada por checkpoint, con la fila held-out
  añadida.
- `scratchpad/run_m47_detached.ps1`: `Start-Process`, porque `nohup` bajo Git-Bash no sobrevive
  al teardown de sesión (le costó dos trabajos largos a M38, en silencio).

## 5. Brazo A (`lambda=0,10`) — el recorte funciona, y por eso deja de moverse

Base = campeón embarcado, 800 partidas por iteración, `clip=0,2 / lr=1e-4 / tau=0,3 / adv-norm`.

| iter | `probe_agree` | M38 en la misma iter | delta | `bc_acc` | WR de recolección (grim/espejo/kanga) |
|---|---|---|---|---|---|
| 1 | 0,984 | 0,966 | **+0,018** | 0,729 | 0,716 / 0,500 / 0,315 |
| 2 | 0,980 | — | — | 0,729 | 0,671 / 0,493 / 0,355 |
| 3 | 0,982 | 0,937 | **+0,045** | 0,731 | 0,713 / 0,444 / 0,300 |
| 4 | 0,979 | 0,925 | **+0,054** | 0,732 | 0,722 / 0,491 / 0,315 |
| 5 | 0,974 | 0,911 | **+0,063** | 0,731 | 0,713 / 0,489 / 0,320 |
| 6 | 0,973 | 0,900 | **+0,073** | 0,730 | 0,650 / 0,514 / 0,265 |

**(1) La deriva que motivó M47 está eliminada.** M38 iba en 0,900 a la iteración 6 y siguió cayendo
a 0,8835; aquí la deriva es de un tercio y `bc_acc` no se mueve del valor base (0,7325). La
hipótesis del recorte se confirma sobre partidas nuevas, no sólo retroactivamente.

**(2) Y ese es justo el problema.** `probe_agree` se queda **plano** en ~0,98 desde la iteración 1:
la política no viaja, oscila alrededor de un punto a ~2,7% del campeón. El mecanismo es
aritmético, no una conjetura: el término de ancla aporta `lambda·(P − P_bc)` al gradiente, que
vale **exactamente cero en el primer update** (P = P_bc) — por eso el barrido §3 midió 1,85% de
puro gradiente de política — y después tira de vuelta con fuerza proporcional al desplazamiento.
El sistema se estabiliza donde ambas fuerzas se igualan. **Lo que ata no es el recorte: es
`lambda=0,10`.**

**(3) La puerta, sobre el checkpoint de la iteración 6** (`m47_arena.py`, n=600):

| | observado | referencia | delta | IC 95% |
|---|---|---|---|---|
| candidato vs campeón (espejo) | **0,521** | 0,500 | +0,021 | [0,481, 0,561] |
| campeón vs campeón (control) | 0,485 | 0,500 | −0,015 | [0,445, 0,525] |

Ajustado por el control, **+0,036** — dirección correcta, magnitud mayor que la del k=7 de M46
(+0,007), pero **el IC cruza 0,500 y por la regla pre-registrada NO pasa**. Es lo que cabe esperar
de una política que se movió 2,7%: no hay nada que medir todavía. 0 intervenciones y
`bc_failures=0` en 1.200 partidas.

## 6. Brazo B (`lambda=0,03`) — el experimento que M19 no pudo hacer

M19 demostró que bajar `lambda` desestabiliza el bucle (`bc_acc` 0,937 → 0,904, muerte por K2) —
pero lo demostró **sin recorte**. Con la región de confianza puesta, bajar el ancla a 0,03 debería
mover el equilibrio más lejos sin esa deriva. Variable única respecto al brazo A.

**Resultado: el recorte hace seguro lo que mató a M19 — y da igual.**

| iter | brazo A (λ=0,10) | brazo B (λ=0,03) |
|---|---|---|
| 1 | 0,984 | 0,982 |
| 2 | 0,980 | 0,979 |
| 3 | 0,982 | 0,979 |
| 4 | 0,979 | 0,978 |
| 5 | 0,974 | 0,975 |
| 6 | **0,973** | **0,974** |

`bc_acc` se mantuvo en 0,729-0,733 en los dos brazos, sin rastro de la muerte por K2 de M19. Pero
las dos curvas de conducta son **indistinguibles en las seis iteraciones**.

## 7. El diagnóstico que reordenó el milestone: no es deriva ni varianza, es OSCILACIÓN

`scratchpad/m47_step_coherence.py` — 2 minutos, cero partidas nuevas, sobre checkpoints ya en
disco. Compara la **longitud del camino** (`Σ|P_t − P_{t−1}|`) con el **desplazamiento neto**
(`|P_T − P_0|`). Su cociente es la eficiencia del recorrido: ~1,0 = línea recta, ~1/√T = pasos
independientes.

| brazo | camino | desplaz. neto | eficiencia | 1/√T | cos medio consecutivo |
|---|---|---|---|---|---|
| A (λ=0,10) | 0,860 | 0,184 | **0,214** | 0,408 | **−0,283** |
| B (λ=0,03) | 0,973 | 0,281 | 0,288 | 0,408 | −0,168 |

**(1) No es un paseo aleatorio: es PEOR que uno.** Las eficiencias están por debajo de 1/√T y los
cosenos consecutivos son **negativos en todos los pares**. Ruido puro daría coseno ≈ 0. Cada
actualización deshace parcialmente la anterior.

**(2) El mecanismo es Adam.** Las longitudes de paso del brazo A son 0,139 / 0,149 / 0,146 / 0,144
/ 0,138 / 0,143 — prácticamente constantes. Adam normaliza el gradiente por su propia RMS, así que
da un paso del **mismo tamaño haya señal o no**. Cuando la dirección se invierte, cinco sextos del
camino se cancelan.

**(3) `lambda` sí participaba, pero no como yo lo había diagnosticado.** Bajarlo de 0,10 a 0,03
subió el desplazamiento neto un **53%** (0,184 → 0,281) y debilitó la anti-correlación (−0,283 →
−0,168): el ancla es una fuerza restauradora y genera parte de la oscilación. **Pero ese 53% más de
movimiento en el espacio de pesos produjo cero cambio de conducta** (`probe_agree` 0,973 vs 0,974).
El movimiento extra va en direcciones que no cambian ninguna decisión.

**(4) Error de método propio, para el registro:** el brazo B costó ~95 minutos de cómputo para
falsar una hipótesis que esta medición de 2 minutos podía haber falsado **antes** de lanzarlo. La
regla del proyecto — señal barata antes de comprometer tiempo grande — estaba escrita y se rompió.
Se recupera en parte porque sus iterados alimentan §8.

## 8. Promediado de iterados — la respuesta al problema que sí se midió

Si los pasos se cancelan, el **centro** de la oscilación es la componente coherente; promediar los
iterados es el remedio clásico. Cuesta un minuto sobre archivos ya en disco y no puede fallar por
un motivo nuevo: o extrae una señal que ya está ahí, o demuestra que no la había.

`scratchpad/m47_average_iterates.py` promedia MAIN miembro a miembro sobre el payload del campeón,
así que los otros 11 contextos, `profile_overrides` y `count_heads` pasan intactos (verificado con
`m47_verify_payload.py`). Promediar pesos de redes entrenadas por separado no significaría nada;
promediar a lo largo de **una** trayectoria de optimización sí — estos iterados están todos a menos
de 0,29 del mismo punto de partida en un espacio de 95.043 dimensiones.

Tres candidatos: promedio del brazo A (6 iterados), del brazo B (6), y de los 12 juntos. Los tres
verificados con `m47_verify_payload.py`: MAIN cambia, los otros 11 contextos byte-idénticos.

## 9. La puerta decisiva — la familia RL entera falla

Espejo n=600 contra el campeón embarcado. Control del día: **0,485** [0,445-0,525], dentro de la
banda válida, así que el instrumento está calibrado.

| candidato | espejo | IC 95% | veredicto |
|---|---|---|---|
| brazo A, iteración 6 | 0,521 | [0,481, 0,561] | no pasa |
| promedio brazo A | 0,512 | [0,472, 0,551] | no pasa |
| promedio brazo B | 0,520 | [0,480, 0,560] | no pasa |
| promedio de los 12 | **0,490** | [0,450, 0,530] | no pasa |

**Agrupando las cuatro medidas (n=2.400): 0,511, IC [0,491, 0,531] — sigue tocando 0,500.** No es
que falte potencia estadística: es que no hay efecto que detectar. El promediado, que era la
respuesta correcta al problema medido, tampoco extrajo nada — y el promedio de 12, el que más
oscilación cancela, salió **por debajo** de 0,500.

**Novena hipótesis falsificada. `imitation-final` se mantiene intacto en los dos slots.**

Corrección a mi propio reporte durante la sesión: llegué a citar "+0,036 ajustado por control". Está
mal y lo retiro. El espejo vale 0,500 **por construcción**; el control sólo verifica que el
instrumento no esté sesgado, no es una corrección que se sume al candidato. Sumarla es contar el
ruido dos veces.

## 10. Lo que sí quedó, y una palanca nueva que no da la talla

### 10.1 El audit de la dimensión de CONTEO — cerrado, negativo, en 10 minutos

`m47_count_audit.py` mide, sobre las 182.862 decisiones de Yushin y sin entrenar nada, cuántas veces
el conteo del maestro difiere del `min(maxCount, n)` que `_top_k` toma siempre — la impossibilidad
estructural que hizo que SETUP_BENCH valiera +0,5098 en M42.

| contexto | /partida | discrepancia |
|---|---|---|
| SETUP_BENCH_POKEMON | 0,43 | **44,3%** (ya arreglado en M42) |
| TO_HAND | 9,05 | 0,7% |
| TO_DECK | 0,58 | 0,3% |
| TO_BENCH | 2,91 | 0,0% |
| los otros 14 contextos | — | **0,0%** |

**SETUP_BENCH era genuinamente único.** La dimensión de conteo queda cerrada para siempre, medida en
diez minutos en vez de construida a ciegas sobre una corazonada. (`more = 0` en todas las filas:
tomar más de `min(maxCount, n)` es ilegal, así que un valor no nulo habría delatado un parseo roto.)

### 10.2 `DRAW_COUNT` — un defecto del 100%, real y demasiado raro

El mismo audit destapó cuatro contextos que **ningún milestone había tocado** (M42 cubrió 2, M43 los
7 siguientes). Midiendo qué elige el maestro en ellos, contra el `_safe_default` = opción 0 que usa
el agente embarcado:

| contexto | modelado | n | /partida | el maestro elige la opción 0 |
|---|---|---|---|---|
| DISCARD_ENERGY | no | 2768 | 1,19 | 92,8% |
| EVOLVE | no | 2424 | 1,04 | 98,3% |
| IS_FIRST | no | 1223 | 0,52 | **100,0%** (acertamos por accidente) |
| **DRAW_COUNT** | **no** | **488** | **0,21** | **0,0%** |
| DAMAGE | no | 23 | 0,01 | 43,5% |

**En `DRAW_COUNT` el maestro elige el ÚLTIMO índice 488 veces de 488, y nosotros el primero 488 de
488.** Divergencia sistemática del 100%, nunca examinada.

No es arreglable con un modelo: las opciones crudas son `{"number": 0, "type": 0}` /
`{"number": 1, "type": 0}` — sin identidad de carta, así que `featurize_option` devuelve el mismo
vector para todas y el softmax es simétrico por construcción. Es el cuello de botella de M34 en su
límite. Se arregla con una **regla**, no con una cabeza.

Implementado como `index_rules` en el payload (`policy.py`), apagado por defecto — el patrón de M27
(`recover_rule`) y M42 (`count_heads`), así que cualquier agente sin la clave queda byte-idéntico.
7 tests nuevos, **245 pasan**.

**Y aun así no da la talla, con la aritmética por delante.** Comprobación obligatoria de M34 ("¿la
intervención cambia la conducta?"): la regla dispara **0,10 veces por partida** en simulación (el
maestro llega a 0,21). M43 corrigió 1,26 decisiones/partida y valió +0,096 en la arena; escalando,
esto predice **~+0,008** contra un error estándar de 0,020 a n=600. **Por debajo de lo que cualquier
instrumento del proyecto puede medir.** Queda construido, probado y listo
(`data/models/bc_alakazam_draw.json`), pero no se sube sin poder medirlo: es exactamente la decisión
que M27 tomó al revés (la regla de recuperación parecía obviamente correcta y el ladder la midió
como perjudicial, 843 contra 899).

## 11. Reutilizable

- **`m47_step_coherence.py`** — el instrumento más valioso del milestone. Distingue "viaja despacio"
  de "oscila" de "es ruido" en 2 minutos, sobre checkpoints ya en disco. Cualquier futura corrida de
  RL debería pasar por aquí **antes** de gastar horas.
- **`m47_count_audit.py`** — cierra o abre la dimensión de conteo de cualquier contexto sin entrenar.
- `refresh_critic --train/--val/--features` (varios conjuntos en una pasada) y
  `v_alakazam_v2.json` (AUC 0,79): entrada nueva para **cualquier** método basado en valor, incluida
  la búsqueda que M18 cerró precisamente por un valor de hoja mal calibrado.
- `rl_update(clip=, adv_norm=, entropy=)` + `--tag` para arms, con el test de regresión que fija el
  camino de M16/M19/M38 bit a bit.
- `m47_arena.py` (la puerta de M43 por checkpoint, con fila held-out) y `m47_verify_payload.py`.
- `index_rules` en `policy.py` — mecanismo general para contextos cuyas opciones no llevan identidad
  de carta y por tanto son inexpresables por cualquier modelo de ranking.

## 12. Recomendación

**No subir nada.** Los dos slots (`55438687` implícito 765, `55438655` implícito 792) siguen
corriendo hasta el domingo; M43 midió que dejarlos correr vale ~+37 y que cada subida corta el flujo
de episodios de la anterior. Ninguno de los cinco candidatos medidos hoy pasa la puerta
pre-registrada, y subir tras fallarla sería mover la portería.
