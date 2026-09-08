Continúo un proyecto de la competencia Kaggle "The Pokémon Company - PTCG AI Battle Challenge
Simulation". Cierre de envíos: domingo 16-ago-2026. Repo: c:\Users\ASUS\Desktop\S\Pokemon TCG
(git, rama main). Lee primero docs/handoff.md (el banner tiene el resumen de M31 en adelante,
en orden) y los docs/mXX_findings.md que cita.

## Estado real (verificar con tools/kaggle_api.py o scratchpad/rating_trajectory.py, nunca
confiar en el puntaje bruto del leaderboard sin corregir por fuerza del rival)

- Agente embarcado en ambos slots: `imitation-final` (submissions 55438687 y 55438655),
  clon de Yushin Ito con perfil ALAKAZAM, WR global ~0,586, pero solo ~28,6% contra rivales
  de 800+ elo. Rango real muy por debajo del corte de bronce (~top 10%).
- Payload: data/models/bc_alakazam_final.json. Mazo: decks/yushinito.csv.
- El único instrumento que ha acertado prediciendo el ladder real: la arena directa n=600
  con control de ruido (scratchpad/test_ctx_heads.py) — dos aciertos confirmados hasta ahora.

## Las diez palancas cerradas hoy con evidencia — NO REPETIR sin datos nuevos

1. Cambiar de maestro: Majkel1337 (#1 mundial) su Alakazam ≈ Yushin (1σ, n=1000 cada uno) y
   su submission activa ya no juega Alakazam (cambió a Mega Lucario, solo 273 eps). Yan
   (clonabilidad 0,813, la más alta jamás medida) **pierde 0,275** en la arena decisiva con
   control calibrado en 0,505. (docs/m45_findings.md)
2. Enriquecer features de MAIN (perfil ALAKAZAM_MAIN2, ataca el mismo cuello de botella
   B⊗R que M34 arregló en TO_HAND): +0,0074 en MLP contra barra de +0,020. Falla.
3. Ensemble k=3→k=7 de MAIN: entrenado en GPU vía Colab (+0,0076 offline), **falla en la
   arena decisiva** (indistinguible del control de ruido, incluso negativo contra
   Grimmsnarl). (docs/m46_findings.md)
4. RL self-play tradicional (M16/M19/M38, sin recorte de PPO): 3 intentos, ninguno superó
   con confianza a la versión congelada; el segundo perdió con confianza.
5-10. (ver docs/m31-m44_findings.md) Capacidad/arquitectura, deck composition, 6+ hipótesis
   de conducta sobre la brecha de ensamblaje — todas falsificadas con medición.

## Lo que quedó EN CURSO — la línea más prometedora ahora mismo

**Hoy se identificó y validó parcialmente un hallazgo real: el RL de auto-juego (M38)
carecía del recorte de razón de probabilidad de PPO** — reutilizaba el mismo lote de
partidas 10 épocas sin corrección, lo cual coincide exactamente con la deriva medida
(acuerdo con la política BC cayendo de 96,6% a 88,4% en 10 iteraciones, sin ganancia de
tasa de victoria).

**Prueba retroactiva en Colab (GPU, sin jugar una sola partida nueva)**, reusando las 10
iteraciones de partidas de auto-juego ya guardadas en `data/rl_alakazam/iter1..10_traj.jsonl.gz`
+ checkpoints encadenados `data/models/rl_alakazam_iter1..10.json`:

| transición | sin recorte (agree) | con recorte (agree) | delta |
|---|---|---|---|
| 4→5 | 0,9065 | 0,9303 | **+0,024** |
| 6→7 | 0,9012 | 0,9090 | +0,008 |
| 9→10 | 0,8870 | 0,9090 | **+0,022** |

**3 de 3 a favor del recorte, en las dos métricas (acuerdo con BC init y precisión contra
el maestro).** Caveat honesto: mi puerto a PyTorch/Colab no reproduce exactamente los
hiperparámetros históricos (sobre todo en 6→7, donde la corrida real se recuperó y mi
réplica no) — así que la comparación RELATIVA (con/sin recorte, mismos datos) es confiable,
pero los números absolutos no calzan perfecto con el manifiesto real de M38.

Script de la prueba: `build/colab_rl_extra/colab_rl_clip_test.py` (usa PyTorch + autograd
para el gradiente recortado, evita derivar el gradiente a mano). Paquete ya armado en
`build/colab_rl_extra.zip` (~12 MB) para subir a Colab junto con el `colab_export.zip`
original de M31/M46 (que trae card_data.json, split.json, yushinito_full.jsonl.gz,
ptcg_ai/ podado).

## Lo que falta decidir mañana

1. **Llevar el recorte de PPO al código local** (`scratchpad/rl_selfplay.py::rl_update`),
   validado ya en Colab pero nunca corrido con recolección de partidas nueva — eso
   **no puede correr en Colab** (necesita el motor nativo `cg`, solo disponible local).
2. **Decidir cuántas iteraciones reales correr** (cada una ~20-30 min de recolección local
   con batch=2000 partidas; M38 corrió 10 para ver la tendencia completa) dado el plazo
   real que queda hasta el domingo.
3. Ideas de RL profundo discutidas para sumar DESPUÉS de validar el recorte solo, en orden
   de prioridad honesta:
   - **Crítico privilegiado/asimétrico** (ve la partida sin niebla de guerra durante el
     entrenamiento, nunca se embarca) — ataca directo la miscalibración del crítico que
     M17/M18 ya documentaron. Técnica real (AlphaStar, OpenAI Five). Costo moderado.
   - **Recompensa moldeada con el conocimiento de dominio ya verificado** (hitos de
     ensamblaje, retención de piezas, disparo de Powerful Hand) en vez de solo
     ganó/perdió al final — aprovecha meses de diagnóstico ya hecho, pero riesgo real de
     "reward hacking" si se moldea mal.
   - **Pesos de rival adaptativos** (curriculum): M38 documentó que Kangaskhan quedaba
     "opacado" por los otros rivales del auto-juego hasta que alguien subió su peso a
     mano; la versión automática ajusta el peso según el WR reciente de cada iteración.
   - Bonificación de entropía (barata, bajo riesgo, estándar en PPO).
   - GAE (estimación de ventaja multi-paso) — real pero más compleja, dado el tiempo.
   - **NO recomendado**: agrandar la red — ya se probó dos veces a escala real (M31, M37),
     sin payoff en el ladder.

## Reglas de oro de esta sesión, aprendidas con evidencia (no romperlas)

- El motor es estocástico: nunca decidir con n=200 (M41: el espejo consigo mismo leyó
  0,435 a n=200 y 0,503 a n=600). Toda arena decisiva va a n=600 con control de ruido.
- No correr dos entrenamientos de MAIN en paralelo localmente — RAM insuficiente (~15,6 GB
  total, historial de OOM silencioso hoy mismo).
- MAIN nunca se toca vía `build_fetch_weights.py` (guardián deliberado desde M14); cualquier
  cambio a MAIN pasa por un script dedicado con verificación sha del resto de contextos.
- Colab acelera el ENTRENAMIENTO (matrices densas) pero NO la recolección de auto-juego
  (necesita el motor nativo, que no está en el paquete stdlib-only que se sube a Colab).
- Verificar SIEMPRE el tarball extraído (mazo real, `_IMITATION_READY`, cabezas disparando)
  antes de subir — M37 perdió una subida entera por un mazo mal empaquetado que pasó todas
  las validaciones estructurales sin detectarlo.
- El usuario prefiere: verificar reclamos con datos antes de aceptarlos, señal barata antes
  de comprometer tiempo grande, honestidad directa sobre probabilidades (no inflar
  expectativas), y no rendirse ante callejones sin salida — pensar en la siguiente idea.

Retoma desde aquí: decide el alcance de la corrida real de RL con el recorte de PPO, y si
hay tiempo, cuál de las mejoras adicionales (crítico privilegiado, recompensa moldeada,
curriculum) sumar antes del domingo.
