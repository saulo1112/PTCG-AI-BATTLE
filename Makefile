# ptcg-ai — atajos de comandos frecuentes.
#
# Uso: `make <target>`. Corre `make help` para ver la lista con descripciones.
# Todos los targets son atajos sobre `uv run ptcg ...` (ver docs/developer_guide.md
# y docs/environment.md para el detalle de cada comando).

.DEFAULT_GOAL := help
.PHONY: help sync test test-unit test-integration \
        battle battle-trace battle-verbose \
        bench bench-battle bench-parser bench-search \
        collect analyze show-episode \
        show-obs capture-fixtures validate-deck \
        build-submission validate-submission submit-check \
        clean

# --- Variables sobreescribibles: `make battle GAMES=10` --------------------
GAMES        ?= 1
POLICY       ?= safe-random
OPPONENT     ?= $(POLICY)
TRACE_FILE   ?= build/trace.jsonl
OBS_FILE     ?=
DECK_FILE    ?= pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv
BENCH_N      ?=
SUBMISSION   ?= build/submission.tar.gz
EXP          ?= my-experiment
SEED         ?= 1
EPISODE      ?=
STEP         ?=

help: ## Muestra esta ayuda
	@echo "Targets disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk -F':.*##' '{printf "  %-22s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (sobreescribir con VAR=valor): GAMES, POLICY, OPPONENT,"
	@echo "TRACE_FILE, OBS_FILE, DECK_FILE, BENCH_N, SUBMISSION, EXP, SEED, EPISODE, STEP"

sync: ## Instala/actualiza dependencias (uv sync)
	uv sync

test: ## Corre toda la suite de pruebas (unit + integration si el motor está disponible)
	uv run pytest

test-unit: ## Solo pruebas unitarias (no requieren el motor nativo)
	uv run pytest tests/unit -q

test-integration: ## Solo pruebas de integración (requieren el motor nativo)
	uv run pytest tests/integration -q

battle: ## Corre GAMES partidas locales (por defecto 1) entre POLICY y OPPONENT
	uv run ptcg battle --games $(GAMES) --policy $(POLICY) --opponent $(OPPONENT)

battle-trace: ## Igual que `battle` pero guarda la traza de decisiones en TRACE_FILE
	uv run ptcg battle --games $(GAMES) --policy $(POLICY) --opponent $(OPPONENT) --trace $(TRACE_FILE)

battle-verbose: ## Partida mostrando cada decisión: tablero, acciones nombradas, elección, tiempo
	uv run ptcg battle --games $(GAMES) --policy $(POLICY) --opponent $(OPPONENT) --verbose

collect: ## Corre GAMES partidas y las guarda como experimento. Uso: make collect EXP=nombre GAMES=1000
	uv run ptcg collect --name $(EXP) --games $(GAMES) --seed $(SEED)

analyze: ## Agrega un experimento a reportes (report.md/json + tables/*.csv). Uso: make analyze EXP=nombre
	uv run ptcg analyze data/experiments/$(EXP)

show-episode: ## Resumen de un episodio grabado (STEP=N para ver una decisión). Uso: make show-episode EPISODE=ruta
	@test -n "$(EPISODE)" || (echo "Uso: make show-episode EPISODE=data/experiments/x/episodes/00000.jsonl.gz [STEP=N]"; exit 1)
	uv run ptcg show-episode $(EPISODE) $(if $(STEP),--step $(STEP),)

bench: ## Corre los tres benchmarks (battle+parser+search) con el perfil "benchmark"
	uv run ptcg --profile benchmark bench all $(if $(BENCH_N),--n $(BENCH_N),)

bench-battle: ## Solo el benchmark de throughput de partidas
	uv run ptcg --profile benchmark bench battle $(if $(BENCH_N),--n $(BENCH_N),)

bench-parser: ## Solo el benchmark de velocidad del parser (usa los fixtures capturados)
	uv run ptcg --profile benchmark bench parser $(if $(BENCH_N),--n $(BENCH_N),)

bench-search: ## Solo el benchmark de la Search API del motor (ADR-0006)
	uv run ptcg --profile benchmark bench search $(if $(BENCH_N),--n $(BENCH_N),)

show-obs: ## Imprime en texto legible una observación capturada. Uso: make show-obs OBS_FILE=ruta.json
	@test -n "$(OBS_FILE)" || (echo "Uso: make show-obs OBS_FILE=tests/fixtures/observations/004_MAIN_MAIN.json"; exit 1)
	uv run ptcg show-obs $(OBS_FILE) --logs

capture-fixtures: ## Juega una partida real y guarda observaciones de ejemplo en tests/fixtures/observations
	uv run ptcg capture-fixtures

validate-deck: ## Valida un deck.csv contra las reglas del juego. Uso: make validate-deck DECK_FILE=ruta.csv
	uv run ptcg validate-deck $(DECK_FILE)

build-submission: ## Arma el submission.tar.gz para Kaggle (perfil "submission")
	uv run ptcg --profile submission build-submission

validate-submission: ## Valida un tarball ya construido. Uso: make validate-submission SUBMISSION=build/x.tar.gz
	uv run ptcg validate-submission $(SUBMISSION)

submit-check: build-submission validate-submission ## Arma y valida el submission en un solo paso (antes de subir a Kaggle)

clean: ## Borra artefactos generados (build/, tarballs, cachés de pytest)
	rm -rf build .pytest_cache .coverage

replay: ## Descarga replays: make replay SUBMISSIONS="53787873 53909538 53815664"
	uv run python tools/download_competitors.py --submissions $(SUBMISSIONS) --cookies tools/cookies.txt.json --out replays/
