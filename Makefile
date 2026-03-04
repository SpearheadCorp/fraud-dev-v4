.PHONY: help build up pipeline stop reset stress stress-stop logs clean

help:
	@echo "fraud-det-v3.1"
	@echo "  make build        - Build all Docker images"
	@echo "  make up           - Start backend + monitoring (dashboard at :8080)"
	@echo "  make pipeline     - Run the ML pipeline once (gather→prep→train→infer)"
	@echo "  make stop         - Stop all services"
	@echo "  make reset        - Stop and clear raw/features data (models preserved)"
	@echo "  make stress       - Trigger stress mode via API"
	@echo "  make stress-stop  - Stop stress mode"
	@echo "  make logs         - Tail all logs"
	@echo "  make clean        - Remove all containers and volumes"

build:
	docker compose build

up:
	docker compose up -d backend prometheus dcgm-exporter
	@echo ""
	@echo "  Dashboard:  http://localhost:8080"
	@echo "  Prometheus: http://localhost:9090"
	@echo ""

pipeline:
	docker compose --profile pipeline up

stop:
	docker compose stop

reset:
	docker compose stop
	docker compose run --rm backend python -c \
		"import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) or pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ['/data/raw', '/data/features']]"

stress:
	curl -s -X POST http://localhost:8080/api/control/stress | python3 -m json.tool

stress-stop:
	curl -s -X POST http://localhost:8080/api/control/stress-stop | python3 -m json.tool

logs:
	docker compose logs -f

clean:
	docker compose down -v --remove-orphans
