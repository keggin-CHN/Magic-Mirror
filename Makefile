.PHONY: dev web build check clean docker-build

dev:
	cd src-python && python server.py

web:
	cd src-python && python web_server.py

build:
	pnpm build

check:
	python -m ruff check check_gpu_support.py scripts src-python tests
	python -m pytest tests/ -q
	node --check scripts/dist.js
	pnpm build
	@if command -v bash >/dev/null 2>&1; then \
		bash -n scripts/build-server.sh scripts/install-server-linux.sh scripts/install-web.sh; \
	else \
		echo "bash not found; skip shell syntax check"; \
	fi
	git diff --check

clean:
	rm -rf dist/ src-tauri/target/

docker-build:
	docker build -t magic-mirror:latest .
