.PHONY: dev build clean docker-build

dev:
	cd src-python && python -m magic.app

web:
	cd src-python && python web_server.py

build:
	pnpm build

clean:
	rm -rf dist/ src-tauri/target/

docker-build:
	docker build -t magic-mirror:latest .
