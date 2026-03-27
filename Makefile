.PHONY: help ingest serve query clean full

help:
	@echo "Available targets:"
	@echo "  ingest  — run pipeline with MAX_PAGES=10 (quick test)"
	@echo "  serve   — start the API server"
	@echo "  query   — send a sample question to POST /query"
	@echo "  clean   — remove generated index and metadata"
	@echo "  full    — run ingest then start the API server"

ingest:
	MAX_PAGES=10 docker compose run --rm pipeline

serve:
	docker compose up api

query:
	curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"question": "How does Kubeflow Pipelines handle artifact passing between components?"}' \
		| python3 -m json.tool

clean:
	rm -f data/index.faiss data/meta.json

full: ingest serve
