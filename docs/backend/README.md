# Backend Docs

The backend is a FastAPI service that owns deck generation, agent orchestration, RAG retrieval, Scryfall access, deterministic skills, and ingestion scripts.

## References

- [API Reference](apis.md)
- [Agent Module](modules/agent.md)
- [API Module](modules/api.md)
- [Core Module](modules/core.md)
- [MCP Module](modules/mcp.md)
- [Models Module](modules/models.md)
- [RAG Module](modules/rag.md)
- [Scripts Module](modules/scripts.md)
- [Skills Module](modules/skills.md)

## Run Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,agent]"
uvicorn app.main:app --reload
```

## Verify

```bash
python -m ruff check app tests scripts
pytest
```
