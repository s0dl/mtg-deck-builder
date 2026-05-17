# Repository Guidelines

## Project Structure & Module Organization

- `backend/` contains the FastAPI service, MCP tool server, RAG pipeline, agent orchestration, deterministic skills, and ingestion scripts.
- `frontend/` contains the React + TypeScript UI and API client.
- `database/` contains Postgres and `pgvector` setup.
- `docs/` contains architecture notes and module references.
- Tests live in `backend/tests/`; shared runtime code lives under `backend/app/`.

## Build, Test, and Development Commands

- `docker compose up --build` starts the full stack locally.
- `cd backend && source venv/bin/activate && pip install -e ".[dev,agent]" && uvicorn app.main:app --reload` runs the backend in development.
- `cd backend && pytest` runs backend tests.
- `cd backend && python -m ruff check app tests scripts` runs lint checks.
- `cd frontend && npm install && npm run dev` starts the frontend.
- `cd frontend && npm run build` type-checks and builds the UI.

## Coding Style & Naming Conventions

- Python targets 3.12, uses 4-space indentation, type hints, and `ruff`-compatible formatting.
- TypeScript uses modern React function components, `camelCase` for variables/functions, and `PascalCase` for components.
- Keep module names lowercase and descriptive, matching existing paths like `app/mcp/server.py` and `src/components/DeckResult.tsx`.
- Prefer small, explicit helpers over implicit cross-module behavior.

## Testing Guidelines

- Use `pytest` for backend behavior tests.
- Name tests `test_*.py` and keep them close to the behavior they verify.
- Prioritize coverage for deck validation, MCP tool routing, RAG retrieval, and agent workflow ordering.
- When adding a new constraint or tool path, add a regression test in `backend/tests/`.

## Commit & Pull Request Guidelines

- Commit messages in this repo are short and imperative, e.g. `Added sideboard` or `Moved to agents sdk`.
- Keep commits focused on one change set.
- PRs should include a short summary, testing notes, and screenshots or video for frontend changes.
- Link any related issue or task, and call out behavior changes to deck generation, validation, or retrieval.

## Security & Configuration Tips

- Never commit secrets such as `OPENAI_API_KEY` or database credentials.
- In Docker, use the service hostname `postgres`; from the host, point `DATABASE_URL` at `localhost`.
- RAG ingestion and embedding settings are environment-dependent, so verify them before reindexing data.
