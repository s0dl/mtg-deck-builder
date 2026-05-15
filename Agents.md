# Current Handoff

## Current State

The app now has a working agent-oriented deck generation path:

- `POST /api/decks/generate` returns deck cards, validation, mana curve, retrieved context, agent steps, and generation mode.
- `GET /api/agent/status` reports OpenAI/Ollama configuration.
- RAG stores rules, MTGDecks articles, MTGDecks meta deck snapshots, and Scryfall card corpus records.
- OpenAI embeddings and OpenAI Agents SDK generation are supported.
- FastEmbed remains available for local embeddings.
- Ollama remains available for local experimentation, but may be too weak for reliable structured deck construction.
- The frontend shows agent activity, context, validation, pricing, and selectable themes.
- Candidate pools now pass through a deterministic deck evaluation skill that scores legality, color identity, curve fit, role fit, budget, and request-term synergy before model/fallback construction.
- Agent strategy retrieval now includes MTGDecks meta deck snapshots.
- Live Scryfall search is the model's guarded backend-executed card discovery path.
- RAG, live Scryfall, card corpus, lookup, and validation tools are registered on `DeckBuilderMcpServer`.
- Initial RAG context, model-planned RAG calls, model-planned Scryfall calls, fallback live card discovery, and price refresh now route through the MCP server boundary.

## Current Preferred Setup

```bash
AGENT_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=384
EMBEDDING_CACHE_ENABLED=true
```

Run ingestion in the environment that can reach the configured Postgres host. For host-local scripts, `DATABASE_URL` should use `localhost`. For Docker container scripts, it should use `postgres`.

## Ingestion Commands

From `backend/` with the virtualenv active, or from the backend container:

```bash
python -m scripts.download_mtg_rules --output data/mtg-comprehensive-rules.txt
python -m scripts.ingest_mtg_rules data/mtg-comprehensive-rules.txt

python -m scripts.download_mtgdecks_articles --pages 3 --output data/mtgdecks-articles.json
python -m scripts.ingest_mtgdecks_articles data/mtgdecks-articles.json

python -m scripts.download_mtgdecks_meta_decks --formats Standard Pioneer Modern Legacy Pauper Commander --top 30
python -m scripts.ingest_mtgdecks_meta_decks data/mtgdecks-meta-decks.json

python -m scripts.download_scryfall_card_corpus --output data/scryfall-card-corpus.json
python -m scripts.ingest_scryfall_card_corpus data/scryfall-card-corpus.json
```

If old card corpus rows were embedded before field formatting changed, delete or re-upsert `source = 'scryfall_bulk'` rows and rerun the Scryfall corpus ingestion. The embedding cache can stay unless you need to reclaim disk or force every text string through the provider again.

## Important Files

- `backend/app/api/decks.py` - main generation flow, fallback behavior, finalization, and validation.
- `backend/app/api/agent.py` - agent status endpoint.
- `backend/app/agent/deck_builder.py` - OpenAI Agents SDK/Ollama agent planning and selection.
- `backend/app/agent/tools.py` - agent-facing client facade over the MCP server.
- `backend/app/mcp/server.py` - request-scoped MCP tool registry and executor for RAG, Scryfall, card corpus, and validation tools.
- `backend/app/mcp/scryfall_client.py` - live Scryfall API adapter used by the MCP server.
- `backend/app/rag/ingestion.py` - source-specific RAG document builders.
- `backend/app/rag/chunking.py` - text chunking and Scryfall card field formatting.
- `backend/app/rag/embeddings.py` - OpenAI/FastEmbed/hash providers and cache wrapper.
- `backend/app/skills/deck_evaluation.py` - deterministic candidate scoring and ranking.
- `frontend/src/components/DeckResult.tsx` - deck result, agent activity, context, and price display.

## Known Gaps

- Rules retrieval is better than before but still needs explicit rule-intent mapping.
- The Scryfall card corpus remains ingestible, but the OpenAI/Ollama agent paths now use live Scryfall for card discovery.
- Candidate scoring is still heuristic and should be refined against real generated deck outputs.
- The deterministic fallback remains heuristic and should become a real scoring and construction pipeline.
- The MCP server is currently request-scoped and in-process; expose a stdio/HTTP transport if external clients need to call it directly.

## Recommended Next Work

1. Improve rules retrieval by mapping request formats to explicit rule intents.
2. Refine deterministic construction so the evaluator can enforce role counts rather than only ranking candidates.
3. Add an external MCP transport if another process needs to call the deck-builder tools outside the FastAPI request path.
