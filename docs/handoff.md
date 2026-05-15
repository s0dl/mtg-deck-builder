# Current Handoff

## Current State

The app now has a working agent-oriented deck generation path:

- `POST /api/decks/generate` returns deck cards, validation, mana curve, retrieved context, agent steps, and generation mode.
- `GET /api/agent/status` reports OpenAI/Ollama configuration.
- RAG stores rules, MTGDecks articles, MTGDecks meta deck snapshots, and Scryfall card corpus records.
- OpenAI embeddings and OpenAI agent generation are supported.
- FastEmbed remains available for local embeddings.
- Ollama remains available for local experimentation, but may be too weak for reliable structured deck construction.
- The frontend shows agent activity, context, validation, pricing, and selectable themes.
- Candidate pools now pass through a deterministic deck evaluation skill that scores legality, color identity, curve fit, role fit, budget, and request-term synergy before model/fallback construction.
- Agent strategy retrieval now includes MTGDecks meta deck snapshots, and card corpus retrieval combines vector and lexical search before ranking.
- Live Scryfall search is available to the model as a guarded backend-executed tool for candidate discovery when corpus retrieval is thin.

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
- `backend/app/agent/deck_builder.py` - OpenAI/Ollama agent planning and selection.
- `backend/app/agent/tools.py` - constrained RAG tools plus backend-controlled Scryfall helpers.
- `backend/app/mcp/scryfall_client.py` - live Scryfall API adapter; this is not a protocol-native MCP server yet.
- `backend/app/rag/ingestion.py` - source-specific RAG document builders.
- `backend/app/rag/chunking.py` - text chunking and Scryfall card field formatting.
- `backend/app/rag/embeddings.py` - OpenAI/FastEmbed/hash providers and cache wrapper.
- `backend/app/skills/deck_evaluation.py` - deterministic candidate scoring and ranking.
- `frontend/src/components/DeckResult.tsx` - deck result, agent activity, context, and price display.

## Known Gaps

- Rules retrieval is better than before but still needs explicit rule-intent mapping.
- Card-corpus RAG query quality now matters more because the OpenAI agent uses it for candidate discovery before Scryfall price checks.
- Meta deck snapshots are now retrievable, but `mtgdecks-meta-decks` appears to contain placeholder or incomplete card lists; the actual top deck cards need to be captured and ingested.
- Candidate scoring is still heuristic and should be refined against real generated deck outputs.
- The deterministic fallback remains heuristic and should become a real scoring and construction pipeline.
- The project has deterministic Python "skills", but not model-native/Codex-style skill packages for the deck builder.
- Scryfall live access is still an internal adapter under `app/mcp`, not a protocol-native MCP server.
- Agent tool schemas are currently prompt-described Python methods rather than clear first-class tool schemas with precise inputs/results.

## Recommended Next Work

1. Fix `mtgdecks-meta-decks` ingestion so meta deck snapshots include the actual top deck card lists, not placeholders; then re-ingest and verify those cards appear in retrieved context.
2. Convert Scryfall API requests into an actual MCP server/tool integration instead of the current direct `ScryfallClient` adapter.
3. Define clearer agent tool schemas for strategy search, meta deck search, rules search, card corpus search, live Scryfall search, lookup, and validation.
4. Improve rules retrieval by mapping request formats to explicit rule intents.
5. Refine deterministic construction so the evaluator can enforce role counts rather than only ranking candidates.
