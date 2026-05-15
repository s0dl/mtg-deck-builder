# Backend API Reference

Base URL in local Docker is `http://localhost:8000`.

## `GET /health`

Health check used by Docker and local smoke tests.

Response:

```json
{
  "status": "ok"
}
```

## `GET /api/agent/status`

Reports which agent provider the backend is configured to use.

Important response fields:

- `provider` - configured `AGENT_PROVIDER`.
- `openai_enabled` - true when `OPENAI_API_KEY` is set.
- `openai_agent_enabled` - true when `AGENT_PROVIDER=openai` and the key is set.
- `openai_model` - configured OpenAI model.
- `ollama_enabled` - true when `AGENT_PROVIDER=ollama`.
- `ollama_reachable` - true only when the backend can reach Ollama.
- `available_models` - Ollama model names when reachable.
- `message` - human-readable status.

Example:

```bash
curl http://localhost:8000/api/agent/status
```

## `POST /api/decks/generate`

Builds a deck draft from a natural-language request.

Request body:

```json
{
  "format": "modern",
  "budget_usd": 150,
  "colors": ["U", "R"],
  "playstyle": "tempo",
  "strategy": "Izzet prowess with cheap threats and burn",
  "must_include": ["Slickshot Show-Off"],
  "avoid": ["Ragavan, Nimble Pilferer"]
}
```

Request fields:

- `format` - one of `standard`, `pioneer`, `modern`, `legacy`, `vintage`, `commander`, `pauper`, or `casual`.
- `budget_usd` - optional non-negative budget.
- `colors` - preferred color letters such as `U` and `R`.
- `playstyle` - short archetype descriptor such as `aggro`, `tempo`, `control`, `combo`, `midrange`, or `ramp`.
- `strategy` - natural-language deck goal.
- `must_include` - card names to seed into searches and final selection when legal.
- `avoid` - card names or terms to avoid where supported.

Response fields:

- `title` - generated deck name.
- `format` - response format.
- `cards` - main deck card rows with `name`, `count`, `role`, and optional `estimated_price_usd`.
- `sideboard` - sideboard card rows. This is currently usually empty.
- `explanation` - model or fallback explanation.
- `mana_curve` - count by mana value.
- `validation` - deterministic validity, errors, and warnings.
- `retrieved_context` - context titles used during generation.
- `agent_steps` - visible agent activity and tool-use summary.
- `generation_mode` - path used for the response.

Generation modes currently include:

- `openai_agent` - OpenAI agent planned tool calls and selected cards.
- `openai` - legacy OpenAI structured deck builder path.
- `ollama_agent` - local Ollama agent path.
- `deterministic` - backend fallback assembly.

The backend finalizes all paths by merging duplicate card rows, trimming oversized responses, adding basic lands to the target size, calculating mana curve, and running deterministic validation.
