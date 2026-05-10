# Architecture Notes

The agent is intentionally split into three independent layers:

- MCP is live truth.
- RAG is historical and strategic context.
- Skills are deterministic constraints.

Deck generation should call all three:

1. Parse the user's request into a structured `DeckRequest`.
2. Retrieve relevant strategic context from pgvector.
3. Use MCP for current legality and price data.
4. Construct candidate cards using retrieved context and live filters.
5. Run deterministic validation and scoring.
6. Return the final list, warnings, substitutions, and explanations.

This keeps creative reasoning separate from objective constraints. It also makes the system easier to test because the skill layer can be validated without the model, embeddings, or Scryfall.
