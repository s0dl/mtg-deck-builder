# Frontend

TypeScript React UI for MTG Deck Builder Agent.

## Responsibilities

- Capture deck-building goals: format, budget, colors, playstyle, strategy, must-includes, and exclusions.
- Submit requests to the FastAPI backend.
- Render generated cards, validation, mana curve, price estimate, retrieved context, and agent activity.
- Provide theme variants for the generation workspace.

## Documentation

- [Frontend Docs Index](../docs/frontend/README.md)
- [App Module](../docs/frontend/modules/app.md)
- [API Client Module](../docs/frontend/modules/api-client.md)
- [Components Module](../docs/frontend/modules/components.md)
- [Styles Module](../docs/frontend/modules/styles.md)

## Development

```bash
npm install
npm run dev
```

The app expects `VITE_API_BASE_URL`, defaulting to `http://localhost:8000`.
