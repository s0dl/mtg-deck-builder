# API Client Module

Path: `src/lib/api.ts`

## Purpose

Typed client wrapper for backend calls and shared response types.

## Responsibilities

- Define frontend copies of `DeckRequest`, `DeckResponse`, `DeckCard`, `DeckValidation`, `ManaCurveBucket`, and `AgentStep`.
- Resolve `VITE_API_BASE_URL`, defaulting to `http://localhost:8000`.
- Submit deck generation requests.
- Surface backend errors as thrown JavaScript errors for the app layer.

Keep this file aligned with `backend/app/models/deck.py`.
