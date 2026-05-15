# Models Module

Path: `app/models`

## Purpose

Pydantic request and response contracts shared by API routes and frontend clients.

## Main File

- `deck.py`

## Key Models

- `Format` - supported Magic formats.
- `DeckRequest` - user input for deck generation.
- `DeckCard` - card row with count, role, and optional estimated price.
- `ManaCurveBucket` - curve summary by mana value.
- `DeckValidation` - deterministic validity, errors, and warnings.
- `AgentStep` - user-visible agent activity entry.
- `DeckResponse` - complete response payload.

Keep these models stable when possible because the frontend API client mirrors their shape.
