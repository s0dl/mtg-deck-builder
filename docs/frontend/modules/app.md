# App Module

Path: `src/App.tsx`

## Purpose

The app module owns the main deck generation workspace.

## Responsibilities

- Manage form state for format, budget, colors, playstyle, strategy, must-includes, and avoid terms.
- Convert comma-separated text inputs into arrays for the backend.
- Submit `DeckRequest` payloads.
- Track loading and error state.
- Render the result panel.
- Apply selected visual theme through the root `data-theme` attribute.

## Theme Variants

Current themes:

- Verdant.
- Arcane.
- Nocturne.

Themes are presentation-only and do not affect backend generation.
