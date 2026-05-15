# Components Module

Path: `src/components`

## Purpose

Reusable UI components for generation results.

## Main File

- `DeckResult.tsx`

## Responsibilities

- Show title, format, generation mode, deck size, and price estimate.
- Render main deck rows and sideboard rows.
- Show validation errors and warnings.
- Render mana curve buckets.
- Display retrieved context references.
- Display agent activity steps so users can tell whether the model used tools.

Price totals depend on backend card rows containing `estimated_price_usd`. If prices are absent, the UI should show that pricing is unavailable instead of staying pending.
