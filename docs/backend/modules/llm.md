# LLM Module

Path: `app/llm`

## Purpose

The LLM module contains the simpler structured OpenAI deck builder. It is separate from the fuller agent module.

## Main File

- `deck_builder.py` - OpenAI Agents SDK call with Pydantic structured output.

## Current Role

The preferred path is `app/agent` with OpenAI Agents SDK tool planning. The LLM module remains useful as a simpler model-backed fallback that receives prepared context and returns a deck-shaped JSON response.

All model output is still filtered and finalized by the API layer before it reaches the frontend.
