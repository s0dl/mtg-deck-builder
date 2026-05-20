from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings


async def run_structured_openai_agent(
    *,
    settings: Settings,
    name: str,
    instructions: str,
    input_payload: dict[str, Any],
    output_type: type[BaseModel],
    tools: list[Any] | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    try:
        from agents import Agent, Runner
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Agents SDK is required for OpenAI generation. "
            "Install backend dependencies so the 'openai-agents' package is available."
        ) from exc

    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = settings.openai_base_url

    agent = Agent(
        name=name,
        instructions=instructions,
        model=settings.openai_model,
        output_type=output_type,
        tools=tools or [],
    )
    result = await Runner.run(
        agent,
        json.dumps(input_payload, separators=(",", ":")),
        max_turns=max_turns,
    )
    final_output = result.final_output
    if isinstance(final_output, BaseModel):
        return final_output.model_dump()
    if isinstance(final_output, dict):
        return final_output
    if isinstance(final_output, str):
        return json.loads(final_output)
    raise ValueError(f"OpenAI Agents SDK returned unsupported output: {type(final_output).__name__}")
