import httpx
from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/status")
async def agent_status() -> dict[str, object]:
    settings = get_settings()
    payload: dict[str, object] = {
        "provider": settings.agent_provider,
        "openai_enabled": settings.openai_enabled,
        "openai_agent_enabled": settings.openai_agent_enabled,
        "openai_model": settings.openai_model,
        "ollama_enabled": settings.ollama_enabled,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "ollama_reachable": False,
    }
    if settings.agent_provider.strip().lower() == "openai":
        if settings.openai_enabled:
            payload["message"] = "OpenAI agent is configured. Live API reachability is checked during generation."
        else:
            payload["message"] = "Set OPENAI_API_KEY to enable the OpenAI agent."
        return payload

    if not settings.ollama_enabled:
        payload["message"] = "Set AGENT_PROVIDER=openai or AGENT_PROVIDER=ollama to enable an agent."
        return payload

    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=3) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
    except httpx.HTTPError as exc:
        payload["message"] = f"Ollama is not reachable: {exc}"
        return payload

    payload["ollama_reachable"] = True
    payload["available_models"] = [model.get("name") for model in models if model.get("name")]
    payload["message"] = "Ollama agent is configured and reachable."
    return payload
