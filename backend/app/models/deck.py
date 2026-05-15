from enum import StrEnum

from pydantic import BaseModel, Field


class Format(StrEnum):
    standard = "standard"
    pioneer = "pioneer"
    modern = "modern"
    legacy = "legacy"
    vintage = "vintage"
    commander = "commander"
    pauper = "pauper"
    casual = "casual"


class DeckRequest(BaseModel):
    format: Format = Format.modern
    budget_usd: float | None = Field(default=None, ge=0)
    colors: list[str] = Field(default_factory=list, description="Preferred color letters, such as U/R.")
    playstyle: str = Field(default="", examples=["tempo", "aggro", "control", "combo"])
    strategy: str = Field(default="", description="Natural-language goal for the deck.")
    must_include: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class DeckCard(BaseModel):
    name: str
    count: int = Field(ge=1)
    role: str = ""
    estimated_price_usd: float | None = None


class ManaCurveBucket(BaseModel):
    mana_value: int
    count: int


class DeckValidation(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    label: str
    detail: str


class DeckResponse(BaseModel):
    title: str
    format: Format
    cards: list[DeckCard]
    sideboard: list[DeckCard] = Field(default_factory=list)
    explanation: str
    mana_curve: list[ManaCurveBucket] = Field(default_factory=list)
    validation: DeckValidation
    retrieved_context: list[str] = Field(default_factory=list)
    agent_steps: list[AgentStep] = Field(default_factory=list)
    generation_mode: str = "deterministic"
