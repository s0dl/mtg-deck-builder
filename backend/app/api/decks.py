from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_session
from app.models.deck import DeckRequest, DeckResponse
from app.rag.retriever import RagRetriever
from app.skills.deck_rules import validate_deck
from app.skills.mana_curve import calculate_mana_curve

router = APIRouter(prefix="/decks", tags=["decks"])


@router.post("/generate", response_model=DeckResponse)
async def generate_deck(request: DeckRequest, session: Session = Depends(get_session)) -> DeckResponse:
    retriever = RagRetriever(session)
    context = retriever.search(
        query=" ".join([request.format.value, request.playstyle, request.strategy]).strip(),
        limit=4,
    )

    # Placeholder shell: production generation will combine retrieved context,
    # Scryfall live filters, ranking, and deterministic validation.
    cards = [
        {"name": name, "count": 1, "role": "requested card", "estimated_price_usd": None}
        for name in request.must_include
    ]

    if not cards:
        cards = [
            {"name": "Lightning Bolt", "count": 4, "role": "efficient interaction", "mana_value": 1},
            {"name": "Consider", "count": 4, "role": "card selection", "mana_value": 1},
            {"name": "Island", "count": 8, "role": "mana source"},
            {"name": "Mountain", "count": 8, "role": "mana source"},
        ]

    validation = validate_deck(cards=cards, mtg_format=request.format)
    mana_curve = calculate_mana_curve(cards)

    return DeckResponse(
        title=f"{request.format.value.title()} {request.playstyle.title() or 'Deck'} Draft",
        format=request.format,
        cards=cards,
        explanation=(
            "This scaffold returns a deterministic placeholder list while the RAG corpus, "
            "embedding provider, and Scryfall-driven card ranking are wired in."
        ),
        mana_curve=mana_curve,
        validation=validation,
        retrieved_context=[f"{item.title} ({item.source})" for item in context],
    )
