"""Seed a small local RAG corpus for development."""

from app.core.database import SessionLocal
from app.rag.embeddings import get_embedding_provider
from app.rag.ingestion import build_strategy_document
from app.rag.repository import RagRepository

SEED_DOCUMENTS = [
    {
        "source": "foundational_strategy",
        "source_id": "mana-curve-basics",
        "title": "Mana Curve Basics",
        "content": (
            "Aggressive decks need a dense early mana curve with enough one and two mana plays "
            "to affect the board immediately. Midrange decks can afford more three and four mana "
            "cards when they have cheap interaction. Control decks still need early answers, but "
            "their win conditions can sit higher on the curve because they plan to extend the game."
        ),
        "metadata": {"topic": "mana_curve"},
    },
    {
        "source": "foundational_strategy",
        "source_id": "tempo-principles",
        "title": "Tempo Deck Principles",
        "content": (
            "Tempo decks pair efficient threats with cheap interaction and card selection. The goal "
            "is to deploy a threat, protect it, and trade mana efficiently so the opponent cannot "
            "stabilize. Cantrips, soft counters, burn, and evasive creatures are common building blocks."
        ),
        "metadata": {"topic": "tempo", "archetypes": ["tempo"]},
    },
    {
        "source": "foundational_strategy",
        "source_id": "budget-building",
        "title": "Budget Deck Building",
        "content": (
            "Budget lists should preserve the deck's functional core before upgrading mana bases or "
            "sideboard bullets. Expensive lands can often be replaced first, but decks with strict "
            "color requirements may lose consistency if the mana base is cut too aggressively."
        ),
        "metadata": {"topic": "budget"},
    },
]


def main() -> None:
    documents = []
    for seed in SEED_DOCUMENTS:
        documents.extend(build_strategy_document(**seed))

    with SessionLocal() as session:
        repository = RagRepository(session, get_embedding_provider())
        count = repository.upsert_many(documents)

    print(f"Seeded {count} RAG documents")


if __name__ == "__main__":
    main()
