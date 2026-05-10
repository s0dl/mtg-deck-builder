export type DeckRequest = {
  format: string;
  budget_usd: number | null;
  colors: string[];
  playstyle: string;
  strategy: string;
  must_include: string[];
  avoid: string[];
};

export type DeckCard = {
  name: string;
  count: number;
  role: string;
  estimated_price_usd?: number | null;
};

export type ManaCurveBucket = {
  mana_value: number;
  count: number;
};

export type DeckResponse = {
  title: string;
  format: string;
  cards: DeckCard[];
  sideboard: DeckCard[];
  explanation: string;
  mana_curve: ManaCurveBucket[];
  validation: {
    is_valid: boolean;
    errors: string[];
    warnings: string[];
  };
  retrieved_context: string[];
};

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function generateDeck(payload: DeckRequest): Promise<DeckResponse> {
  const response = await fetch(`${apiBaseUrl}/api/decks/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`Deck generation failed with status ${response.status}`);
  }

  return response.json();
}
