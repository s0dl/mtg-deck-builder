import { FormEvent, useState } from "react";
import { Wand2 } from "lucide-react";

import { DeckResult } from "./components/DeckResult";
import { DeckResponse, generateDeck } from "./lib/api";
import "./styles/app.css";

const formats = ["standard", "pioneer", "modern", "legacy", "vintage", "commander", "pauper", "casual"];
const colors = ["W", "U", "B", "R", "G"];

function parseList(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function App() {
  const [format, setFormat] = useState("modern");
  const [budget, setBudget] = useState("150");
  const [selectedColors, setSelectedColors] = useState<string[]>(["U", "R"]);
  const [playstyle, setPlaystyle] = useState("tempo");
  const [strategy, setStrategy] = useState("Efficient threats, cheap interaction, and card selection.");
  const [mustInclude, setMustInclude] = useState("");
  const [avoid, setAvoid] = useState("");
  const [deck, setDeck] = useState<DeckResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const result = await generateDeck({
        format,
        budget_usd: budget ? Number(budget) : null,
        colors: selectedColors,
        playstyle,
        strategy,
        must_include: parseList(mustInclude),
        avoid: parseList(avoid)
      });
      setDeck(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Deck generation failed.");
    } finally {
      setIsLoading(false);
    }
  }

  function toggleColor(color: string) {
    setSelectedColors((current) =>
      current.includes(color) ? current.filter((item) => item !== color) : [...current, color]
    );
  }

  return (
    <main className="app-shell">
      <section className="builder">
        <div className="intro">
          <p className="eyebrow">MCP + RAG + deterministic skills</p>
          <h1>MTG Deck Builder Agent</h1>
        </div>

        <form className="deck-form" onSubmit={handleSubmit}>
          <label>
            Format
            <select value={format} onChange={(event) => setFormat(event.target.value)}>
              {formats.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>

          <label>
            Budget USD
            <input
              min="0"
              type="number"
              value={budget}
              onChange={(event) => setBudget(event.target.value)}
            />
          </label>

          <fieldset>
            <legend>Colors</legend>
            <div className="color-row">
              {colors.map((color) => (
                <button
                  className={selectedColors.includes(color) ? "color active" : "color"}
                  key={color}
                  onClick={() => toggleColor(color)}
                  title={`${color} color identity`}
                  type="button"
                >
                  {color}
                </button>
              ))}
            </div>
          </fieldset>

          <label>
            Playstyle
            <input value={playstyle} onChange={(event) => setPlaystyle(event.target.value)} />
          </label>

          <label className="wide">
            Strategy
            <textarea value={strategy} onChange={(event) => setStrategy(event.target.value)} />
          </label>

          <label>
            Must Include
            <input
              placeholder="Ragavan, Ledger Shredder"
              value={mustInclude}
              onChange={(event) => setMustInclude(event.target.value)}
            />
          </label>

          <label>
            Avoid
            <input
              placeholder="Expensive fetch lands"
              value={avoid}
              onChange={(event) => setAvoid(event.target.value)}
            />
          </label>

          <button className="submit" disabled={isLoading} type="submit">
            <Wand2 size={18} />
            <span>{isLoading ? "Building" : "Generate Deck"}</span>
          </button>
        </form>

        {error && <p className="form-error">{error}</p>}
      </section>

      {deck && <DeckResult deck={deck} />}
    </main>
  );
}
