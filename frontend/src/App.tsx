import { FormEvent, useState } from "react";
import { flushSync } from "react-dom";
import { Moon, Sparkles, Sun, Wand2 } from "lucide-react";

import { AgentActivity } from "./components/AgentActivity";
import { DeckResult } from "./components/DeckResult";
import { generateDeck } from "./lib/api";
import type { DeckRequest, DeckResponse } from "./lib/api";
import "./styles/app.css";

const formats = ["standard", "pioneer", "modern", "legacy", "vintage", "commander", "pauper", "casual"];
const colors = ["W", "U", "B", "R", "G"];
const themes = [
  { id: "verdant", label: "Verdant", icon: Sun },
  { id: "arcane", label: "Arcane", icon: Sparkles },
  { id: "nocturne", label: "Nocturne", icon: Moon }
];

function parseList(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function updateWithViewTransition(update: () => void) {
  const transitionDocument = document as Document & {
    startViewTransition?: (callback: () => void) => unknown;
  };

  if (typeof transitionDocument.startViewTransition === "function") {
    transitionDocument.startViewTransition(() => flushSync(update));
    return;
  }

  update();
}

export default function App() {
  const [format, setFormat] = useState("modern");
  const [budget, setBudget] = useState("150");
  const [selectedColors, setSelectedColors] = useState<string[]>(["U", "R"]);
  const [playstyle, setPlaystyle] = useState("tempo");
  const [strategy, setStrategy] = useState("Efficient threats, cheap interaction, and card selection.");
  const [mustInclude, setMustInclude] = useState("");
  const [avoid, setAvoid] = useState("");
  const [theme, setTheme] = useState("verdant");
  const [deck, setDeck] = useState<DeckResponse | null>(null);
  const [activeRequest, setActiveRequest] = useState<DeckRequest | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const payload: DeckRequest = {
      format,
      budget_usd: budget ? Number(budget) : null,
      colors: selectedColors,
      playstyle,
      strategy,
      must_include: parseList(mustInclude),
      avoid: parseList(avoid)
    };

    setActiveRequest(payload);
    setIsLoading(true);
    setError("");
    setDeck(null);

    try {
      const result = await generateDeck(payload);
      updateWithViewTransition(() => {
        setDeck(result);
        setIsLoading(false);
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Deck generation failed.");
      setIsLoading(false);
    }
  }

  function toggleColor(color: string) {
    setSelectedColors((current) =>
      current.includes(color) ? current.filter((item) => item !== color) : [...current, color]
    );
  }

  return (
    <main className={isLoading ? "app-shell is-building" : "app-shell"} data-theme={theme}>
      {isLoading ? (
        <AgentActivity isLive mode="immersive" request={activeRequest} />
      ) : (
        <>
          <section className="builder" aria-label="Deck request">
            <div className="intro-block">
              <div className="intro">
                <p className="eyebrow">MCP + RAG + Scryfall tools</p>
                <h1>MTG Deck Builder Agent</h1>
              </div>

              <div className="theme-picker" aria-label="Theme">
                {themes.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      aria-pressed={theme === item.id}
                      className={theme === item.id ? "theme-button active" : "theme-button"}
                      key={item.id}
                      onClick={() => setTheme(item.id)}
                      title={`${item.label} theme`}
                      type="button"
                    >
                      <Icon size={16} />
                    </button>
                  );
                })}
              </div>
            </div>

            <form className="deck-form" onSubmit={handleSubmit}>
              <div className="form-grid">
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
              </div>

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

              <div className="form-grid">
                <label>
                  Playstyle
                  <input value={playstyle} onChange={(event) => setPlaystyle(event.target.value)} />
                </label>

                <label>
                  Must Include
                  <input
                    placeholder="Ragavan, Ledger Shredder"
                    value={mustInclude}
                    onChange={(event) => setMustInclude(event.target.value)}
                  />
                </label>
              </div>

              <label className="wide">
                Strategy
                <textarea value={strategy} onChange={(event) => setStrategy(event.target.value)} />
              </label>

              <label>
                Avoid
                <input
                  placeholder="Expensive fetch lands"
                  value={avoid}
                  onChange={(event) => setAvoid(event.target.value)}
                />
              </label>

              <div className="form-actions">
                <button className="submit" disabled={isLoading} type="submit">
                  <Wand2 size={18} />
                  <span>{isLoading ? "Building" : "Generate Deck"}</span>
                </button>
              </div>
            </form>

            {error && <p className="form-error">{error}</p>}
          </section>

          {deck && <DeckResult deck={deck} />}
        </>
      )}
    </main>
  );
}
