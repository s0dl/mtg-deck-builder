import { AlertTriangle, CheckCircle2 } from "lucide-react";

import type { DeckResponse } from "../lib/api";

type DeckResultProps = {
  deck: DeckResponse;
};

export function DeckResult({ deck }: DeckResultProps) {
  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-header">
        <div>
          <p className="eyebrow">{deck.format}</p>
          <h2>{deck.title}</h2>
        </div>
        <div className={deck.validation.is_valid ? "status valid" : "status invalid"}>
          {deck.validation.is_valid ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          <span>{deck.validation.is_valid ? "Valid draft" : "Needs fixes"}</span>
        </div>
      </div>

      <p className="explanation">{deck.explanation}</p>

      {(deck.validation.errors.length > 0 || deck.validation.warnings.length > 0) && (
        <div className="messages">
          {deck.validation.errors.map((message) => (
            <p className="error" key={message}>{message}</p>
          ))}
          {deck.validation.warnings.map((message) => (
            <p className="warning" key={message}>{message}</p>
          ))}
        </div>
      )}

      <div className="content-grid">
        <div>
          <h3>Main Deck</h3>
          <div className="card-list">
            {deck.cards.map((card) => (
              <div className="card-row" key={card.name}>
                <strong>{card.count}x {card.name}</strong>
                <span>{card.role}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3>Mana Curve</h3>
          <div className="curve">
            {deck.mana_curve.map((bucket) => (
              <div className="curve-row" key={bucket.mana_value}>
                <span>{bucket.mana_value}</span>
                <div className="bar-track">
                  <div
                    className="bar"
                    style={{ width: `${Math.min(bucket.count * 8, 100)}%` }}
                  />
                </div>
                <strong>{bucket.count}</strong>
              </div>
            ))}
          </div>

          <h3>Retrieved Context</h3>
          {deck.retrieved_context.length > 0 ? (
            <ul className="context-list">
              {deck.retrieved_context.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No indexed context matched yet.</p>
          )}
        </div>
      </div>
    </section>
  );
}
