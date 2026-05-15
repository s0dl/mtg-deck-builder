import { AlertTriangle, BrainCircuit, CheckCircle2 } from "lucide-react";

import type { DeckResponse } from "../lib/api";

type DeckResultProps = {
  deck: DeckResponse;
};

export function DeckResult({ deck }: DeckResultProps) {
  const totalCards = deck.cards.reduce((sum, card) => sum + card.count, 0);
  const totalPrice = deck.cards.reduce((sum, card) => sum + (card.estimated_price_usd ?? 0) * card.count, 0);
  const hasPrice = deck.cards.some((card) => card.estimated_price_usd != null);

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

      <div className="summary-strip">
        <div className="summary-item">
          <span>Mode</span>
          <strong><BrainCircuit size={16} /> {deck.generation_mode}</strong>
        </div>
        <div className="summary-item">
          <span>Total</span>
          <strong>{totalCards} cards</strong>
        </div>
        <div className="summary-item">
          <span>Context</span>
          <strong>{deck.retrieved_context.length} refs</strong>
        </div>
        <div className="summary-item">
          <span>Estimate</span>
          <strong>{hasPrice ? `$${totalPrice.toFixed(2)}` : "no prices"}</strong>
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
        <div className="deck-section">
          <div className="section-heading">
            <h3>Main Deck</h3>
            <span>{deck.cards.length} unique · {totalCards} total</span>
          </div>
          <div className="card-list" aria-label="Main deck cards">
            {deck.cards.map((card) => (
              <div className="card-row" key={card.name}>
                <div>
                  <strong>{card.count}x {card.name}</strong>
                  <span>{card.role}</span>
                </div>
                {card.estimated_price_usd != null && (
                  <em>${(card.estimated_price_usd * card.count).toFixed(2)}</em>
                )}
              </div>
            ))}
          </div>
        </div>

        <aside className="inspector" aria-label="Agent workflow and deck context">
          <section className="inspector-section">
            <div className="section-heading">
              <h3>Agent Workflow</h3>
              <span>{deck.agent_steps.length} steps</span>
            </div>
            <div className="workflow-strip" aria-label="Expected agent workflow">
              <span>Initial RAG</span>
              <span>GPT plan</span>
              <span>RAG tools</span>
              <span>Validation</span>
              <span>Scryfall prices</span>
            </div>
            {deck.agent_steps.length > 0 ? (
              <div className="agent-steps">
                {deck.agent_steps.map((step, index) => (
                  <div className="agent-step" key={`${step.label}-${index}`}>
                    <span className="step-index">{index + 1}</span>
                    <div>
                      <strong>{step.label}</strong>
                      <span>{step.detail}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">No model or tool activity reported.</p>
            )}
          </section>

          <section className="inspector-section">
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
          </section>

          <section className="inspector-section">
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
          </section>
        </aside>
      </div>
    </section>
  );
}
