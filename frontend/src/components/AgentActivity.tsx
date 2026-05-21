import { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  Coins,
  Compass,
  Library,
  Loader2,
  Search,
  ShieldCheck,
  Sparkles,
  Wand2
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import type { AgentStep, DeckRequest } from "../lib/api";

type ActivityMode = "immersive" | "compact";
type ActivityStatus = "complete" | "active" | "pending";

type ActivityPhase = {
  durationMs: number;
  label: string;
  detail: string;
  Icon: LucideIcon;
};

type ActivityItem = {
  label: string;
  detail: string;
  Icon: LucideIcon;
  status: ActivityStatus;
};

type AgentActivityProps = {
  generationMode?: string;
  isLive?: boolean;
  mode?: ActivityMode;
  request?: DeckRequest | null;
  steps?: AgentStep[];
};

const liveProgressCap = 96;

function colorsLabel(colors: string[] | undefined) {
  return colors && colors.length > 0 ? colors.join("/") : "any color";
}

function requestLabel(request: DeckRequest | null | undefined) {
  if (!request) {
    return "deck request";
  }

  const playstyle = request.playstyle.trim() || "custom";
  return `${request.format} ${colorsLabel(request.colors)} ${playstyle}`;
}

function livePhases(request: DeckRequest | null | undefined): ActivityPhase[] {
  const format = request?.format ?? "selected format";
  const colors = colorsLabel(request?.colors);
  const playstyle = request?.playstyle.trim() || "requested playstyle";

  return [
    {
      durationMs: 3000,
      label: "Reading request",
      detail: `Locking ${format} constraints for ${colors} ${playstyle}.`,
      Icon: Compass
    },
    {
      durationMs: 9000,
      label: "Retrieving strategy",
      detail: "Searching local RAG strategy, meta, and archetype context.",
      Icon: Library
    },
    {
      durationMs: 6000,
      label: "Checking rules",
      detail: "Pulling format rules and deck construction constraints.",
      Icon: Search
    },
    {
      durationMs: 18000,
      label: "Planning tool calls",
      detail: "Preparing corpus-backed card searches and inclusion checks.",
      Icon: BrainCircuit
    },
    {
      durationMs: 30000,
      label: "Selecting cards",
      detail: "Balancing threats, interaction, card flow, and mana sources.",
      Icon: Wand2
    },
    {
      durationMs: 6000,
      label: "Validating deck",
      detail: "Running deterministic count, legality, and deck size checks.",
      Icon: ShieldCheck
    },
    {
      durationMs: 4000,
      label: "Finalizing response",
      detail: "Hydrating prices when available and packaging final context notes.",
      Icon: Coins
    }
  ];
}

const compactIcons = [Library, BrainCircuit, Search, ShieldCheck, Coins, Sparkles];

function liveTiming(phases: ActivityPhase[], elapsedMs: number) {
  const totalDurationMs = phases.reduce((sum, phase) => sum + phase.durationMs, 0);
  let elapsedBeforePhase = 0;

  for (let index = 0; index < phases.length; index += 1) {
    const phase = phases[index];
    const elapsedThroughPhase = elapsedBeforePhase + phase.durationMs;
    if (elapsedMs < elapsedThroughPhase) {
      return {
        activeIndex: index,
        isPastEstimate: false,
        progress: Math.min((elapsedMs / totalDurationMs) * liveProgressCap, liveProgressCap)
      };
    }
    elapsedBeforePhase = elapsedThroughPhase;
  }

  return {
    activeIndex: phases.length - 1,
    isPastEstimate: true,
    progress: liveProgressCap
  };
}

export function AgentActivity({
  generationMode,
  isLive = false,
  mode = "compact",
  request,
  steps = []
}: AgentActivityProps) {
  const [elapsedMs, setElapsedMs] = useState(0);
  const phases = useMemo(() => livePhases(request), [request]);

  useEffect(() => {
    if (!isLive) {
      setElapsedMs(0);
      return undefined;
    }

    const startedAt = window.performance.now();
    setElapsedMs(0);
    const timer = window.setInterval(() => {
      setElapsedMs(window.performance.now() - startedAt);
    }, 500);

    return () => window.clearInterval(timer);
  }, [isLive, request]);

  const timing = liveTiming(phases, elapsedMs);
  const activeIndex = timing.activeIndex;

  const items: ActivityItem[] = isLive
    ? phases.map((phase, index) => ({
        ...phase,
        status: index < activeIndex ? "complete" : index === activeIndex ? "active" : "pending"
      }))
    : steps.map((step, index) => ({
        label: step.label,
        detail: step.detail,
        Icon: compactIcons[index % compactIcons.length],
        status: "complete"
      }));

  const fallbackItem: ActivityItem = {
    label: "No activity reported",
    detail: "The backend returned no model or tool workflow entries for this run.",
    Icon: BrainCircuit,
    status: "pending"
  };
  const stillWorkingItem: ActivityItem = {
    label: "Still working",
    detail: "The backend is still running retrieval, model, and tool work. Final reported steps will appear with the deck.",
    Icon: BrainCircuit,
    status: "active"
  };
  const displayItems = items.length > 0 ? items : [fallbackItem];
  const primary = isLive && timing.isPastEstimate ? stillWorkingItem : isLive ? displayItems[activeIndex] : displayItems[displayItems.length - 1];
  const isCompactComplete = !isLive && mode === "compact";
  const PrimaryIcon = isCompactComplete ? BrainCircuit : primary.Icon;
  const elapsedSeconds = Math.max(1, Math.floor(elapsedMs / 1000));
  const progress = isLive ? timing.progress : steps.length > 0 ? 100 : 0;

  return (
    <div
      className={`agent-activity ${mode} ${isLive ? "live" : "complete"}`}
    >
      <div className="activity-main">
        <div className="activity-mark" aria-hidden="true">
          {isLive ? <Loader2 className="activity-spinner" size={28} /> : <PrimaryIcon size={24} />}
        </div>

        <div className="activity-copy" aria-live={isLive ? "polite" : undefined}>
          {!isCompactComplete && <p className="eyebrow">{isLive ? "Live agent build" : "Agent workflow"}</p>}
          {mode === "immersive" ? <h2>{primary.label}</h2> : <h3>{isCompactComplete ? "Agent workflow" : primary.label}</h3>}
          {!isCompactComplete && <p>{primary.detail}</p>}
        </div>

        <div className="activity-state">
          <span className={isLive ? "activity-badge running" : "activity-badge done"}>
            {isLive ? `${elapsedSeconds}s` : generationMode ?? "complete"}
          </span>
        </div>
      </div>

      {mode === "immersive" && request && (
        <div className="activity-request" aria-label="Current deck request">
          <div>
            <span>Request</span>
            <strong>{requestLabel(request)}</strong>
          </div>
          <div>
            <span>Budget</span>
            <strong>{request.budget_usd == null ? "open" : `$${request.budget_usd}`}</strong>
          </div>
          <div>
            <span>Must include</span>
            <strong>{request.must_include.length > 0 ? request.must_include.join(", ") : "none"}</strong>
          </div>
          <div>
            <span>Avoid</span>
            <strong>{request.avoid.length > 0 ? request.avoid.join(", ") : "none"}</strong>
          </div>
        </div>
      )}

      <div className="activity-progress" aria-hidden="true">
        <div className="activity-progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="activity-timeline" role="list">
        {displayItems.map((item, index) => {
          const Icon = item.Icon;
          return (
            <div className={`activity-step ${item.status}`} key={`${item.label}-${index}`} role="listitem">
              <span className="activity-step-icon" aria-hidden="true">
                {item.status === "complete" ? (
                  <CheckCircle2 size={16} />
                ) : item.status === "active" ? (
                  <Loader2 className="activity-spinner" size={16} />
                ) : (
                  <Icon size={16} />
                )}
              </span>
              <div>
                <strong>{item.label}</strong>
                <span>{item.detail}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
