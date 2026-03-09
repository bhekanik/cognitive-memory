/**
 * Cognitive Memory System - Decay Calculations
 *
 * Implements Ebbinghaus forgetting curve with retention floors
 * and spaced repetition mechanics.
 */

import type {
  DecayParameters,
  MemoryCategory,
  ResolvedCognitiveMemoryConfig,
} from "./types";
import { DEFAULT_CONFIG, getBaseDecayRate, getRetentionFloor } from "./types";

/**
 * Base decay rates (in days) for different memory categories
 * Matches Python SDK Table 2
 */
export const BASE_DECAY_RATES: Record<MemoryCategory, number> = {
  episodic: 45,
  semantic: 120,
  procedural: Number.POSITIVE_INFINITY,
  core: 120,
};

/**
 * Decay floors by category type
 */
export const DECAY_FLOORS = {
  core: 0.60,
  regular: 0.02,
} as const;

function assertUnitInterval(field: string, value: number) {
  if (Number.isNaN(value) || value < 0 || value > 1) {
    throw new Error(`Invalid ${field}: ${value} (must be [0.0, 1.0])`);
  }
}

/**
 * Calculate current retention level (0.0-1.0) for a memory
 *
 * Equation 1 from the paper:
 * R(m) = max(floor, exp(-dt / (S * B * beta_c)))
 *
 * Where:
 * - dt = days since last access
 * - S = stability (0.0-1.0, grows with retrievals)
 * - B = importance boost = 1 + (importance * 2), capped at 3.0
 * - beta_c = category-specific base decay rate
 * - floor = 0.60 for core, 0.02 for regular
 */
export function calculateRetention(
  params: DecayParameters,
  config?: Partial<ResolvedCognitiveMemoryConfig>,
): number {
  const { stability, importance, lastAccessed } = params;
  const category = params.category ?? (params.memoryType as MemoryCategory) ?? "semantic";

  assertUnitInterval("stability", stability);
  assertUnitInterval("importance", importance);

  const rates = config?.decayRates ?? DEFAULT_CONFIG.decayRates;
  const baseDecay = getBaseDecayRate(category, rates);

  // Procedural memories never decay
  if (category === "procedural" || baseDecay === Number.POSITIVE_INFINITY) {
    return 1.0;
  }

  // Get retention floor
  const floor = getRetentionFloor(category, {
    coreRetentionFloor: config?.coreRetentionFloor ?? DEFAULT_CONFIG.coreRetentionFloor,
    regularRetentionFloor: config?.regularRetentionFloor ?? DEFAULT_CONFIG.regularRetentionFloor,
  });

  // Calculate days since last access
  const now = Date.now();
  const daysSinceAccess = Math.max(
    0,
    (now - lastAccessed) / (1000 * 60 * 60 * 24),
  );

  // Importance boosts decay resistance (multiplier: 1.0 to 3.0)
  const importanceBoost = Math.min(3.0, 1.0 + importance * 2.0);

  // Stability (avoid division by zero)
  const S = Math.max(stability, 0.01);

  // Combined decay constant
  const effectiveRate = S * importanceBoost * baseDecay;

  // Decay calculation (exponential or power-law)
  const decayModel = config?.decayModel ?? DEFAULT_CONFIG.decayModel;
  const raw = decayModel === "power"
    ? (1 + daysSinceAccess / effectiveRate) ** (-(config?.powerDecayGamma ?? DEFAULT_CONFIG.powerDecayGamma))
    : Math.exp(-daysSinceAccess / effectiveRate);

  // Apply floor
  return Math.max(floor, Math.min(1, raw));
}

/**
 * Update stability after a retrieval (spaced repetition)
 *
 * Formula:
 * new_stability = min(1.0, old_stability + boost * spacing_factor)
 * spacing_factor = min(maxMultiplier, days_since_last_access / intervalDays)
 */
export function updateStability(
  currentStability: number,
  daysSinceLastAccess: number,
  boost: number = 0.1,
  maxMultiplier: number = 2.0,
  intervalDays: number = 7.0,
): number {
  assertUnitInterval("stability", currentStability);

  const days = Math.max(0, daysSinceLastAccess);

  // Calculate spacing factor
  const spacingFactor = Math.min(maxMultiplier, days / intervalDays);

  // Stability increase
  const stabilityIncrease = boost * spacingFactor;

  // Cap at 1.0
  return Math.min(1.0, currentStability + stabilityIncrease);
}
