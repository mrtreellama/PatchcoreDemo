export type Verdict = 'PASS' | 'DEFECT';

/**
 * Debounces the per-frame decision so the banner does not flicker:
 * DEFECT after `toDefect` consecutive frames above threshold, back to PASS
 * after `toPass` consecutive frames at or below it.
 */
export class Hysteresis {
  state: Verdict = 'PASS';
  private above = 0;
  private below = 0;

  constructor(private toDefect = 2, private toPass = 3) {}

  update(score: number, threshold: number): Verdict {
    if (score > threshold) {
      this.above++;
      this.below = 0;
      if (this.state === 'PASS' && this.above >= this.toDefect) this.state = 'DEFECT';
    } else {
      this.below++;
      this.above = 0;
      if (this.state === 'DEFECT' && this.below >= this.toPass) this.state = 'PASS';
    }
    return this.state;
  }

  reset(): void {
    this.state = 'PASS';
    this.above = 0;
    this.below = 0;
  }
}
