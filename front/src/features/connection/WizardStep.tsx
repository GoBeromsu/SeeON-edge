import { ReactNode, type Ref } from 'react';
import { statusBadgeClassName } from '@/shared/ui/StatusBadge';
import type { WizardStepNumber } from '@/features/connection/wizardSteps';

type Props = {
  readonly index: WizardStepNumber;
  readonly title: string;
  readonly description: string;
  readonly complete: boolean;
  readonly locked: boolean;
  readonly lockReason: string;
  /** True when this is the step the operator should continue on right now (deriveActiveStep).
   * Purely a visual cue for where to resume -- it plays no role in the actual gating logic. */
  readonly active: boolean;
  readonly focusRef?: Ref<HTMLElement>;
  readonly children: ReactNode;
};

/** One gated step in the Edge setup wizard: numbered header, completion badge, and a locked
 * placeholder (with an explicit reason, never a bare disabled control) in place of the body. */
export function WizardStep({ index, title, description, complete, locked, lockReason, active, focusRef, children }: Props): JSX.Element {
  const titleId = `wizard-step-${index}-title`;
  return (
    <section
      ref={focusRef}
      tabIndex={-1}
      className={`rounded-card border bg-card p-5 ${active ? 'border-primary/40' : 'border-border'}`}
      aria-labelledby={titleId}
      aria-current={active ? 'step' : undefined}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span
            aria-hidden="true"
            className={`flex h-6 w-6 flex-none items-center justify-center rounded-full text-xs font-semibold ${complete ? 'bg-status-approvedBg text-status-approvedFg' : 'bg-muted text-muted-foreground'}`}
          >
            {index}
          </span>
          <div>
            <h2 id={titleId} className="text-base font-semibold text-foreground">{index}단계 · {title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
        {complete ? <span className={statusBadgeClassName('approved')}>완료</span> : null}
      </div>

      {locked ? (
        <p role="status" className="mt-4 rounded-control bg-muted p-3 text-sm text-muted-foreground">{lockReason}</p>
      ) : (
        <div className="mt-4">{children}</div>
      )}
    </section>
  );
}
