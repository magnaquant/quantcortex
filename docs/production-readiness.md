# Production Readiness

The repository is a research and guarded paper-execution platform. Passing CI
does not certify it for production capital. Complete and independently review
the controls below before any real-money deployment.

## Broker Certification

- Alpaca uses `alpaca-py`; Interactive Brokers uses `ib_async`. CI constructs
  requests with the installed SDK models, and offline mocks cover request and
  response mapping.
- Run authenticated paper tests for permissions, reconnects, rejects, partial
  fills, cancellations, duplicate submissions, and venue-side idempotency.
- Reconcile broker positions, cash, orders, and fills before every trading cycle.

## State and Recovery

- Positions, known orders, submission intents, and reconciliation metadata are
  persisted as one versioned snapshot with optimistic concurrency.
- Submission intent is durably marked `ATTEMPTING` before a broker call. An
  uncertain outcome blocks automatic retry until broker reconciliation.
- Add backup restoration, corruption, concurrent-writer, and disaster-recovery
  drills for Redis and the local file backend. File concurrency is tested;
  Redis transaction behavior still needs a real-service integration test.

## Research and Data

- Use data with an explicit license or permission basis and retained provenance.
- Obtain exact filing timestamps, point-in-time membership, and delisted-security
  prices for production single-name research.
- Validate corporate actions, calendars, stale-price policy, and symbol mapping
  against the intended venues.
- Replace flat slippage with calibrated spread, volatility, size, and capacity
  models when expected order size makes market impact material.

## Deployment Controls

- Dependencies, Python exports, and container images are locked. CI rejects
  stale dependency exports.
- Run container, database, Redis, and broker integration tests in a staging
  environment matching production.
- The application container runs as a non-root user with a read-only root
  filesystem. Add managed secrets, least privilege, structured logs, metrics,
  alerting, and immutable audit records before production deployment.
- Define kill switches, exposure limits, incident ownership, rollback steps,
  and independent release approval.

## Release Evidence

Archive the exact code revision, environment, data digest, configuration,
research trial count, validation report, paper-certification evidence, and
sign-off for each release. Treat unresolved checklist items as release blockers,
not documentation caveats.
