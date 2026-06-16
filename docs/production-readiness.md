# Production Readiness

The repository is a research and guarded paper-execution platform. Passing CI
does not certify it for production capital. Complete and independently review
the controls below before any real-money deployment.

## Broker Certification

- Migrate Alpaca from `alpaca-trade-api` to `alpaca-py`.
- Migrate Interactive Brokers from archived `ib_insync` to a maintained client.
- Run authenticated paper tests for permissions, reconnects, rejects, partial
  fills, cancellations, duplicate submissions, and venue-side idempotency.
- Reconcile broker positions, cash, orders, and fills before every trading cycle.

## State and Recovery

- Persist positions, orders, submission intents, and reconciliation metadata as
  one recoverable transaction or event stream.
- Define restart behavior for crashes between order submission and persistence.
- Add backup restoration, corruption, concurrent-writer, and disaster-recovery
  tests for every configured storage backend.

## Research and Data

- Use data with an explicit license or permission basis and retained provenance.
- Obtain exact filing timestamps, point-in-time membership, and delisted-security
  prices for production single-name research.
- Validate corporate actions, calendars, stale-price policy, and symbol mapping
  against the intended venues.
- Replace flat slippage with calibrated spread, volatility, size, and capacity
  models when expected order size makes market impact material.

## Deployment Controls

- Pin and lock all dependencies and container base images.
- Run container, database, Redis, and broker integration tests in a staging
  environment matching production.
- Run services as a non-root user with secret management, least privilege,
  structured logs, metrics, alerting, and immutable audit records.
- Define kill switches, exposure limits, incident ownership, rollback steps,
  and independent release approval.

## Release Evidence

Archive the exact code revision, environment, data digest, configuration,
research trial count, validation report, paper-certification evidence, and
sign-off for each release. Treat unresolved checklist items as release blockers,
not documentation caveats.
