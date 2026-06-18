# Data-Source Due Diligence

This document records research-data decisions; it is not legal advice. Before a
new empirical panel becomes publication evidence, record the provider, dataset,
contracting party, permitted uses, redistribution terms, reviewer-access path,
retrieval timestamp, adjustment method, symbol mapping, and content digest.

## Current Historical Evidence

The fixed 2018-2025 case and the two-panel expansion were computed from
owner-supplied Yahoo Finance adjusted-close matrices retrieved through
yfinance. The raw matrices are ignored and not distributed. The repository
publishes derived aggregates and records each input SHA-256, while explicitly
stating that provider authorization for public publication has not been
independently verified.

The first case is retained as historical negative evidence. The expansion was
repository-frozen before retrieval but was not externally registered and is not
a temporal holdout. Neither is represented as reviewer-reproducible empirical
evidence. The open code and conformance fixtures reproduce software semantics,
not the unavailable observations.

## Acceptance Criteria for New Panels

A panel may be represented as publication-ready, independently reproducible
empirical evidence only when all of the following are documented before results
are inspected:

1. Lawful research use and publication of derived results are permitted.
2. Adjustment, calendar, timestamp, and survivorship policies are explicit.
3. Reviewer access is described honestly; redistribution and access are
   separate questions.
4. A frozen local input is content-addressed and retained under the applicable
   terms.
5. The paper checklist answers are evaluated item by item rather than inferred
   from a generic license label.

Public-domain macro or rates series can support an open real-data panel, but
constructed asset returns introduce modeling assumptions and cannot substitute
for quoted equity prices without explicit justification. Paid commercial data
may permit research use while still forbidding redistribution. Synthetic data
remains appropriate for contract and regression tests only, never headline
performance.

## Decision Log

For each candidate source, create a dated record under ignored local research
notes before acquisition. Do not add provider files to Git. If the permission
basis is ambiguous, disclose that limitation and do not claim the panel is open,
independently reproducible, or submission-ready.
