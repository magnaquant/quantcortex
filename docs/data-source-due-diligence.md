# Data-Source Due Diligence

This document records research-data decisions; it is not legal advice. Before a
new empirical panel becomes publication evidence, record the provider, dataset,
contracting party, permitted uses, redistribution terms, reviewer-access path,
retrieval timestamp, adjustment method, symbol mapping, and content digest.

## Current Historical Case

The fixed 2018-2025 case was computed from an owner-supplied Yahoo Finance
adjusted-close matrix retrieved through yfinance. The raw matrix is ignored and
not distributed. The repository publishes derived aggregates and records the
input SHA-256, while explicitly stating that provider authorization for public
publication has not been independently verified.

This case is retained as historical negative evidence. It is not an acceptable
source for a new confirmatory panel unless the author documents an applicable
permission basis. The open code and conformance fixtures reproduce software
semantics, not the unavailable observations.

## Acceptance Criteria for New Panels

A panel may enter a preregistered evaluation only when all of the following are
documented before results are inspected:

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
basis is ambiguous, exclude the panel from publication evidence and report the
scope reduction.
