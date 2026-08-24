# Side finding: raw gene IDs used as entity names in the generation prompts

Measured over all 35,658 relation claims scored on 2026-08-24.

- **2,622 claims (7.4%)** name at least one entity by a bare
  numeric identifier rather than a text name.
- **1,199** of those (45.7%) have the raw identifier written
  literally into the generated abstract.
- Papers affected: **49** of 592.

## Does the QC model reject them by itself?

Share of claims the QC model calls supported, at cut-off 0.5:

| claim group | claims | called supported | mean probability |
|---|---|---|---|
| normal entity names | 33,036 | 66.0% | 0.656 |
| numeric id as a name | 2,622 | 9.9% | 0.102 |
|   of those, id written into the abstract | 1,199 | 20.9% | 0.216 |
|   of those, id not in the abstract | 1,423 | 0.6% | 0.006 |

The gap is **+56.2 points** of acceptance rate between normal claims and claims whose entity arrived as a bare identifier.

## How to read that, in both directions

The flattering reading is that the filter removes these almost entirely without having been designed to, which is evidence it keys on something real rather than rejecting at random.

The unflattering reading matters more for the report. Part of what the filter is catching here is a **data defect, not a failure of the generator to express a relation**. An entity called `54624` has no readable mention to support, so the claim is unsupported for a trivial reason. Those claims should not be counted as evidence that the QC model detects subtle unsupported relations, and if the identifier bug is fixed upstream the filter's measured value will drop a little.

Either way Fred should hear it: the entity-name lookup falls back to the identifier for some entities, and the generator faithfully writes the number into the abstract. It affects the unfiltered synthetic training data that BERT 1 has already been trained on.
