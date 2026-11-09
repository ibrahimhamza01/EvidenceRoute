# Labeling Log

Ambiguous questions and labeling decisions for the routing dataset.

| Date | Questions | Issue | Decision |
| --- | --- | --- | --- |
| 2026-11-09 | q0004, q0014, q0015, q0019 | Each asks about a change from the prior year. The rule "more than one fiscal year means multi_source" conflicted with the fact that each filing already reports the prior-year figure. | Labeled simple. Added an edge case to LABELING.md: a prior-year comparison reported inside one filing is simple; if the older filing itself is needed, it is multi_source. |
