# SirRobin project resumption decision

**Decision date:** 2026-08-09  
**Decision:** Resume this original SirRobin repository as the active
implementation base.

The successor SirRobin Living scientific-recovery effort improved bounded test,
checkpoint, accounting, and provenance evidence, but was halted after live use
showed that its simulation, durable observability, transport, and viewer cadence
were coupled too tightly for practical evolutionary-scale operation. That
repository is now historical evidence and a selective donor rather than the
foundation for continued implementation.

This branch starts from the reviewed Tranche A implementation at `3e007af`. The
pre-existing dirty `recovery/living-loop` checkout remains preserved separately;
no reset, cleanup, or history rewrite is authorized by this decision.

The first milestone is deliberately operational:

1. run the existing headless simulation and viewer;
2. measure useful whole-world simulation throughput;
3. identify the minimum missing vertical-loop mechanisms; and
4. make only the minimum repairs needed to regain a coherent runnable loop.

Later recovery tests or accounting mechanisms may be imported individually when
they address a demonstrated defect. The successor recovery process and its
governance machinery are not inherited wholesale.
