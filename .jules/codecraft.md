## 2026-04-25 - [Delta-based Optimization in Partial Mode]
**Mode:** Bolt
**Learning:** The "Partial" optimization mode was previously performing a full O(T) effort recalculation (where T is the number of triads) for every candidate swap in an O(N^2) search space (where N is the number of relocatable keys). By utilizing the existing delta-based effort calculation (which is O(A) where A is the number of affected triads, typically ~50), the complexity per swap is dramatically reduced.
**Action:** Always check if a full re-calculation can be replaced by a delta-based update in iterative optimization algorithms. Ensure that state-changing delta functions are properly balanced by the caller.
