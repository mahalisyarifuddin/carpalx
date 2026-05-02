## 2026-04-25 - [Delta-based Optimization in Partial Mode]
**Mode:** Bolt
**Learning:** The "Partial" optimization mode was previously performing a full O(T) effort recalculation (where T is the number of triads) for every candidate swap in an O(N^2) search space (where N is the number of relocatable keys). By utilizing the existing delta-based effort calculation (which is O(A) where A is the number of affected triads, typically ~50), the complexity per swap is dramatically reduced.
**Action:** Always check if a full re-calculation can be replaced by a delta-based update in iterative optimization algorithms. Ensure that state-changing delta functions are properly balanced by the caller.

## 2026-05-02 - [Hot Path Optimization in Triad Effort Calculation]
**Mode:** Bolt
**Learning:** In both JavaScript and Python, the triad effort calculation is a critical hot path. In JavaScript, using `Array.sort()` to find the maximum row difference and using string-concatenated keys for `pathCosts` lookups were major bottlenecks. Replacing them with direct logical comparisons and `Float64Array` with numeric indexing (`h*64 + r*8 + f`) yielded a ~5.7x speedup. In Python, switching from dictionary lookups with tuple keys to a flat list with the same numeric indexing scheme provided a ~20% performance improvement.
**Action:** Avoid generic utility functions (like sort) and non-primitive lookups in hot loops. Use typed arrays or flat lists with pre-calculated numeric indices for multi-dimensional data access.
