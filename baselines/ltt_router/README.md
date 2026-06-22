LTT Router v2

This package generalises the validated two-model proof-of-concept in
``baselines/ltt_v1/`` to the N-model setting, with three design goals:

  1. **Router-agnostic calibration.** The routing function is *injected*
     (a callable satisfying the ``RoutingFunction`` protocol).
     The Learn-Then-Test (LTT) calibration core consumes scores + outcomes and
     is reused unchanged across any router.

  2. **Cost-ordered, Pareto-aware routing.** Expensive does not mean better.
     We Pareto-filter dominated models offline, order survivors by cost, and
     route each query to the cheapest model whose score clears the calibrated
     threshold (falling back to the most capable model otherwise).

  3. **Benchmark-comparable + guarantee metrics.** A thin adaptor lets us stand
     beside the existing LLMRouterBench baselines on their metrics, while we
     additionally report the risk guarantee (alpha, delta, realized risk,
     violation rate) that no other baseline provides.
