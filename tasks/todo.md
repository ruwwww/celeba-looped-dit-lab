# Tasks Checklist

- [ ] Implement `models/looped_dit.py` (LoopedDiT ~50M param, XSA, (pre=2, core=4, post=2), deep supervision exits)
- [ ] Implement `models/flow.py` (OT Flow matching with deep supervision loss and Euler ODE sampler)
- [ ] Implement `models/__init__.py`
- [ ] Implement `tests/test_looped_dit.py` and pass all unit tests
- [ ] Implement `train.py` using `ExperimentTracker`
- [ ] Implement `sample.py` with multi-loop inference and CFG
