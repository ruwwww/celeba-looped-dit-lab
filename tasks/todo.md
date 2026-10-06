# Tasks Checklist

- [x] Implement `models/looped_dit.py` (LoopedDiT ~50M param, XSA, (pre=2, core=4, post=2), deep supervision exits)
- [x] Implement `models/flow.py` (OT Flow matching with deep supervision loss and Euler ODE sampler)
- [x] Implement `models/__init__.py`
- [x] Implement `tests/test_looped_dit.py` and pass all unit tests
- [x] Implement `train.py` using `ExperimentTracker`
- [x] Implement `sample.py` with multi-loop inference and CFG
