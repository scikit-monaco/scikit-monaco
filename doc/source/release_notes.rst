
Release Notes
-------------

Scikit-monaco v0.2 release notes
++++++++++++++++++++++++++++++++

* Add MISER algorith for recursive stratified sampling.

* All integration routines now respond to KeyboardInterrupt.

* Additional benchmarks for importance sampling.

Scikit-monaco v0.1.5 release notes
++++++++++++++++++++++++++++++++++

Version 0.1.5 provides an important bugfix. Seeds were not being properly
generated, such that repeated calls to ``mcquad`` or ``mcimport`` that came
within the same value of ``int(time.time())`` had the same seed. 

Seeding is now left to the (more capable) hands of the RNG, which just gets
passed ``seed(None)`` if the user hasn't specified a seed.
