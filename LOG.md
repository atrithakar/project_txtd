# Technical Thought Log

### Decision: Didn't publish a CUDA implementation to convert CSV into TXTD and then back into CSV
Reason: The CUDA implementation to convert CSV into TXTD and then back into CSV wasn't giving much performance boost against the Single-threaded CPU implementation and had some minor bugs too. So it was decided not to publish it now and keep it a low priority fix for the future.

The CPU implementation encoded a ~70MB CSV in 1.82 seconds and the CUDA version encoded the same file in 1.74 seconds. The performance boost is almost negligible.


> More logs will be added later as the project grows