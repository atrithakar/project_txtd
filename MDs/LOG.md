# Technical Thought Log

### Decision: Didn't publish a CUDA implementation to convert CSV into TXTD and then back into CSV
Reason: The CUDA implementation to convert CSV into TXTD and then back into CSV wasn't giving much performance boost against the Single-threaded CPU implementation and had some minor bugs too. So it was decided not to publish it now and keep it a low priority fix for the future.

The CPU implementation encoded a ~70MB CSV in 1.82 seconds and the CUDA version encoded the same file in 1.74 seconds. The performance boost is almost negligible.

### Decision: No bug fixes for [Issue 1](https://github.com/atrithakar/project_txtd/issues/1)
Reason: Currently it is a low priority fix as it is present for only one type of CSV file format and users can be prompted to save their original CSVs in different CSV file format to get the best results. Even in the CSV format in which the bug appears, after one full encoding-decoding cycle, the contents of decoding CSV are identical to that of original CSV from a normal human POV but the decoded file increases by approximate 5.5% in size compared to the original file.


> More logs will be added later as the project grows