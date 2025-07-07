# What is `smart_decoder.c`?
**Ans:** `smart_decoder.c` is a utility tool for the TXTD format that automatically infers the original format (CSV or plain TXT) of a `.txtd` file and applies the appropriate decoding logic. Rather than requiring the user to remember or guess the source format, it reads the file’s internal preamble to determine the correct decoding path.

It is currently categorized as a utility because it prioritizes convenience over performance or precision — ideal for quick recoveries or when the file origin is unclear. This tool was developed to enhance user experience by eliminating trial-and-error decoding when the original format is forgotten.

# How to compile?
```
gcc smart_decoder.c -o smart_decoder
```
