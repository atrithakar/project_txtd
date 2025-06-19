# TXTD File Format Specification

## Overview

**TXTD** is a custom nibble-based file format designed to improve the storage efficiency of numerical and symbolic data. It encodes each symbol (digit, delimiter, or whitespace) into a 4-bit representation (nibble), allowing **two symbols to be packed into a single byte**. This results in approximately **50% reduction in file size** for compatible datasets.

The TXTD format is particularly useful for applications dealing with:
- Pure numerical datasets
- Sensor logs and telemetry
- Compressed CSV archives
- Large digit-based computations (e.g., Pi digit dumps)

---

## Features

- 🔒 **Lossless** encoding/decoding with byte-level fidelity  
- 📉 **Up to 50%** reduction in storage for compatible files  
- 🚀 **CUDA-accelerated** implementations available for fast processing on Nvidia GPUs  
- 🧪 **SHA-256 hashing** for checksum verification and round-trip integrity  
- 📁 Supports `.txt` and `.csv` as input formats  

---

## Symbol Mapping

The TXTD format supports only a limited symbol set (defined in `mappings.txt`):

| Symbol | ASCII | Nibble (Hex) | Binary |
|--------|-------|--------------|--------|
| `0`    | 48    | `0x0`         | `0000` |
| `1`    | 49    | `0x1`         | `0001` |
| `2`    | 50    | `0x2`         | `0010` |
| `3`    | 51    | `0x3`         | `0011` |
| `4`    | 52    | `0x4`         | `0100` |
| `5`    | 53    | `0x5`         | `0101` |
| `6`    | 54    | `0x6`         | `0110` |
| `7`    | 55    | `0x7`         | `0111` |
| `8`    | 56    | `0x8`         | `1000` |
| `9`    | 57    | `0x9`         | `1001` |
| `.`    | 46    | `0xA`         | `1010` |
| ` `    | 32    | `0xB`         | `1011` |
| `\t`   | 9     | `0xC`         | `1100` |
| `\n`   | 10    | `0xD`         | `1101` |
| `,`    | 44    | `0xE`         | `1110` |
| EOF    | -     | `0xF`         | `1111` | *(used as padding or terminator)*

---

## File Structure

### Encoded TXTD file (`.txtd`)

- **Header**: None (flat binary format)  
- **Body**:
  - Packed stream of nibbles representing valid symbols
  - 2 symbols per byte: `[upper_nibble][lower_nibble]`
  - If the total number of symbols is odd, the last nibble is padded with `0xF`

### Checksum File (`.checksum.txt`)

- Contains the SHA-256 checksum (in hex) of the original `.txt` or `.csv` file

---

## Encoding Pipeline

```
Input File (TXT/CSV)
        ↓
[Mapping: Char → Nibble]
        ↓
[Packing: 2 Nibbles → 1 Byte]
        ↓
Output File (.txtd)
```

---

## Decoding Pipeline

```
Input File (.txtd)
        ↓
[Unpacking: Byte → 2 Nibbles]
        ↓
[Mapping: Nibble → Char]
        ↓
Output File (TXT/CSV) + SHA-256 Match
```

---

## Example

Given a text:

```
123,456
```

Encoding steps:
- `1` → `0001`
- `2` → `0010`
- `3` → `0011`
- `,` → `1110`
- `4` → `0100`
- `5` → `0101`
- `6` → `0110`
- `\n` → `1101`

Pairs:
- `0001 0010` → `0x12`
- `0011 1110` → `0x3E`
- `0100 0101` → `0x45`
- `0110 1101` → `0x6D`

Encoded bytes: `0x12 0x3E 0x45 0x6D`

---

## Implementation

- ✅ Single-threaded CPU implementation in **C**
- ✅ CUDA-accelerated implementation for Nvidia GPUs
- ✅ Multi-threaded CPU
- 🚧 OpenCL version for iGPUs (planned)

---

## Limitations

- ❌ No Unicode/multibyte character support
- ❌ Input must contain only symbols defined in `mappings.txt`
- ⚠️ Recommended to use **UTF-8 (comma-delimited)** format when saving CSVs

---

## Use Cases

- Compressing large digit logs (e.g., π)
- IoT sensor data archival
- Efficient CSV storage

---

## License

MIT License — Free to use, modify, and distribute.

---

## Author

**Atri Thakar**  
GitHub: [@atrithakar](https://github.com/atrithakar)

---
