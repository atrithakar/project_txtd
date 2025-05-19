# txtd

**txtd** is a command-line utility suite designed specifically for **numerical data text files** that contain only symbols defined in `mappings.txt`. It encodes `.txt` files into a compact nibble-based format `.txtd` and decodes them back perfectly, ensuring **lossless** conversion.

This tool can save **up to 50%** storage space by efficiently compressing numerical and symbol data based on  custom nibble mappings.

Additionally, a file comparison utility verifies the correctness of encoding and decoding by comparing SHA-256 hashes of files.

---

## Core Functionality

- **Lossless encoding/decoding:** Original numerical text files are restored exactly from the encoded files with zero data loss.
- **Storage savings:** Saves **up to 50%** space by encoding only the allowed symbols in `mappings.txt` into nibble sequences.
- **Limited symbol set:** Works only with numerical data and symbols listed in `mappings.txt`.
- **File verification:** Uses SHA-256 hashing to compare files and confirm exact matches.
- **Command-line utilities:** Easy-to-use CLI tools for encoding, decoding, and file comparison.

---

## Getting Started

### Prerequisites

- MinGW or any GCC-compatible compiler  
- OpenSSL libraries (`libssl` and `libcrypto`) for SHA-256 support

### Build

```
gcc -o encode encoder.c
gcc -o decode decoder.c
gcc -o compare_files compare_files.c -lssl -lcrypto
```

*Adjust include/library paths if necessary.*

---

## Usage

### Encode a TXT file (numerical data with allowed symbols)

```
./encode input.txt output.txtd
```

### Decode a TXTD file

```
./decode input.txtd output.txt
```

### Compare two files

```
./compare_files file1 file2
```

Output on success:

Files are IDENTICAL

Output on mismatch:

Files are DIFFERENT

---

## Storage Savings Example

| File                | Original Size (bytes) | Encoded Size (bytes) | Savings     |
|---------------------|----------------------|---------------------|-------------|
| pi_digits.txt       | 300,000,001          | 150,000,001         | ~50% smaller |


*Savings depend on input content matching symbols in `mappings.txt`.*

---

## Project Structure

```
.
├── encoder.c          # Encoder source code
├── decoder.c          # Decoder source code
├── compare_files.c   # File comparison utility
├── mappings.txt      # Nibble-to-symbol mapping rules
├── README.md         # This file
├── .gitignore
└── ...
```

---

## License

MIT License — see the LICENSE file.

---

## Author

Atri Thakar  
[GitHub](https://github.com/atri-thakar)

---

Feel free to open issues, request features, or contribute!
