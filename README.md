# txtd

**txtd** is a command-line utility suite designed specifically for **numerical data text files** and **CSV files** that contain only symbols defined in `mappings.txt`. It encodes `.txt` and `.csv` files into a compact nibble-based format `.txtd` and decodes them back perfectly, ensuring **lossless** conversion.

This tool can save **up to 50%** storage space by efficiently encoding numerical and symbol data based on custom nibble mappings.

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
./txt_to_txtd.exe input.txt output.txtd
```

### Decode a TXTD file into TXT file

```
./txtd_to_txt.exe input.txtd output.txt
```

### Encode a CSV file (only numerical data in the body and allowed delimiters)
```
./csv_to_txtd.exe csv_name.csv
```

### Decode a TXTD file into CSV file
```
./txtd_to_csv.exe csv_name
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
| hw_200.csv          | 17,998,823           | 8,475,152           | ~47% smaller |


*Savings depend on input content matching symbols in `mappings.txt`.*<br>
*NOTE: For best results, make sure the original CSV is saved in "CSV UTF-8 (Comma Delimited)" format*

---

## Project Structure

```
.
├── encoder.c          # Encoder source code
├── decoder.c          # Decoder source code
├── compare_files.c    # File comparison 
├── txt_to_txtd.c      # TXT to TXTD encoder source code
├── txtd_to_txt.c      # TXTD to TXT decoder source code
├── csv_to_txtd.c      # CSV to TXTD encoder source code
├── txtd_to_csv.c      # TXTD to CSV decoder source code
├── mappings.txt       # Nibble-to-symbol mapping rules
├── README.md          # This file
├── .gitignore
└── ...
```

---

## License

MIT License — see the LICENSE file.

---

## Author

Atri Thakar  
[GitHub](https://github.com/atrithakar)

---

### Story / Background

txtd started as a personal challenge to build a simple yet efficient tool to encode and decode text files containing mostly numerical data and a limited set of special symbols. I was inspired by the need to save storage space and simplify handling of numerical datasets by encoding them into a custom file format.

The inspiration hit me after I calculated and stored **300 million digits of Pi**, which took up a massive **286MB** of space on disk. Seeing that unexpectedly large file size, I realized there had to be a better way to store this kind of data. That’s when I decided to develop txtd — a tool to encode large numerical data containing the digits and symbols using custom nibble mappings. The encoded Pi digits file ended up taking only **143MB**, cutting the storage requirement by approximately **50%**, which was exactly the result I was aiming for.

By applying nibble-sized mappings defined in `mappings.txt` to the digits and symbols, txtd’s encoder effectively reduces file size by approximately **50%**, making it ideal for encoding large numerical datasets.

The biggest challenge was to ensure the encoder and decoder perfectly restored the original file without losing a single bit, verified by comparing SHA-256 hashes. This also helped me debug and improve the code, including handling deprecated OpenSSL functions during development.

In the end, txtd saves up to **50%** storage space on compatible numerical text and numerical csv files, making it a practical tool for anyone dealing with large numerical datasets or CSV files that fit within the defined symbol set.

This project helped me deepen my understanding of file handling, hashing, and command-line utility design — and gave me a powerful tool for personal and future projects.

---

Feel free to open issues, request features, or contribute!
