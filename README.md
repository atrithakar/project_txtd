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
- NVCC compiler (for GPU-accelerated implementation)
- **For Windows/CUDA:**  
  - [OpenSSL precompiled binaries](https://slproweb.com/products/Win32OpenSSL.html) (Win64 "Full" version recommended)
  - Add `C:\OpenSSL-Win64\bin` to your system `PATH` for runtime DLLs

### Build

#### Standard (CPU) tools

```
gcc -o encode encoder.c
gcc -o decode decoder.c
gcc -o compare_files compare_files.c -lssl -lcrypto
```

#### GPU-accelerated tools (CUDA)

**On Windows (with OpenSSL in C:\OpenSSL-Win64):**
```
nvcc -I"C:\OpenSSL-Win64\include" -L"C:\OpenSSL-Win64\lib\VC\x64\MD" txt_to_txtd_cuda.cu -o txt_to_txtd_cuda.exe -llibssl -llibcrypto
nvcc -I"C:\OpenSSL-Win64\include" -L"C:\OpenSSL-Win64\lib\VC\x64\MD" txtd_to_txt_cuda.cu -o txtd_to_txt_cuda.exe -llibssl -llibcrypto
```

*Adjust include/library paths if necessary.*

---

## Usage

### Encode a TXT file (CPU)
```
./txt_to_txtd.exe input.txt
```

### Encode a TXT file (GPU/CUDA)
```
./txt_to_txtd_cuda.exe input.txt
```
- Output: Creates a folder named after the input file (without extension), containing `.encoded.txtd` and `.checksum.txt`.

### Decode a TXTD file into TXT file (CPU)
```
./txtd_to_txt.exe folder_name
```

### Decode a TXTD file into TXT file (GPU/CUDA)
```
./txtd_to_txt_cuda.exe folder_name
```
- Looks for `folder_name/folder_name.encoded.txtd`, decodes to `folder_name/folder_name_decoded.txt`, and verifies checksum with `folder_name/folder_name.checksum.txt`.

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

## Performance Comparison: CUDA implementation vs Single-threaded CPU implementation

Below are timing results for encoding and decoding using both the CUDA-accelerated implementation and the single-threaded CPU implementation on the same file (`pi_digits.txt`, **286MB**):

| Step                        | CUDA Pipeline Time | CPU Pipeline Time |
|-----------------------------|-------------------|-------------------|
| Encode                      | 0.98s             | 9.86s             | 
| Decode                      | 5.78s             | 11.22s            |

**Summary:**
- **CUDA implementation** is significantly faster for encoding and decoding large files.
- **File integrity** is verified in both implementations using SHA-256 hashes.

*NOTE: The timings may change from system to system and based on many other variables like ambient temperature too. This comparison is here just to give a rough idea about performance boost in CUDA implementation against the Single-Threaded CPU implementation*

---

## Project Structure

```
.
├── txtd-st/
│   ├── encoder.c             # Encoder source code
│   ├── decoder.c             # Decoder source code
│   ├── compare_files.c       # File comparison
│   ├── txt_to_txtd.c         # TXT to TXTD encoder
│   ├── txtd_to_txt.c         # TXTD to TXT decoder
│   ├── csv_to_txtd.c         # CSV to TXTD encoder
│   └── txtd_to_csv.c         # TXTD to CSV decoder
├── txtd-cuda/
│   ├── txt_to_txtd_cuda.cu   # TXT to TXTD encoder (GPU Accelerated)
│   └── txtd_to_txt_cuda.cu   # TXTD to TXT decoder (GPU Accelerated)
├── mappings.txt              # Nibble-to-symbol mapping rules
├── README.md                 # This file
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
