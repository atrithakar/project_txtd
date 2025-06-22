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

> **Note**: The **File Verification** feature (based on SHA-256 checksums) is currently included to assist during the development and testing phase. Once the tool has been sufficiently validated for consistent and reliable performance, this feature may be removed to further improve execution speed and reduce resource usage.

---

## Getting Started

### Prerequisites

- MinGW or any GCC-compatible compiler  
- OpenSSL libraries (`libssl` and `libcrypto`) for SHA-256 support  
- NVCC compiler (for GPU-accelerated implementation)
- **For Windows/CUDA:**  
  - [OpenSSL precompiled binaries](https://slproweb.com/products/Win32OpenSSL.html) (Win64 "Full" version recommended)
  - Add `C:\OpenSSL-Win64\bin` to your system `PATH` for runtime DLLs

---

### Build

The instructions to compile the files have been provided [here](https://github.com/atrithakar/project_txtd/blob/main/MDs/BUILD.md).

---

## Usage

The usage instructions have been provided [here](https://github.com/atrithakar/project_txtd/blob/main/MDs/USAGE.md).

---

## Storage Savings Example

| File                | Original Size (bytes) | Encoded Size (bytes) | Savings     |
|---------------------|----------------------|---------------------|-------------|
| pi_digits.txt       | 300,000,001          | 150,000,001         | ~50% smaller |
| hw_200.csv          | 17,998,823           | 8,475,152           | ~47% smaller |


*Savings depend on input content matching symbols in `mappings.txt`.*<br>
*NOTE: For best results, make sure the original CSV is saved in "CSV UTF-8 (Comma Delimited)" format*

---

## Performance Comparison: CUDA vs Multithreaded CPU vs Single-threaded CPU Implementation

Timing results for encoding and decoding `pi_digits.txt` (286MB):

| Step   | CUDA (GPU) | CPU (Multithreaded) | CPU (Single-threaded) |
|--------|------------|----------------------|------------------------|
| Encode | 0.98s      | 3.49s                | 9.86s                  |
| Decode | 5.75s      | 6.06s                | 11.22s                 |


Timing results for encoding and decoding `hw_201.csv` (17.1MB):

| Step   | CPU (Single-threaded) | CPU (Multithreaded) |
|--------|----------------------|------------------------|
| Encode | 0.92s                | 0.54s                  |
| Decode | 1.06s                | 0.39s                  |

*NOTE: The timings may change from system to system and based on many other variables like ambient temperature too. This comparison is here just to give a rough idea about performance comparison among different implementations*

**For more detailed benchmarks, [click here](https://github.com/atrithakar/project_txtd/blob/main/MDs/BENCHMARKS.md).**

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
