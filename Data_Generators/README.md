# What are Data Generators?

**Data Generators** are small C utilities included in the `txtd` project that generate test files of customizable size filled with synthetic random data. These tools eliminate the need to search for suitable sample datasets manually and provide a standardized, repeatable input source for testing and benchmarking.

---

## Why They Exist

- **Benchmarking**: Measure encoding/decoding performance under consistent conditions.
- **Validation**: Ensure correctness across platforms and hardware.
- **Convenience**: Developers and contributors don't need to gather large files manually.
- **Stress Testing**: Evaluate how `txtd` handles different patterns (dense digits, floating-point CSV data, etc).

---

## Included Generators

| File Name                    | Description                                                                 | Output File             |
|-----------------------------|-----------------------------------------------------------------------------|--------------------------|
| `generate_random_digits.c`  | Generates a user-defined sized file containing only numeric digits (0–9)    | `random_digits.txt`     |
| `generate_csv_data.c`       | Generates a CSV file of user-defined size, optionally with fixed rows/cols  | `random_data.csv`       |

All files are written in plain UTF-8 encoding and optimized for realistic compression benchmarks.

---

## How to Use

1. **Compile** the generators using GCC (or any C compiler):
```
gcc generate_random_digits.c -o gen_digits
gcc generate_csv_data.c -o gen_csv
```

2. **Run** the executables from the terminal and follow on-screen prompts.
3. **Use the output** with any of the `txtd` encoders (ST, MT, or CUDA), then decode and verify correctness using the checksums.

---

## Usage: `gen_digits`

Generates a text file (`random_digits.txt`) of your specified size, filled with random digits (0–9), without any separators or formatting.

### Example:
```
$ ./gen_digits.exe
Enter desired file size (e.g., 1MiB, 100KB, 2GiB): 100MiB
Successfully generated 100MiB random digit file: random_digits.txt
```

---

## Usage: `gen_csv`

Generates a CSV file (`random_data.csv`) of your specified size. You can choose how the data is structured.

### Step-by-step:

1. Enter size:
    - Supports units: `KB`, `MB`, `GB`, `KiB`, `MiB`, `GiB`
    - Examples: `1MiB`, `500KB`, `2GiB`

2. Choose mode:
    - `0`: Auto → Fixed 3 columns, rows adjusted to match file size
    - `1`: User defines number of rows → columns adjusted automatically
    - `2`: User defines number of columns → rows adjusted automatically

3. If using mode `1` or `2`, you'll be asked to enter the specific row/column count.

### Example:
```
$ ../gen_csv.exe
Enter desired file size (e.g. 10MiB, 1GB): 100MiB
Select data configuration mode:
  0 - Auto (3 columns, variable rows)
  1 - Fixed number of rows
  2 - Fixed number of columns
Enter choice (0/1/2): 2
Enter number of columns: 67
File has 67 columns and approximately 177964 rows.
CSV file 'random_data.csv' generated successfully. Size: 100.96 MiB

```

---

## Output Format Details

- **CSV Header**: Random lowercase words (1–20 characters), one per column.
- **CSV Cells**: Floating-point numbers like `1.23456`.
- **Encoding**: Plain UTF-8 text, line-separated rows.

---

## Customization Ideas

- Change CSV delimiter (e.g., `;`, `|`, tab-separated)
- Simulate invalid/dirty CSVs for testing validation logic

---

## Purpose in `txtd`

The `Data Generators` suite makes the `txtd` test workflow fully self-contained. Whether you're:

- Optimizing performance
- Stress-testing edge cases
- Validating checksum integrity
- Comparing different backends (ST, MT, CUDA)

you’ll always have controlled and consistent data ready for your testing pipeline.
