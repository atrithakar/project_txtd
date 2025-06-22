# What are Data Generators?

**Data Generators** are small C utilities included in the `txtd` project that generate test files of controlled sizes filled with random data. These tools eliminate the need to search for suitable sample datasets manually and provide a standardized, repeatable input source for testing and benchmarking.

---

## Why They Exist

- **Benchmarking**: Measure encoding/decoding performance under consistent conditions.
- **Validation**: Ensure correctness across platforms and hardware.
- **Convenience**: Developers and contributors don't need to gather large files manually.
- **Stress Testing**: Evaluate how `txtd` handles different patterns (dense digits, CSV delimiters, etc).

---

## Included Generators

| File Name                    | Description                                                                 | Output File             | Size     |
|-----------------------------|-----------------------------------------------------------------------------|--------------------------|----------|
| `generate_1gb_random_digits.c` | Generates a 1 GiB file containing only numeric digits (0–9)                | `random_digits.txt`     | 1 GiB    |
| `generate_1gb_csv_data.c`      | Generates a CSV file >1 GiB with structured rows and standard headers      | `random_data.csv`       | ~1.06 GiB|

All files are written in plain UTF-8 encoding and optimized for realistic compression benchmarks.

---

## How to Use

1. **Compile** using your preferred C compiler:
```sh
gcc generate_1gb_random_digits.c -o gen_digits
gcc generate_1gb_csv_data.c -o gen_csv
```

2. **Run** the generators:
```sh
./gen_digits
./gen_csv
```

3. **Use the output** with any of the `txtd` encoders (OpenMP or CUDA), then decode and verify correctness using the checksums.

---

## Customization Ideas

- Change CSV headers or delimiter style
- Adjust total size for smaller or larger test runs
- Use floating-point values or special symbols (mentioned in [mappings.txt](https://github.com/atrithakar/project_txtd/blob/main/mappings.txt) only) to simulate noisy datasets

---

## Purpose in `txtd`

The `Data Generators` suite makes the `txtd` test workflow fully self-contained. Whether you're:

- Optimizing performance
- Stress-testing edge cases
- Validating checksum integrity
- Comparing different backends (ST, MT, CUDA)


