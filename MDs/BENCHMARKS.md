# Benchmarks

## 🔧 Benchmark System Info

| Component            | Specification                             |
| -------------------- | ----------------------------------------- |
| **CPU**              | Intel Core i7-13620H (10-core, 16-thread) |
| **GPU**              | NVIDIA RTX 4060 Laptop GPU (8GB GDDR6)    |
| **RAM**              | 16GB DDR5 @ 5200 MT/s, Single Channel     |
| **Storage**          | 512GB Gen 4 NVMe SSD                      |
| **Operating System** | Windows 11                                |

### 💻 Runtime State During Benchmark

| Parameter            | Value                       |
| -------------------- | --------------------------- |
| **Power Source**     | Plugged in (AC Power)       |
| **Background Load**  | Minimal                     |
| **Thermals**         | GPU < 55°C, CPU < 65°C      |
| **Performance Mode** | Turbo / Maximum Performance |
| **Free RAM**         | 6GB out of 16GB available   |

> **Note:** The system specifications and operational state during testing are provided above to ensure complete transparency and reproducibility. This information allows others to accurately interpret the performance results, compare them with their own systems, and understand any potential variations. Factors like CPU generation, GPU model, storage type, available memory, background load, thermal conditions, and power settings can significantly affect benchmark outcomes. Sharing these details helps establish a reliable and fair performance baseline.

---

## 🎯 Test Files Used

| File Name           | Type               | Size       |
| ------------------- | ------------------ | ---------- |
| `random_digits.txt` | Text (0-9 digits)  |  ~1.00 GiB |
| `pi_digits.txt`     | Text (digits of π) |  ~286 MiB  |
|`hw_201.csv`         | CSV                |  ~17.1 MiB |
|`random_data.csv`    | CSV                |  ~1.00 GiB |


---

## ⏱️ TXTD-CUDA Benchmark Results

| File                 | Operation | Time Taken | Input File Size  | Effective Throughput (MB/s)  |
|----------------------|-----------|------------|------------------|------------------------------|
| `random_digits.txt`  | Encode    | 1.839s     | ~1024 MiB        | ~278.46 MB/s                 |
| `random_digits.txtd` | Decode    | 2.605s     | ~512 MiB         | ~393.02 MB/s                 |
| `pi_digits.txt`      | Encode    | 0.598s     | ~286 MiB         | ~119.57 MB/s                 |
| `pi_digits.txtd`     | Decode    | 0.756s     | ~143 MiB         | ~378.97 MB/s                 |

## ⏱️ TXTD-MT Benchmark Results

| File                 | Operation | Time Taken | Input File Size  | Effective Throughput (MB/s)  |
|----------------------|-----------|------------|------------------|------------------------------|
| `random_digits.txt`  | Encode    | 3.450s     | ~1024 MiB        | ~148.41 MB/s                 |
| `random_digits.txtd` | Decode    | 4.494s     | ~512 MiB         | ~227.91 MB/s                 |
| `pi_digits.txt`      | Encode    | 0.941s     | ~286 MiB         | ~151.99 MB/s                 |
| `pi_digits.txtd`     | Decode    | 1.147s     | ~143 MiB         | ~249.30 MB/s                 |
| `hw_201.csv`         | Encode    | 0.381s     | ~17.1 MiB        | ~22.44 MB/s                  |
| `hw_201.txtd`        | Decode    | 0.086s     | ~8.08 MiB        | ~187.91 MB/s                 |
| `random_data.csv`    | Encode    | 19.581s    | ~1024 MiB        | ~26.17 MB/s                  |
| `random_data.txtd`   | Decode    | 5.405s     | ~512 MiB         | ~189.44 MB/s                 |

## ⏱️ TXTD-ST Benchmark Results

| File                 | Operation | Time Taken | Input File Size  | Effective Throughput (MB/s)  |
|----------------------|-----------|------------|------------------|------------------------------|
| `random_digits.txt`  | Encode    | 32.687s    | ~1024 MiB        | ~16.41 MB/s                  |
| `random_digits.txtd` | Decode    | 34.388s    | ~512 MiB         | ~31.17 MB/s                  |
| `pi_digits.txt`      | Encode    | 9.141s     | ~286 MiB         | ~16.38 MB/s                  |
| `pi_digits.txtd`     | Decode    | 9.605s     | ~143 MiB         | ~31.18 MB/s                  |
| `hw_201.csv`         | Encode    | 0.508s     | ~17.1 MiB        | ~17.63 MB/s                  |
| `hw_201.txtd`        | Decode    | 0.560s     | ~8.08 MiB        | ~30.20 MB/s                  |
| `random_data.csv`    | Encode    | 27.601s    | ~1024 MiB        | ~19.44 MB/s                  |
| `random_data.txtd`   | Decode    | 32.209s    | ~512 MiB         | ~33.26 MB/s                  |



> *Note: Throughput is calculated as (output file size) / time taken.* <br>
> For encode step: output file size ≈ input file size ÷ 2 <br>
> For decode step: output file size ≈ input file size × 2

---

## 🔢 Observations

* **TXTD-CUDA** delivers the highest throughput across all files, reaching up to ~393 MB/s on decoding workloads, proving exceptionally efficient for large files.
* **TXTD-MT (OpenMP Multithreaded)** offers a strong balance between performance and CPU compatibility, achieving upto ~249 MB/s on certain decoding tasks.
* **TXTD-ST (Single Threaded)** shows expected lower performance, particularly on encoding operations, with throughput falling below ~20 MB/s in most cases.
* **CUDA decoding consistently outperforms all other methods**, making it ideal for read-heavy applications and environments with a capable GPU.
* **Encoding is generally faster than decoding** in the multithreaded implementation, but decoding still maintains a strong throughput.
* Results confirm **good scalability with file size**: larger files maintain high throughput, particularly on CUDA and MT pipelines.

---

## ✅ Conclusion

The benchmark results demonstrate that the `txtd` format, especially when paired with CUDA acceleration, excels in handling large-scale text and numerical data. Here's a breakdown:

- **TXTD-CUDA** is optimal for performance-critical workloads with access to GPU resources. It delivers the highest throughput and lowest latency.
- **TXTD-MT** offers excellent CPU-based performance and is a solid fallback where CUDA is not available.
- **TXTD-ST** serves as a baseline for comparison, clearly highlighting the advantages of parallel and GPU-accelerated approaches.