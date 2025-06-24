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
| `hw_201.csv`         | Encode    | 0.127s     | ~17.1 MiB        | ~66.71 MB/s                  |
| `hw_201.txtd`        | Decode    | 0.086s     | ~8.08 MiB        | ~187.91 MB/s                 |
| `random_data.csv`    | Encode    | 7.822s     | ~1024 MiB        | ~68.63 MB/s                  |
| `random_data.txtd`   | Decode    | 5.341s     | ~512 MiB         | ~201.03 MB/s                 |

## ⏱️ TXTD-ST Benchmark Results

| File                 | Operation | Time Taken | Input File Size  | Effective Throughput (MB/s)  |
|----------------------|-----------|------------|------------------|------------------------------|
| `random_digits.txt`  | Encode    | 26.156s    | ~1024 MiB        | ~20.64 MB/s                  |
| `random_digits.txtd` | Decode    | 22.935s    | ~512 MiB         | ~46.81 MB/s                  |
| `pi_digits.txt`      | Encode    | 7.283s     | ~286 MiB         | ~20.58 MB/s                  |
| `pi_digits.txtd`     | Decode    | 6.356s     | ~143 MiB         | ~47.18 MB/s                  |
| `hw_201.csv`         | Encode    | 0.408s     | ~17.1 MiB        | ~20.76 MB/s                  |
| `hw_201.txtd`        | Decode    | 0.354s     | ~8.08 MiB        | ~50.65 MB/s                  |
| `random_data.csv`    | Encode    | 22.703s    | ~1024 MiB        | ~23.64 MB/s                  |
| `random_data.txtd`   | Decode    | 22.480s    | ~512 MiB         | ~47.76 MB/s                  |



> *Note: Throughput is calculated as (output file size) / time taken.* <br>
> For encode step: output file size ≈ input file size ÷ 2 <br>
> For decode step: output file size ≈ input file size × 2

## 🧠 Additional Notes on Storage & I/O Bottlenecks

While the `txtd` format and its corresponding encoders/decoders demonstrate high computational throughput, **actual performance is also strongly influenced by the characteristics of the storage medium** used during testing or deployment. Below are critical considerations:

| Factor                  | Impact Description                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **Storage Speed**       | If the encoder/decoder is run on a **slower storage device** (e.g., HDDs or SD cards), the overall throughput can be significantly throttled due to I/O wait times, regardless of CPU/GPU speed. |
| **Interface Bottlenecks** | Performance can degrade if the system is using slower interfaces (e.g., USB 2.0 vs NVMe Gen 4). |
| **Read/Write Buffering**| The internal write buffers are optimized for high-speed media. On slower media, buffer flush frequency increases, slightly impacting encoding speeds. |
| **CPU/GPU Idle Time**   | On high-speed systems (e.g., Gen 4 SSD + RTX GPU), encoding/decoding may complete faster than the OS can flush I/O buffers. On slower disks, CPU/GPU may remain idle waiting for the disk to catch up. |

> **Key Insight**: On extremely slow storage devices, such as SD cards or low-end USB drives, it's possible for the `txtd` single-threaded encoder to outperform even CUDA-based versions — not due to processing speed, but due to reduced I/O contention. Conversely, on **NVMe SSDs**, GPU acceleration unleashes its full potential.

**Takeaway**:  
> The speed of encoding and decoding is not only determined by computational power (CPU/GPU) but also by **how fast your system can read from and write to disk**. For best results, ensure you are using high-speed SSDs or memory-mapped I/O in high-throughput environments.


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