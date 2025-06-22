

# Build

## Single-threaded CPU Implementation

```
gcc txt_to_txtd.c -o txt_to_txtd.exe -lssl -lcrypto
gcc txtd_to_txt.c -o txtd_to_txt.exe -lssl -lcrypto
gcc csv_to_txtd.c -o csv_to_txtd.exe -lssl -lcrypto
gcc txtd_to_csv.c -o txtd_to_csv.exe -lssl -lcrypto
```

## Multithreaded CPU Implementation
```
gcc -fopenmp txt_to_txtd_mt.c -o txt_to_txtd_mt.exe -lssl -lcrypto
gcc -fopenmp txtd_to_txt_mt.c -o txtd_to_txt_mt.exe -lssl -lcrypto
gcc -fopenmp csv_to_txtd_mt.c -o csv_to_txtd_mt.exe -lssl -lcrypto
gcc -fopenmp txtd_to_csv_mt.c -o txtd_to_csv_mt.exe -lssl -lcrypto
```

> **Note: The above implementations were compiled using the GCC compiler from the MINGW64 environment in the MSYS2 terminal suite. If you're using GCC directly, the compilation commands will be similar to those shown for the Nvidia GPU implementation — except you'll use `gcc` instead of `nvcc`, and you'll need to adjust the name of the files to be compiled and the name of the output files accordingly.**

## Nvidia GPU Implementation

**On Windows (with OpenSSL in C:\OpenSSL-Win64):**
```
nvcc -I"C:\OpenSSL-Win64\include" -L"C:\OpenSSL-Win64\lib\VC\x64\MD" txt_to_txtd_cuda.cu -o txt_to_txtd_cuda.exe -llibssl -llibcrypto
nvcc -I"C:\OpenSSL-Win64\include" -L"C:\OpenSSL-Win64\lib\VC\x64\MD" txtd_to_txt_cuda.cu -o txtd_to_txt_cuda.exe -llibssl -llibcrypto
```

*Adjust include/library paths if necessary.*

---


