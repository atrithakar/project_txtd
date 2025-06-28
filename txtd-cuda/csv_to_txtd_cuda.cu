#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define MKDIR(dir) mkdir(dir, 0777)
#endif

#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE (64 * 1024 * 1024)  // 64MB chunk size
#define LINE_BUF_SIZE 8192

__device__ __constant__ unsigned char encode_table[128];

// Kernel: encode chars to nibbles and pack into bytes
__global__ void encode_and_pack_kernel(const char *in, unsigned char *out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 2 >= n) return;
    unsigned char high = (in[i * 2] < 128) ? encode_table[(int)in[i * 2]] : 0xFF;
    unsigned char low  = (i * 2 + 1 < n && in[i * 2 + 1] < 128) ? encode_table[(int)in[i * 2 + 1]] : 0x0F;
    out[i] = (high << 4) | low;
}

void fill_encode_table(unsigned char *table) {
    for (int i = 0; i < 128; ++i) table[i] = 0xFF;
    table['0'] = 0x0; table['1'] = 0x1; table['2'] = 0x2; table['3'] = 0x3;
    table['4'] = 0x4; table['5'] = 0x5; table['6'] = 0x6; table['7'] = 0x7;
    table['8'] = 0x8; table['9'] = 0x9; table['.'] = 0xA; table[' '] = 0xB;
    table['\t'] = 0xC; table['\n'] = 0xD; table[','] = 0xE; table['\0'] = 0xF;
}

const char COMMON_DELIMS[] = {',', ';', '\t', '|', '^', '~', 0};
int is_common_delim(char c) {
    for (int i = 0; COMMON_DELIMS[i]; ++i)
        if (c == COMMON_DELIMS[i]) return 1;
    return 0;
}

int write_checksum(const char *input_filename, const char *output_filename) {
    FILE *in = fopen(input_filename, "rb");
    if (!in) return 1;
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    unsigned char buf[32768];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
        SHA256_Update(&sha256, buf, n);
    fclose(in);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256);
    FILE *out = fopen(output_filename, "w");
    if (!out) return 2;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        fprintf(out, "%02x", hash[i]);
    fprintf(out, "\n");
    fclose(out);
    return 0;
}

int main(int argc, char *argv[]) {
    clock_t clock_start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.csv\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) { perror("Failed to open input"); return 1; }

    char basename[256];
    const char *slash = strrchr(argv[1], '/');
    #ifdef _WIN32
    const char *bslash = strrchr(argv[1], '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
    #endif
    const char *start = slash ? slash + 1 : argv[1];
    const char *dot = strrchr(start, '.');
    size_t len = dot ? (size_t)(dot - start) : strlen(start);
    strncpy(basename, start, len); basename[len] = '\0';

    MKDIR(basename);

    char txtd_path[300], checksum_path[300];
    snprintf(txtd_path, sizeof(txtd_path), "%s/%s.txtd", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    if (write_checksum(argv[1], checksum_path) != 0) {
        fprintf(stderr, "Failed to write checksum.\n");
        fclose(in);
        return 1;
    }

    FILE *out = fopen(txtd_path, "wb");
    if (!out) { perror("Failed to open output"); fclose(in); return 1; }

    fputc(0xFF, out); fputc('\n', out);  // preamble

    // Read and write header line as-is
    char line[LINE_BUF_SIZE];
    if (!fgets(line, sizeof(line), in)) {
        fprintf(stderr, "Empty input file.\n");
        fclose(in); fclose(out);
        return 1;
    }
    fputs(line, out);
    if (line[strlen(line) - 1] != '\n') fputc('\n', out);

    unsigned char encode_map[128];
    fill_encode_table(encode_map);
    cudaMemcpyToSymbol(encode_table, encode_map, 128);

    // Allocate pinned memory
    char *host_chunk;
    cudaHostAlloc((void**)&host_chunk, CHUNK_SIZE, cudaHostAllocDefault);

    char *device_in;
    unsigned char *device_out;
    size_t encoded_chunk_size = (CHUNK_SIZE + 1) / 2;
    cudaMalloc(&device_in, CHUNK_SIZE);
    cudaMalloc(&device_out, encoded_chunk_size);

    size_t total_read = 0;
    while (!feof(in)) {
        size_t bytes_read = fread(host_chunk, 1, CHUNK_SIZE, in);
        if (bytes_read == 0) break;

        cudaMemcpy(device_in, host_chunk, bytes_read, cudaMemcpyHostToDevice);

        int num_threads = (bytes_read + 1) / 2;
        int blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        encode_and_pack_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_in, device_out, bytes_read);
        cudaDeviceSynchronize();

        unsigned char *encoded_host;
        cudaHostAlloc((void**)&encoded_host, encoded_chunk_size, cudaHostAllocDefault);
        cudaMemcpy(encoded_host, device_out, (bytes_read + 1) / 2, cudaMemcpyDeviceToHost);

        fwrite(encoded_host, 1, (bytes_read + 1) / 2, out);
        cudaFreeHost(encoded_host);

        total_read += bytes_read;
    }

    fclose(in);
    fclose(out);
    cudaFree(device_in); cudaFree(device_out);
    cudaFreeHost(host_chunk);

    printf("Encoding complete.\nStored at: %s\n", txtd_path);
    printf("Checksum stored at: %s\n", checksum_path);
    printf("Elapsed: %.3fs\n", (double)(clock() - clock_start) / CLOCKS_PER_SEC);

    return 0;
}
