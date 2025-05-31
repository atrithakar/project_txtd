#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define MKDIR(dir) mkdir(dir, 0777)
#endif

#define CHUNK_SIZE 500  // Number of bytes per GPU thread (change as needed)

// Host-side encode_char table
__device__ __constant__ unsigned char encode_table[128] = {
    // ...fill with 0xFF for all, then set valid chars below...
    // 0-9, ., space, tab, newline, comma, '\0'
    // ASCII: '0' = 48, '1' = 49, ..., '9' = 57, '.' = 46, ' ' = 32, '\t' = 9, '\n' = 10, ',' = 44
    // e.g. encode_table['0'] = 0b0000; encode_table['1'] = 0b0001; etc.
};

__global__ void encode_kernel(const char *in, unsigned char *out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t start = idx * CHUNK_SIZE;
    size_t end = start + CHUNK_SIZE;
    if (start >= n) return;
    if (end > n) end = n;
    for (size_t i = start; i < end; ++i) {
        char c = in[i];
        unsigned char code = (c >= 0 && c < 128) ? encode_table[(int)c] : 0xFF;
        out[i] = code;
    }
}

void fill_encode_table(unsigned char *table) {
    for (int i = 0; i < 128; ++i) table[i] = 0xFF;
    table['0'] = 0b0000; table['1'] = 0b0001; table['2'] = 0b0010; table['3'] = 0b0011;
    table['4'] = 0b0100; table['5'] = 0b0101; table['6'] = 0b0110; table['7'] = 0b0111;
    table['8'] = 0b1000; table['9'] = 0b1001; table['.'] = 0b1010; table[' '] = 0b1011;
    table['\t'] = 0b1100; table['\n'] = 0b1101; table[','] = 0b1110; table['\0'] = 0b1111;
}

// Helper to compute SHA-256 checksum and write as hex string to file
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
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input.txt>\n", argv[0]);
        return 1;
    }
    char *input_path = argv[1];
    char basename[256];
    const char *slash = strrchr(input_path, '/');
    #ifdef _WIN32
    const char *bslash = strrchr(input_path, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
    #endif
    const char *start = slash ? slash + 1 : input_path;
    const char *dot = strrchr(start, '.');
    size_t len = dot ? (size_t)(dot - start) : strlen(start);
    strncpy(basename, start, len); basename[len] = '\0';

    // Create output directory named after input file (without extension)
    MKDIR(basename);

    char outpath[512], checksum_path[512];
    snprintf(outpath, sizeof(outpath), "%s/%s.encoded.txtd", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    // Read input file
    FILE *in = fopen(input_path, "rb");
    if (!in) { perror("open input"); return 1; }
    fseek(in, 0, SEEK_END);
    size_t n = ftell(in);
    fseek(in, 0, SEEK_SET);
    char *buf = (char*)malloc(n);
    fread(buf, 1, n, in);
    fclose(in);

    // CUDA encode
    char *d_in; unsigned char *d_out;
    cudaMalloc(&d_in, n); cudaMalloc(&d_out, n);
    cudaMemcpy(d_in, buf, n, cudaMemcpyHostToDevice);

    unsigned char table[128]; fill_encode_table(table);
    cudaMemcpyToSymbol(encode_table, table, 128);

    int threads_per_block = 256;
    int num_threads = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    encode_kernel<<<num_blocks, threads_per_block>>>(d_in, d_out, n);

    unsigned char *codes = (unsigned char*)malloc(n);
    cudaMemcpy(codes, d_out, n, cudaMemcpyDeviceToHost);

    // Pack nibbles into bytes
    FILE *out = fopen(outpath, "wb");
    unsigned char byte = 0;
    int half = 0;
    for (size_t i = 0; i < n; ++i) {
        if (codes[i] == 0xFF) { fprintf(stderr, "Invalid char at %zu\n", i); exit(1); }
        if (half == 0) { byte = codes[i] << 4; half = 1; }
        else { byte |= codes[i]; fwrite(&byte, 1, 1, out); half = 0; }
    }
    if (half == 0) { byte = 0b1111 << 4; fwrite(&byte, 1, 1, out); }
    else { byte |= 0b1111; fwrite(&byte, 1, 1, out); }
    fclose(out);

    // Write checksum of original input file
    if (write_checksum(input_path, checksum_path) != 0) {
        fprintf(stderr, "Failed to write checksum file\n");
        // continue anyway
    }

    cudaFree(d_in); cudaFree(d_out); free(buf); free(codes);

    printf("Encoded file: %s\nChecksum file: %s\n", outpath, checksum_path);
    return 0;
}
