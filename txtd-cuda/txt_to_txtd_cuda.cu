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

#define THREADS_PER_BLOCK 256

__device__ __constant__ unsigned char encode_table[128];

__global__ void encode_and_pack_kernel(const char *in, unsigned char *out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 2 >= n) return;
    unsigned char high = (in[i * 2] >= 0 && in[i * 2] < 128) ? encode_table[(int)in[i * 2]] : 0xFF;
    unsigned char low  = (i * 2 + 1 < n && in[i * 2 + 1] >= 0 && in[i * 2 + 1] < 128) ? encode_table[(int)in[i * 2 + 1]] : 0x0F;
    out[i] = (high << 4) | low;
}

void fill_encode_table(unsigned char *table) {
    for (int i = 0; i < 128; ++i) table[i] = 0xFF;
    table['0'] = 0b0000; table['1'] = 0b0001; table['2'] = 0b0010; table['3'] = 0b0011;
    table['4'] = 0b0100; table['5'] = 0b0101; table['6'] = 0b0110; table['7'] = 0b0111;
    table['8'] = 0b1000; table['9'] = 0b1001; table['.'] = 0b1010; table[' '] = 0b1011;
    table['\t'] = 0b1100; table['\n'] = 0b1101; table[','] = 0b1110; table['\0'] = 0b1111;
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
    MKDIR(basename);

    char outpath[512], checksum_path[512];
    snprintf(outpath, sizeof(outpath), "%s/%s.encoded.txtd", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    FILE *in = fopen(input_path, "rb");
    if (!in) { perror("open input"); return 1; }
    fseek(in, 0, SEEK_END);
    size_t n = ftell(in);
    fseek(in, 0, SEEK_SET);

    char *buf;
    cudaMallocHost(&buf, n);
    fread(buf, 1, n, in);
    fclose(in);

    unsigned char *d_out, *out;
    char *d_in;
    size_t out_size = (n + 1) / 2;

    cudaMalloc(&d_in, n);
    cudaMalloc(&d_out, out_size);
    cudaMemcpy(d_in, buf, n, cudaMemcpyHostToDevice);

    unsigned char table[128];
    fill_encode_table(table);
    cudaMemcpyToSymbol(encode_table, table, 128);

    int num_threads = (int)((n + 1) / 2);
    int blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    encode_and_pack_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, n);

    cudaMallocHost(&out, out_size);
    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

    FILE *fout = fopen(outpath, "wb");
    fwrite(out, 1, out_size, fout);
    fclose(fout);

    write_checksum(input_path, checksum_path);

    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(buf); cudaFreeHost(out);

    printf("Encoded file: %s\nChecksum file: %s\n", outpath, checksum_path);
    return 0;
}
