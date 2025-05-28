#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHUNK_SIZE 1  // Number of nibbles per GPU thread (change as needed)

// Device decode table
__device__ __constant__ char decode_table[16] = {
    '0','1','2','3','4','5','6','7','8','9','.',' ','\t','\n',',','\0'
};

__global__ void decode_kernel(const unsigned char *in, char *out, size_t n_nibbles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t start = idx * CHUNK_SIZE;
    size_t end = start + CHUNK_SIZE;
    if (start >= n_nibbles) return;
    if (end > n_nibbles) end = n_nibbles;
    for (size_t i = start; i < end; ++i) {
        unsigned char nibble = (i % 2 == 0) ? (in[i/2] >> 4) & 0x0F : in[i/2] & 0x0F;
        out[i] = decode_table[nibble];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder_name>\n", argv[0]);
        return 1;
    }
    char *folder = argv[1];
    char inpath[512], outpath[512];
    snprintf(inpath, sizeof(inpath), "%s/%s.encoded.txtd", folder, folder);
    snprintf(outpath, sizeof(outpath), "%s/%s_decoded.txt", folder, folder);

    FILE *in = fopen(inpath, "rb");
    if (!in) { perror("open input"); return 1; }
    fseek(in, 0, SEEK_END);
    size_t n_bytes = ftell(in);
    fseek(in, 0, SEEK_SET);
    unsigned char *buf = (unsigned char*)malloc(n_bytes);
    fread(buf, 1, n_bytes, in);
    fclose(in);

    size_t n_nibbles = n_bytes * 2;
    char *d_out, *outbuf = (char*)malloc(n_nibbles);
    unsigned char *d_in;
    cudaMalloc(&d_in, n_bytes); cudaMalloc(&d_out, n_nibbles);
    cudaMemcpy(d_in, buf, n_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_threads = (n_nibbles + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    decode_kernel<<<num_blocks, threads_per_block>>>(d_in, d_out, n_nibbles);
    cudaMemcpy(outbuf, d_out, n_nibbles, cudaMemcpyDeviceToHost);

    // Write output, stop at first '\0'
    FILE *out = fopen(outpath, "w");
    for (size_t i = 0; i < n_nibbles; ++i) {
        if (outbuf[i] == '\0') break;
        fputc(outbuf[i], out);
    }
    fclose(out);

    cudaFree(d_in); cudaFree(d_out); free(buf); free(outbuf);

    printf("Decoded file: %s\n", outpath);
    return 0;
}
