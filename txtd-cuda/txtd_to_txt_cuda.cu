#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#define CHUNK_SIZE 2000  // Number of nibbles per GPU thread (change as needed)

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

// Helper to compute SHA-256 checksum and write as hex string to buffer
int compute_checksum(const char *filename, char *out_hex, size_t hex_size) {
    FILE *in = fopen(filename, "rb");
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
    char *p = out_hex;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        p += snprintf(p, hex_size - (p - out_hex), "%02x", hash[i]);
    *p = '\0';
    return 0;
}

// Helper to read checksum from file (first line)
int read_checksum(const char *filename, char *out_hex, size_t hex_size) {
    FILE *in = fopen(filename, "r");
    if (!in) return 1;
    if (!fgets(out_hex, (int)hex_size, in)) {
        fclose(in);
        return 2;
    }
    // Remove trailing newline if present
    size_t len = strlen(out_hex);
    if (len > 0 && out_hex[len-1] == '\n') out_hex[len-1] = '\0';
    fclose(in);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder_name>\n", argv[0]);
        return 1;
    }
    char *folder = argv[1];
    char inpath[512], outpath[512], checksum_path[512];
    snprintf(inpath, sizeof(inpath), "%s/%s.encoded.txtd", folder, folder);
    snprintf(outpath, sizeof(outpath), "%s/%s_decoded.txt", folder, folder);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", folder, folder);

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

    // Compute checksum of decoded file
    char decoded_checksum[SHA256_DIGEST_LENGTH*2+1];
    if (compute_checksum(outpath, decoded_checksum, sizeof(decoded_checksum)) != 0) {
        fprintf(stderr, "Failed to compute checksum for %s\n", outpath);
        return 1;
    }

    // Read original checksum
    char original_checksum[SHA256_DIGEST_LENGTH*2+1];
    if (read_checksum(checksum_path, original_checksum, sizeof(original_checksum)) != 0) {
        fprintf(stderr, "Failed to read checksum file %s\n", checksum_path);
        return 1;
    }

    // Compare checksums
    if (strcmp(decoded_checksum, original_checksum) == 0) {
        printf("Decoding completed successfully. The decoded file matches the original checksum.\n");
    } else {
        printf("Decoding completed, but the decoded file does NOT match the original checksum. Please verify the integrity of your files.\n");
    }
    return 0;
}
