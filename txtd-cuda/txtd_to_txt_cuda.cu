#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <time.h>

#define THREADS_PER_BLOCK 256

__device__ __constant__ char decode_table[16] = {
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', '.', ' ', '\t', '\n', ',', '\0'
};

__global__ void decode_and_unpack_kernel(const unsigned char *in, char *out, size_t in_size, size_t expected_len) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_size) return;

    unsigned char byte = in[i];
    unsigned char high = (byte >> 4) & 0x0F;
    unsigned char low  = byte & 0x0F;

    if (i * 2 < expected_len)
        out[i * 2] = decode_table[high];
    if (i * 2 + 1 < expected_len)
        out[i * 2 + 1] = decode_table[low];
}

int verify_checksum(const char *decoded_path, const char *checksum_path) {
    // Read expected hash
    FILE *checkf = fopen(checksum_path, "r");
    if (!checkf) {
        perror("open checksum");
        return -1;
    }

    char expected[SHA256_DIGEST_LENGTH * 2 + 1];
    if (!fgets(expected, sizeof(expected), checkf)) {
        fclose(checkf);
        return -1;
    }
    fclose(checkf);

    // Calculate hash of decoded file
    FILE *fp = fopen(decoded_path, "rb");
    if (!fp) {
        perror("open decoded");
        return -1;
    }

    SHA256_CTX sha;
    SHA256_Init(&sha);
    unsigned char buf[32768];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        SHA256_Update(&sha, buf, n);
    }
    fclose(fp);

    unsigned char hash_bin[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash_bin, &sha);

    char actual[SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        sprintf(&actual[i * 2], "%02x", hash_bin[i]);
    actual[SHA256_DIGEST_LENGTH * 2] = '\0';

    return strcmp(expected, actual) == 0;
}

int main(int argc, char *argv[]) {
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder_path>\n", argv[0]);
        return 1;
    }

    char *folder = argv[1];
    char base[256];
    const char *slash = strrchr(folder, '/');
#ifdef _WIN32
    const char *bslash = strrchr(folder, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
#endif
    const char *start_ptr = slash ? slash + 1 : folder;
    strncpy(base, start_ptr, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';

    char encoded_file[512], checksum_file[512], output_file[512];
    snprintf(encoded_file, sizeof(encoded_file), "%s/%s.encoded.txtd", folder, base);
    snprintf(checksum_file, sizeof(checksum_file), "%s/%s.checksum.txt", folder, base);
    snprintf(output_file, sizeof(output_file), "%s/%s_decoded.txt", folder, base);

    FILE *fin = fopen(encoded_file, "rb");
    if (!fin) {
        perror("open encoded");
        return 1;
    }

    fseek(fin, 0, SEEK_END);
    size_t total_size = ftell(fin);
    fseek(fin, 2, SEEK_SET);  // Skip 0x00 + newline

    size_t encoded_size = total_size - 2;
    unsigned char *packed = (unsigned char *)malloc(encoded_size);
    fread(packed, 1, encoded_size, fin);
    fclose(fin);

    size_t decoded_len = encoded_size * 2;
    char *decoded = (char *)malloc(decoded_len);

    unsigned char *d_in;
    char *d_out;
    cudaMalloc(&d_in, encoded_size);
    cudaMalloc(&d_out, decoded_len);
    cudaMemcpy(d_in, packed, encoded_size, cudaMemcpyHostToDevice);

    int blocks = (encoded_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    decode_and_unpack_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, encoded_size, decoded_len);

    cudaMemcpy(decoded, d_out, decoded_len, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    free(packed);

    FILE *fout = fopen(output_file, "wb");
    if (!fout) {
        perror("open output");
        free(decoded);
        return 1;
    }
    fwrite(decoded, 1, decoded_len, fout);
    fclose(fout);
    free(decoded);

    printf("Decoded to: %s\n", output_file);
    int verified = verify_checksum(output_file, checksum_file);
    if (verified == 1) {
        printf("Decoding completed successfully. The decoded file matches the original checksum.\n");
    } else if (verified == 0) {
        printf("Decoding completed, but the decoded file does NOT match the original checksum.\n");
    } else {
        printf("Checksum verification failed due to file access error.\n");
    }

    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
