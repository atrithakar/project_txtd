#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <time.h>

#define READ_BUF_SIZE 65536
#define WRITE_BUF_SIZE 65536

__device__ __constant__ char decode_table[16] = {
    '0','1','2','3','4','5','6','7','8','9','.',' ','\t','\n',',','\0'
};

__global__ void decode_kernel(const unsigned char *in, char *out, size_t n_bytes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_bytes * 2) {
        unsigned char byte = in[idx / 2];
        unsigned char nibble = (idx % 2 == 0) ? (byte >> 4) & 0x0F : byte & 0x0F;
        out[idx] = decode_table[nibble];
    }
}

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

int read_checksum(const char *filename, char *out_hex, size_t hex_size) {
    FILE *in = fopen(filename, "r");
    if (!in) return 1;
    if (!fgets(out_hex, (int)hex_size, in)) {
        fclose(in);
        return 2;
    }
    size_t len = strlen(out_hex);
    if (len > 0 && out_hex[len - 1] == '\n') out_hex[len - 1] = '\0';
    fclose(in);
    return 0;
}

char infer_delimiter(const char *line) {
    const char COMMON_DELIMS[] = {',', ';', '\t', '|', '^', '~', 0};
    int counts[256] = {0}, in_quote = 0;
    for (size_t i = 0; line[i] && line[i] != '\n'; ++i) {
        char c = line[i];
        if (c == '"' || c == '\'') {
            in_quote = !in_quote;
            continue;
        }
        if (in_quote) continue;
        for (int j = 0; COMMON_DELIMS[j]; ++j) {
            if (c == COMMON_DELIMS[j]) counts[(unsigned char)c]++;
        }
    }
    int max_count = 0;
    char delimiter = ',';
    for (int j = 0; COMMON_DELIMS[j]; ++j) {
        char c = COMMON_DELIMS[j];
        if (counts[(unsigned char)c] > max_count) {
            max_count = counts[(unsigned char)c];
            delimiter = c;
        }
    }
    return delimiter;
}

int main(int argc, char *argv[]) {
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
        return 1;
    }

    char folder[256];
    strncpy(folder, argv[1], sizeof(folder) - 1);
    folder[sizeof(folder) - 1] = '\0';

    size_t len = strlen(folder);
    while (len > 0 && (folder[len - 1] == '/' || folder[len - 1] == '\\')) {
        folder[--len] = '\0';
    }

    char *basename = folder;
    char *slash = strrchr(folder, '/');
    #ifdef _WIN32
    char *bslash = strrchr(folder, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
    #endif
    if (slash) basename = slash + 1;

    char txtd_path[300], csv_path[300], checksum_path[300];
    snprintf(txtd_path, sizeof(txtd_path), "%s/%s.txtd", folder, basename);
    snprintf(csv_path, sizeof(csv_path), "%s/%s.csv", folder, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", folder, basename);

    FILE *in = fopen(txtd_path, "rb");
    if (!in) { perror("Failed to open packed txtd file"); return 1; }
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { perror("Failed to create output CSV file"); fclose(in); return 1; }

    fgetc(in); // skip 0xFF
    fgetc(in); // skip newline

    char header[8192];
    if (!fgets(header, sizeof(header), in)) {
        fprintf(stderr, "Failed to read header.\n");
        fclose(in); fclose(csv); return 1;
    }
    fputs(header, csv);

    char delimiter = infer_delimiter(header);
    if (header[strlen(header) - 1] != '\n') fputc('\n', csv);

    // --- Optimized: Use larger pinned buffers and batch kernel launches ---
    const size_t CHUNK_SIZE = 64 * 1024 * 1024; // 64MB chunk for GPU
    unsigned char *h_in;
    char *h_out;
    cudaHostAlloc((void**)&h_in, CHUNK_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_out, CHUNK_SIZE * 2, cudaHostAllocDefault);

    unsigned char *d_in;
    char *d_out;
    cudaMalloc(&d_in, CHUNK_SIZE);
    cudaMalloc(&d_out, CHUNK_SIZE * 2);

    char write_buf[WRITE_BUF_SIZE];
    size_t write_pos = 0;
    size_t bytes_read;

    while ((bytes_read = fread(h_in, 1, CHUNK_SIZE, in)) > 0) {
        cudaMemcpy(d_in, h_in, bytes_read, cudaMemcpyHostToDevice);
        size_t n_nibbles = bytes_read * 2;
        int blocks = (n_nibbles + 255) / 256;
        decode_kernel<<<blocks, 256>>>(d_in, d_out, bytes_read);
        cudaMemcpyAsync(h_out, d_out, n_nibbles, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Write output, replace ',' with delimiter, stop at first '\0'
        for (size_t i = 0; i < n_nibbles; ++i) {
            char ch = h_out[i];
            if ((unsigned char)ch == 0x0F || ch == '\0') goto finish;
            if (ch == ',' && delimiter != ',') ch = delimiter;
            write_buf[write_pos++] = ch;
            if (write_pos == WRITE_BUF_SIZE) {
                fwrite(write_buf, 1, WRITE_BUF_SIZE, csv);
                write_pos = 0;
            }
        }
    }

finish:
    if (write_pos > 0) fwrite(write_buf, 1, write_pos, csv);
    fclose(in); fclose(csv);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_in); cudaFree(d_out);

    printf("CSV reconstructed at: %s\n", csv_path);

    char decoded_checksum[65], original_checksum[65];
    if (compute_checksum(csv_path, decoded_checksum, sizeof(decoded_checksum)) != 0) {
        fprintf(stderr, "Failed to compute checksum.\n");
        return 1;
    }
    if (read_checksum(checksum_path, original_checksum, sizeof(original_checksum)) != 0) {
        fprintf(stderr, "Failed to read original checksum.\n");
        return 1;
    }

    if (strcmp(decoded_checksum, original_checksum) == 0)
        printf("Decoding completed successfully. The decoded CSV matches the original checksum.\n");
    else
        printf("Decoded CSV does NOT match the original checksum.\n");

    printf("Elapsed: %.3fs\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    return 0;
}
