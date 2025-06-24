#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <openssl/sha.h>
#include "encoder.c"
#include <time.h>

// Helper to extract base name (without extension)
void get_basename(const char *filename, char *basename, size_t size) {
    const char *slash = strrchr(filename, '/');
    #ifdef _WIN32
    const char *bslash = strrchr(filename, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
    #endif
    const char *start = slash ? slash + 1 : filename;
    const char *dot = strrchr(start, '.');
    size_t len = dot ? (size_t)(dot - start) : strlen(start);
    if (len >= size) len = size - 1;
    strncpy(basename, start, len);
    basename[len] = '\0';
}

// Compute SHA-256 checksum and write as hex string to file
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
    // Uncomment to enable timing
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input.txt>\n", argv[0]);
        return 1;
    }

    char basename[256];
    get_basename(argv[1], basename, sizeof(basename));

    // Create directory
    #ifdef _WIN32
    mkdir(basename);
    #else
    mkdir(basename, 0777);
    #endif

    // Prepare output file paths
    char encoded_path[512], checksum_path[512];
    snprintf(encoded_path, sizeof(encoded_path), "%s/%s.encoded.txtd", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    // Open input and output files
    FILE *in = fopen(argv[1], "r");
    if (!in) {
        perror("Failed to open input file");
        return 1;
    }
    FILE *out = fopen(encoded_path, "wb");
    if (!out) {
        perror("Failed to create output file");
        fclose(in);
        return 1;
    }

    // Write first byte as 00000000 and a newline
    unsigned char first = 0x00;
    fwrite(&first, 1, 1, out);
    fputc('\n', out);

    // Buffering for input and output
    #define READ_BUF_SIZE 65536
    #define WRITE_BUF_SIZE 65536
    unsigned char write_buf[WRITE_BUF_SIZE];
    size_t write_pos = 0;
    char read_buf[READ_BUF_SIZE];
    size_t read_len = 0, read_pos = 0;

    unsigned char buffer = 0;
    int half = 0;
    int ch;

    while (1) {
        // Refill read buffer if needed
        if (read_pos >= read_len) {
            read_len = fread(read_buf, 1, READ_BUF_SIZE, in);
            read_pos = 0;
            if (read_len == 0) break;
        }
        ch = (unsigned char)read_buf[read_pos++];
        unsigned char val = encode_char((char)ch);
        if (half == 0) {
            buffer = val << 4;
            half = 1;
        } else {
            buffer |= val;
            write_buf[write_pos++] = buffer;
            half = 0;
            if (write_pos == WRITE_BUF_SIZE) {
                fwrite(write_buf, 1, WRITE_BUF_SIZE, out);
                write_pos = 0;
            }
        }
    }
    if (half == 0) {
        buffer = 0b1111 << 4;
        write_buf[write_pos++] = buffer;
    } else {
        buffer |= 0b1111;
        write_buf[write_pos++] = buffer;
    }
    if (write_pos > 0) {
        fwrite(write_buf, 1, write_pos, out);
    }

    fclose(in);
    fclose(out);

    // Write checksum
    if (write_checksum(argv[1], checksum_path) != 0) {
        fprintf(stderr, "Failed to write checksum file\n");
        return 1;
    }

    printf("Encoded file: %s\nChecksum file: %s\n", encoded_path, checksum_path);

    // Uncomment to enable timing
    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}