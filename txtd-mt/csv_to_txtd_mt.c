#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <openssl/sha.h>
#include <omp.h>
#include <time.h>

#define WRITE_BUF_SIZE 65536
#define MAX_LINE_LEN 8192

// Map character to 4-bit code
unsigned char encode_char(char c) {
    switch (c) {
        case '0': return 0b0000;
        case '1': return 0b0001;
        case '2': return 0b0010;
        case '3': return 0b0011;
        case '4': return 0b0100;
        case '5': return 0b0101;
        case '6': return 0b0110;
        case '7': return 0b0111;
        case '8': return 0b1000;
        case '9': return 0b1001;
        case '.': return 0b1010;
        case ' ': return 0b1011;
        case '\t': return 0b1100;
        case '\n': return 0b1101;
        case ',' : return 0b1110;
        case '\0': return 0b1111;
        default: return 0xFF;
    }
}

// Utility: get file name without extension
void get_basename(char *basename, const char *path) {
    const char *last_slash = strrchr(path, '/');
    #ifdef _WIN32
    const char *bslash = strrchr(path, '\\');
    if (!last_slash || (bslash && bslash > last_slash)) last_slash = bslash;
    #endif
    const char *filename = last_slash ? last_slash + 1 : path;
    strcpy(basename, filename);
    char *dot = strrchr(basename, '.');
    if (dot) *dot = '\0';
}

// List of common delimiters
const char COMMON_DELIMS[] = {',', ';', '\t', '|', '^', '~', 0};
int is_common_delim(char c) {
    for (int i = 0; COMMON_DELIMS[i]; ++i)
        if (c == COMMON_DELIMS[i]) return 1;
    return 0;
}

// Improved delimiter detection: only count common delimiters outside quotes
char detect_delimiter(const char *line) {
    int counts[256] = {0};
    int in_quote = 0;
    for (size_t i = 0; line[i] && line[i] != '\n'; ++i) {
        char c = line[i];
        if (c == '"' || c == '\'') { in_quote = !in_quote; continue; }
        if (in_quote) continue;
        if (is_common_delim(c)) counts[(unsigned char)c]++;
    }
    int max_count = 0;
    char delimiter = ',';
    for (int i = 0; COMMON_DELIMS[i]; ++i) {
        char c = COMMON_DELIMS[i];
        if (counts[(unsigned char)c] > max_count) {
            max_count = counts[(unsigned char)c];
            delimiter = c;
        }
    }
    return delimiter;
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
    // Uncomment to enable timing
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.csv\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) {
        perror("Failed to open input CSV file");
        return 1;
    }

    char basename[256];
    get_basename(basename, argv[1]);

    // Create directory with csv name
    #ifdef _WIN32
    mkdir(basename);
    #else
    mkdir(basename, 0777);
    #endif

    char txtd_path[300], checksum_path[300];
    snprintf(txtd_path, sizeof(txtd_path), "%s/%s.txtd", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    // Write checksum of original CSV file before encoding
    if (write_checksum(argv[1], checksum_path) != 0) {
        fprintf(stderr, "Failed to write checksum file\n");
        fclose(in);
        return 1;
    }

    FILE *out = fopen(txtd_path, "wb");
    if (!out) {
        perror("Failed to create output txtd file");
        fclose(in);
        return 1;
    }

    // Write first 8 bits as 1111 1111, then newline
    unsigned char first = 0xFF; // 1111 1111
    fwrite(&first, 1, 1, out);
    fputc('\n', out);

    // Read and write header (first line) as-is, then newline
    char line[MAX_LINE_LEN];
    if (!fgets(line, sizeof(line), in)) {
        fprintf(stderr, "Input CSV is empty.\n");
        fclose(in); fclose(out);
        return 1;
    }
    fputs(line, out);
    if (line[strlen(line)-1] != '\n') fputc('\n', out); // ensure newline

    // --- Efficient multithreaded encoding and buffering ---
    // Read the rest of the file into a large buffer
    size_t data_cap = 16 * 1024 * 1024;
    size_t data_len = 0;
    char *data_buf = (char*)malloc(data_cap);
    while (fgets(line, sizeof(line), in)) {
        size_t len = strlen(line);
        if (data_len + len >= data_cap) {
            data_cap *= 2;
            data_buf = (char*)realloc(data_buf, data_cap);
        }
        memcpy(data_buf + data_len, line, len);
        data_len += len;
    }
    fclose(in);

    // Encode in parallel
    unsigned char *encoded = (unsigned char*)malloc(data_len);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < data_len; ++i) {
        char c = data_buf[i];
        // Delimiter normalization (outside quotes not handled, but matches previous logic)
        if (is_common_delim(c)) c = ',';
        encoded[i] = encode_char(c);
    }

    // Pack nibbles into bytes and buffer writes
    unsigned char *write_buffer = (unsigned char*)malloc(WRITE_BUF_SIZE);
    size_t write_pos = 0;
    int half = 0;
    unsigned char byte = 0;
    for (size_t i = 0; i < data_len; ++i) {
        if (encoded[i] == 0xFF) continue;
        if (half == 0) {
            byte = encoded[i] << 4;
            half = 1;
        } else {
            byte |= encoded[i];
            write_buffer[write_pos++] = byte;
            half = 0;
            if (write_pos == WRITE_BUF_SIZE) {
                fwrite(write_buffer, 1, WRITE_BUF_SIZE, out);
                write_pos = 0;
            }
        }
    }
    // Final nibble padding
    if (half == 0) {
        byte = 0b1111 << 4;
        write_buffer[write_pos++] = byte;
    } else {
        byte |= 0b1111;
        write_buffer[write_pos++] = byte;
    }
    if (write_pos > 0) {
        fwrite(write_buffer, 1, write_pos, out);
    }
    free(write_buffer);
    free(data_buf);
    free(encoded);

    fclose(out);

    printf("Encoding complete.\n");
    printf("Packed file stored at: %s\n", txtd_path);

    // Uncomment to enable timing
    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
