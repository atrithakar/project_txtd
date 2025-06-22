#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <openssl/sha.h>
#include <time.h>
#include <omp.h>

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
        default:
            fprintf(stderr, "Invalid character: '%c'\n", c);
            exit(EXIT_FAILURE);
    }
}

void encode_mt(const char *input_filename, const char *output_filename) {
    FILE *in = fopen(input_filename, "rb");
    if (!in) { perror("Failed to open input file"); exit(EXIT_FAILURE); }
    fseek(in, 0, SEEK_END);
    size_t n = ftell(in);
    fseek(in, 0, SEEK_SET);

    char *buf = (char*)malloc(n);
    fread(buf, 1, n, in);
    fclose(in);

    unsigned char *codes = (unsigned char*)malloc(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        codes[i] = encode_char(buf[i]);
    }

    FILE *out = fopen(output_filename, "wb");
    unsigned char byte = 0;
    int half = 0;
    for (size_t i = 0; i < n; ++i) {
        if (half == 0) { byte = codes[i] << 4; half = 1; }
        else { byte |= codes[i]; fwrite(&byte, 1, 1, out); half = 0; }
    }
    if (half == 0) { byte = 0b1111 << 4; fwrite(&byte, 1, 1, out); }
    else { byte |= 0b1111; fwrite(&byte, 1, 1, out); }
    fclose(out);

    free(buf); free(codes);
    printf("Encoded to %s\n", output_filename);
}

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

    char encoded_path[512], checksum_path[512];
    snprintf(encoded_path, sizeof(encoded_path), "%s/%s.encoded.txtd", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    encode_mt(argv[1], encoded_path);

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
