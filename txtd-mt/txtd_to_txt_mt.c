#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include <time.h>
#include <omp.h>

#define WRITE_BUFFER_SIZE 65536  // 64KB output buffer

// Map 4-bit value to character
char decode_nibble(unsigned char nibble) {
    switch (nibble) {
        case 0b0000: return '0';
        case 0b0001: return '1';
        case 0b0010: return '2';
        case 0b0011: return '3';
        case 0b0100: return '4';
        case 0b0101: return '5';
        case 0b0110: return '6';
        case 0b0111: return '7';
        case 0b1000: return '8';
        case 0b1001: return '9';
        case 0b1010: return '.';
        case 0b1011: return ' ';
        case 0b1100: return '\t';
        case 0b1101: return '\n';
        case 0b1110: return ',';
        case 0b1111: return '\0';
        default:
            fprintf(stderr, "Invalid nibble in file: 0x%X\n", nibble);
            exit(EXIT_FAILURE);
    }
}

void decode_mt(const char *input_filename, const char *output_filename) {
    FILE *in = fopen(input_filename, "rb");
    if (!in) { perror("Failed to open input file"); exit(EXIT_FAILURE); }

    // Skip first two bytes (header and newline)
    fgetc(in);
    fgetc(in);

    fseek(in, 0, SEEK_END);
    size_t n = ftell(in) - 2;
    fseek(in, 2, SEEK_SET);

    unsigned char *buf = (unsigned char*)malloc(n);
    fread(buf, 1, n, in);
    fclose(in);

    char *outbuf = (char*)malloc(n * 2);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        unsigned char high = (buf[i] >> 4) & 0x0F;
        unsigned char low  = buf[i] & 0x0F;
        outbuf[i * 2]     = (high != 0xF) ? decode_nibble(high) : '\0';
        outbuf[i * 2 + 1] = (low  != 0xF) ? decode_nibble(low)  : '\0';
    }

    FILE *out = fopen(output_filename, "w");
    if (!out) { perror("Failed to open output file"); exit(EXIT_FAILURE); }

    char write_buffer[WRITE_BUFFER_SIZE];
    size_t buf_index = 0;

    for (size_t i = 0; i < n * 2; ++i) {
        if (outbuf[i] == '\0') break;
        write_buffer[buf_index++] = outbuf[i];
        if (buf_index == WRITE_BUFFER_SIZE) {
            fwrite(write_buffer, 1, buf_index, out);
            buf_index = 0;
        }
    }

    if (buf_index > 0) {
        fwrite(write_buffer, 1, buf_index, out);
    }

    fclose(out);
    free(buf);
    free(outbuf);

    printf("Decoded to %s\n", output_filename);
}

// Compute SHA-256 checksum of file to hex
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

// Read checksum from file
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

int main(int argc, char *argv[]) {
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder_name>\n", argv[0]);
        return 1;
    }

    char *folder = argv[1];
    char encoded_path[512], checksum_path[512], decoded_path[512], basename[256];

    strncpy(basename, folder, sizeof(basename) - 1);
    basename[sizeof(basename) - 1] = '\0';

    snprintf(encoded_path, sizeof(encoded_path), "%s/%s.encoded.txtd", folder, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", folder, basename);
    snprintf(decoded_path, sizeof(decoded_path), "%s/%s_decoded.txt", folder, basename);

    decode_mt(encoded_path, decoded_path);

    char decoded_checksum[SHA256_DIGEST_LENGTH * 2 + 1];
    if (compute_checksum(decoded_path, decoded_checksum, sizeof(decoded_checksum)) != 0) {
        fprintf(stderr, "Failed to compute checksum for %s\n", decoded_path);
        return 1;
    }

    char original_checksum[SHA256_DIGEST_LENGTH * 2 + 1];
    if (read_checksum(checksum_path, original_checksum, sizeof(original_checksum)) != 0) {
        fprintf(stderr, "Failed to read checksum file %s\n", checksum_path);
        return 1;
    }

    if (strcmp(decoded_checksum, original_checksum) == 0) {
        printf("Decoding completed successfully. The decoded file matches the original checksum.\n");
    } else {
        printf("Decoding completed, but the decoded file does NOT match the original checksum. Please verify the integrity of your files.\n");
    }

    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
