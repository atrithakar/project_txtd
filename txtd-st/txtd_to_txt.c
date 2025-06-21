#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include "decoder.c"
#include <time.h>

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

    // Uncomment to enable timing
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder_name>\n", argv[0]);
        return 1;
    }

    char *folder = argv[1];
    char encoded_path[512], checksum_path[512], decoded_path[512], basename[256];

    // Get basename from folder name (assume folder name is the base)
    strncpy(basename, folder, sizeof(basename)-1);
    basename[sizeof(basename)-1] = '\0';

    snprintf(encoded_path, sizeof(encoded_path), "%s/%s.encoded.txtd", folder, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", folder, basename);
    snprintf(decoded_path, sizeof(decoded_path), "%s/%s_decoded.txt", folder, basename);

    // Open encoded file and skip first two bytes, then decode and write output
    FILE *in = fopen(encoded_path, "rb");
    if (!in) {
        perror("Failed to open encoded file");
        return 1;
    }
    FILE *out = fopen(decoded_path, "w");
    if (!out) {
        perror("Failed to create output file");
        fclose(in);
        return 1;
    }
    // Skip first two bytes (header and newline)
    fgetc(in);
    fgetc(in);

    // Decode the rest of the file
    unsigned char byte;
    while (fread(&byte, 1, 1, in) == 1) {
        unsigned char high = (byte >> 4) & 0x0F;
        unsigned char low  = byte & 0x0F;

        if (high == 0b1111) break;
        fputc(decode_nibble(high), out);

        if (low == 0b1111) break;
        fputc(decode_nibble(low), out);
    }
    fclose(in);
    fclose(out);

    // Compute checksum of decoded file
    char decoded_checksum[SHA256_DIGEST_LENGTH*2+1];
    if (compute_checksum(decoded_path, decoded_checksum, sizeof(decoded_checksum)) != 0) {
        fprintf(stderr, "Failed to compute checksum for %s\n", decoded_path);
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

    // Uncomment to enable timing
    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}