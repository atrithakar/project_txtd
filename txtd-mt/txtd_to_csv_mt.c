#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <openssl/sha.h>
#include <omp.h>
#include <time.h>

// ...decode_nibble as in st version...
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
        case 0b1110: return ','; // will be replaced by delimiter
        case 0b1111: return '\0';
        default: return 0xFF;
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
    size_t len = strlen(out_hex);
    if (len > 0 && out_hex[len-1] == '\n') out_hex[len-1] = '\0';
    fclose(in);
    return 0;
}

int main(int argc, char *argv[]) {
    // Uncomment to enable timing
    clock_t start = clock();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
        return 1;
    }

    char folder[256];
    strncpy(folder, argv[1], sizeof(folder)-1);
    folder[sizeof(folder)-1] = '\0';

    // Remove trailing slash if present
    size_t len = strlen(folder);
    while (len > 0 && (folder[len-1] == '/' || folder[len-1] == '\\')) {
        folder[len-1] = '\0';
        len--;
    }

    // Extract basename (last component after / or \)
    char *basename = folder;
    char *slash = strrchr(folder, '/');
    #ifdef _WIN32
    char *bslash = strrchr(folder, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
    #endif
    if (slash) basename = slash + 1;

    // Compose file paths
    char header_path[300], data_path[300], delim_path[300], csv_path[300], checksum_path[300];
    snprintf(header_path, sizeof(header_path), "%s/%s.header.txt", folder, basename);
    snprintf(data_path, sizeof(data_path), "%s/%s.data.txtd", folder, basename);
    snprintf(delim_path, sizeof(delim_path), "%s/%s.delimeter.txt", folder, basename);
    snprintf(csv_path, sizeof(csv_path), "%s/%s.csv", folder, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", folder, basename);

    // Read delimiter
    FILE *delim_file = fopen(delim_path, "r");
    if (!delim_file) {
        char try_path[300];
        snprintf(try_path, sizeof(try_path), "%s.delimeter.txt", basename);
        delim_file = fopen(try_path, "r");
        if (!delim_file) {
            perror("Failed to open delimiter file");
            return 1;
        }
    }
    char delimiter = fgetc(delim_file);
    fclose(delim_file);

    // Read header
    FILE *header = fopen(header_path, "r");
    if (!header) {
        perror("Failed to open header file");
        return 1;
    }
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror("Failed to create output CSV file");
        fclose(header);
        return 1;
    }
    char line[8192];
    while (fgets(line, sizeof(line), header)) {
        fputs(line, csv);
    }
    fclose(header);

    // Read and decode data
    FILE *data = fopen(data_path, "rb");
    if (!data) {
        perror("Failed to open data file");
        fclose(csv);
        return 1;
    }
    fseek(data, 0, SEEK_END);
    size_t n_bytes = ftell(data);
    fseek(data, 0, SEEK_SET);
    unsigned char *buf = (unsigned char*)malloc(n_bytes);
    fread(buf, 1, n_bytes, data);
    fclose(data);

    size_t n_nibbles = n_bytes * 2;
    char *outbuf = (char*)malloc(n_nibbles);

    #pragma omp parallel for
    for (size_t i = 0; i < n_nibbles; ++i) {
        unsigned char byte = buf[i/2];
        unsigned char nibble = (i % 2 == 0) ? (byte >> 4) & 0x0F : byte & 0x0F;
        outbuf[i] = decode_nibble(nibble);
    }

    // Write output, replace ',' with delimiter, stop at first '\0'
    for (size_t i = 0; i < n_nibbles; ++i) {
        if ((unsigned char)outbuf[i] == 0x0F || outbuf[i] == '\0') break;
        char ch = outbuf[i];
        if (ch == ',' && delimiter != ',') ch = delimiter;
        fputc(ch, csv);
    }
    fclose(csv);
    free(buf); free(outbuf);

    printf("CSV reconstructed at: %s\n", csv_path);

    // Checksum verification
    char decoded_checksum[SHA256_DIGEST_LENGTH*2+1];
    if (compute_checksum(csv_path, decoded_checksum, sizeof(decoded_checksum)) != 0) {
        fprintf(stderr, "Failed to compute checksum for %s\n", csv_path);
        return 1;
    }

    char original_checksum[SHA256_DIGEST_LENGTH*2+1];
    if (read_checksum(checksum_path, original_checksum, sizeof(original_checksum)) != 0) {
        fprintf(stderr, "Failed to read checksum file %s\n", checksum_path);
        return 1;
    }

    if (strcmp(decoded_checksum, original_checksum) == 0) {
        printf("Decoding completed successfully. The decoded CSV matches the original checksum.\n");
    } else {
        printf("Decoding completed, but the decoded CSV does NOT match the original checksum. Please verify the integrity of your files.\n");
    }

    // Uncomment to enable timing
    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
