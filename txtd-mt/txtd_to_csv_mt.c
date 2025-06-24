#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <openssl/sha.h>
#include <omp.h>
#include <time.h>

#define WRITE_BUFFER_SIZE 65536  // 64KB output buffer

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
        default: return 0xFF;
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
    if (len > 0 && out_hex[len-1] == '\n') out_hex[len-1] = '\0';
    fclose(in);
    return 0;
}

// Infer delimiter from header line
char infer_delimiter(const char *line) {
    const char COMMON_DELIMS[] = {',', ';', '\t', '|', '^', '~', 0};
    int counts[256] = {0};
    int in_quote = 0;
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
    char txtd_path[300], csv_path[300], checksum_path[300];
    snprintf(txtd_path, sizeof(txtd_path), "%s/%s.txtd", folder, basename);
    snprintf(csv_path, sizeof(csv_path), "%s/%s.csv", folder, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", folder, basename);

    FILE *in = fopen(txtd_path, "rb");
    if (!in) {
        perror("Failed to open packed txtd file");
        return 1;
    }
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror("Failed to create output CSV file");
        fclose(in);
        return 1;
    }

    // Skip first byte (0xFF), then skip the newline
    fgetc(in);
    fgetc(in);

    // Read header line (as-is, until newline)
    char header[8192];
    if (!fgets(header, sizeof(header), in)) {
        fprintf(stderr, "Failed to read header from txtd file.\n");
        fclose(in); fclose(csv);
        return 1;
    }
    fputs(header, csv);

    // Infer delimiter from header
    char delimiter = infer_delimiter(header);

    // If header does not end with newline, consume next char (should be newline)
    if (header[strlen(header)-1] != '\n') {
        int ch = fgetc(in);
        if (ch != '\n' && ch != EOF) ungetc(ch, in);
        fputc('\n', csv);
    }

    // Read and decode the rest of the file
    // Read all bytes into memory for parallel decode
    fseek(in, 0, SEEK_END);
    size_t file_end = ftell(in);
    long data_start = ftell(in);
    fseek(in, 0, SEEK_SET);
    // Skip preamble, newline, and header
    fgetc(in); // 0xFF
    fgetc(in); // newline
    while (1) {
        int c = fgetc(in);
        if (c == EOF) break;
        if (c == '\n') break;
    }
    long data_offset = ftell(in);
    size_t n_bytes = file_end - data_offset;
    if (n_bytes == 0) {
        fclose(in); fclose(csv);
        printf("CSV reconstructed at: %s\n", csv_path);
        printf("No encoded data found.\n");
        return 0;
    }
    fseek(in, data_offset, SEEK_SET);

    unsigned char *buf = (unsigned char*)malloc(n_bytes);
    fread(buf, 1, n_bytes, in);
    fclose(in);

    size_t n_nibbles = n_bytes * 2;
    char *outbuf = (char*)malloc(n_nibbles);

    #pragma omp parallel for
    for (size_t i = 0; i < n_nibbles; ++i) {
        unsigned char byte = buf[i/2];
        unsigned char nibble = (i % 2 == 0) ? (byte >> 4) & 0x0F : byte & 0x0F;
        outbuf[i] = decode_nibble(nibble);
    }

    char *write_buf = (char*)malloc(WRITE_BUFFER_SIZE);
    size_t write_pos = 0;

    for (size_t i = 0; i < n_nibbles; ++i) {
        if ((unsigned char)outbuf[i] == 0x0F || outbuf[i] == '\0') break;
        char ch = outbuf[i];
        if (ch == ',' && delimiter != ',') ch = delimiter;
        write_buf[write_pos++] = ch;
        if (write_pos == WRITE_BUFFER_SIZE) {
            fwrite(write_buf, 1, WRITE_BUFFER_SIZE, csv);
            write_pos = 0;
        }
    }
    if (write_pos > 0) fwrite(write_buf, 1, write_pos, csv);

    fclose(csv);
    free(buf); free(outbuf); free(write_buf);

    printf("CSV reconstructed at: %s\n", csv_path);

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

    clock_t end = clock();
    printf("Elapsed: %.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
