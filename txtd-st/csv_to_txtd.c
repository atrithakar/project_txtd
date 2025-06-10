#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <openssl/sha.h>

// Same encode_char as in encoder.c
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
            return 0xFF;  // invalid character
    }
}

// Utility: get file name without extension
void get_basename(char *basename, const char *path) {
    const char *last_slash = strrchr(path, '/');
    const char *filename = last_slash ? last_slash + 1 : path;
    strcpy(basename, filename);
    char *dot = strrchr(basename, '.');
    if (dot) *dot = '\0';
}

// List of common delimiters
const char COMMON_DELIMS[] = {',', ';', '\t', '|', '^', '~', 0};

// Utility: check if character is a common delimiter
int is_common_delim(char c) {
    for (int i = 0; COMMON_DELIMS[i]; ++i) {
        if (c == COMMON_DELIMS[i]) return 1;
    }
    return 0;
}

// Improved delimiter detection: only count common delimiters outside quotes
char detect_delimiter(const char *line) {
    int counts[256] = {0};
    int in_quote = 0;
    for (size_t i = 0; line[i] && line[i] != '\n'; ++i) {
        char c = line[i];
        if (c == '"' || c == '\'') {
            in_quote = !in_quote;
            continue;
        }
        if (in_quote) continue;
        if (is_common_delim(c)) {
            counts[(unsigned char)c]++;
        }
    }
    // Find the most frequent common delimiter
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

// Utility: detect delimiter from header line
char detect_delimiter_old(const char *line) {
    int counts[256] = {0};
    int in_quote = 0;
    for (size_t i = 0; line[i] && line[i] != '\n'; ++i) {
        char c = line[i];
        if (c == '"' || c == '\'') {
            in_quote = !in_quote;
            continue;
        }
        if (in_quote) continue;
        // Ignore spaces, tabs, alphanumerics, dot
        if (c == ' ' || c == '\t' || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '.') continue;
        counts[(unsigned char)c]++;
    }
    // Find the most frequent non-zero character
    int max_count = 0;
    char delimiter = ',';
    for (int i = 0; i < 256; ++i) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            delimiter = (char)i;
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
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.csv\n", argv[0]);
        return 1;
    }

    // Remove trailing slash if present
    char input_path[512];
    strncpy(input_path, argv[1], sizeof(input_path)-1);
    input_path[sizeof(input_path)-1] = '\0';
    size_t len = strlen(input_path);
    while (len > 0 && (input_path[len-1] == '/' || input_path[len-1] == '\\')) {
        input_path[len-1] = '\0';
        len--;
    }

    // Check if input_path is a directory
    FILE *test = fopen(input_path, "r");
    if (!test) {
        // Try to append a filename if a directory was given
        char try_path[600];
        snprintf(try_path, sizeof(try_path), "%s.csv", input_path);
        test = fopen(try_path, "r");
        if (!test) {
            perror("Failed to open input CSV file");
            return 1;
        }
        strncpy(input_path, try_path, sizeof(input_path)-1);
        input_path[sizeof(input_path)-1] = '\0';
    }
    fclose(test);

    FILE *in = fopen(input_path, "r");
    if (!in) {
        perror("Failed to open input CSV file");
        return 1;
    }

    char basename[256];
    get_basename(basename, input_path);

    // Create directory with csv name
    if (mkdir(basename) != 0 && errno != EEXIST) {
        perror("Failed to create output directory");
        fclose(in);
        return 1;
    }

    char header_path[300], data_path[300], delim_path[300], checksum_path[300];
    snprintf(header_path, sizeof(header_path), "%s/%s.header.txt", basename, basename);
    snprintf(data_path, sizeof(data_path), "%s/%s.data.txtd", basename, basename);
    snprintf(delim_path, sizeof(delim_path), "%s/%s.delimeter.txt", basename, basename);
    snprintf(checksum_path, sizeof(checksum_path), "%s/%s.checksum.txt", basename, basename);

    // Write checksum of original CSV file before encoding
    if (write_checksum(input_path, checksum_path) != 0) {
        fprintf(stderr, "Failed to write checksum file\n");
        fclose(in);
        return 1;
    }

    FILE *header = NULL;
    FILE *data = fopen(data_path, "wb");
    FILE *delim_file = NULL;
    if (!data) {
        perror("Failed to create data.txtd");
        fclose(in);
        return 1;
    }

    char line[8192];
    int is_first_line = 1;
    int half = 0;
    unsigned char buffer = 0;
    char delimiter = 0;

    while (fgets(line, sizeof(line), in)) {
        if (is_first_line) {
            delimiter = detect_delimiter(line);
            // Write delimiter to file
            delim_file = fopen(delim_path, "w");
            if (!delim_file) {
                perror("Failed to write delimiter");
                fclose(in);
                fclose(data);
                return 1;
            }
            fputc(delimiter, delim_file);
            fclose(delim_file);

            // Write header as-is
            header = fopen(header_path, "w");
            if (!header) {
                perror("Failed to write header");
                fclose(in);
                fclose(data);
                return 1;
            }
            fputs(line, header);
            fclose(header);

            is_first_line = 0;
            continue;  // skip header from encoding
        }

        int in_quote = 0;
        for (size_t i = 0; i < strlen(line); ++i) {
            char c = line[i];
            if (c == '"' || c == '\'') {
                in_quote = !in_quote;
            }
            // If not in quotes and c is a common delimiter, treat as comma
            if (!in_quote && is_common_delim(c)) {
                c = ',';
            }
            unsigned char val = encode_char(c);
            if (val == 0xFF) {
                fprintf(stderr, "Unexpected invalid character '%c' in line: %s\n", line[i], line);
                fclose(in);
                fclose(data);
                return 1;
            }

            if (half == 0) {
                buffer = val << 4;
                half = 1;
            } else {
                buffer |= val;
                fwrite(&buffer, 1, 1, data);
                half = 0;
            }
        }
    }

    if (half == 0) {
        buffer = 0b1111 << 4;
        fwrite(&buffer, 1, 1, data);
    } else {
        buffer |= 0b1111;
        fwrite(&buffer, 1, 1, data);
    }

    fclose(in);
    fclose(data);

    printf("Encoding complete.\n");
    printf("Delimiter stored at: %s\n", delim_path);
    printf("Header stored at: %s\n", header_path);
    printf("Encoded data stored at: %s\n", data_path);
    printf("Checksum file stored at: %s\n", checksum_path);
    return 0;
}
