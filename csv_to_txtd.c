#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

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

int main(int argc, char *argv[]) {
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
    if (mkdir(basename) != 0 && errno != EEXIST) {
        perror("Failed to create output directory");
        fclose(in);
        return 1;
    }

    char header_path[300], data_path[300];
    snprintf(header_path, sizeof(header_path), "%s/%s.header.txt", basename, basename);
    snprintf(data_path, sizeof(data_path), "%s/%s.data.txtd", basename, basename);

    FILE *header = NULL;
    FILE *data = fopen(data_path, "wb");
    if (!data) {
        perror("Failed to create data.txtd");
        fclose(in);
        return 1;
    }

    char line[8192];
    int is_first_line = 1;
    int half = 0;
    unsigned char buffer = 0;

    while (fgets(line, sizeof(line), in)) {
        if (is_first_line) {
            int has_invalid = 0;
            for (size_t i = 0; i < strlen(line); ++i) {
                if (encode_char(line[i]) == 0xFF) {
                    has_invalid = 1;
                    break;
                }
            }

            if (has_invalid) {
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
            // if no invalid characters, encode it normally
        }

        for (size_t i = 0; i < strlen(line); ++i) {
            unsigned char val = encode_char(line[i]);
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

        is_first_line = 0;
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
    if (header) {
        printf("Header stored at: %s\n", header_path);
    }
    printf("Encoded data stored at: %s\n", data_path);
    return 0;
}
