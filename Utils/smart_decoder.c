#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// --- CSV decode_nibble and infer_delimiter ---
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

// --- TXT decode logic ---
void decode_txt(FILE *in, FILE *out) {
    // Already skipped first byte and newline
    unsigned char byte;
    while (fread(&byte, 1, 1, in) == 1) {
        unsigned char high = (byte >> 4) & 0x0F;
        unsigned char low  = byte & 0x0F;

        if (high == 0b1111) break;
        fputc(decode_nibble(high), out);

        if (low == 0b1111) break;
        fputc(decode_nibble(low), out);
    }
}

// --- CSV decode logic ---
void decode_csv(FILE *in, FILE *out) {
    // Already skipped first byte and newline
    // Read header line (as-is, until newline)
    char header[8192];
    if (!fgets(header, sizeof(header), in)) {
        fprintf(stderr, "Failed to read header from txtd file.\n");
        return;
    }
    fputs(header, out);

    // Infer delimiter from header
    char delimiter = infer_delimiter(header);

    // If header does not end with newline, consume next char (should be newline)
    if (header[strlen(header)-1] != '\n') {
        int ch = fgetc(in);
        if (ch != '\n' && ch != EOF) ungetc(ch, in);
        fputc('\n', out);
    }

    // Now decode the rest of the file (encoded body)
    unsigned char byte;
    int stop = 0;
    while (!stop && fread(&byte, 1, 1, in) == 1) {
        unsigned char nibbles[2];
        nibbles[0] = (byte & 0xF0) >> 4;
        nibbles[1] = (byte & 0x0F);

        for (int i = 0; i < 2; ++i) {
            if (nibbles[i] == 0xF) {
                stop = 1;
                break;
            }
            char ch = decode_nibble(nibbles[i]);
            if (ch == '\0') continue;
            if (ch == ',' && delimiter != ',') ch = delimiter;
            fputc(ch, out);
        }
        if (stop) break;
    }
}

// --- Infer output file name and extension ---
void infer_output_name(const char *input, char *output, size_t outsize, int is_csv) {
    // Remove .txtd extension if present, else just append
    const char *dot = strrchr(input, '.');
    if (dot && strcmp(dot, ".txtd") == 0) {
        size_t len = dot - input;
        if (len > outsize - 5) len = outsize - 5;
        strncpy(output, input, len);
        output[len] = '\0';
    } else {
        strncpy(output, input, outsize-5);
        output[outsize-5] = '\0';
    }
    strcat(output, is_csv ? ".csv" : ".txt");
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input.txtd>\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror("Failed to open input txtd file");
        return 1;
    }

    // Read first byte to determine format
    int first = fgetc(in);
    int newline = fgetc(in); // skip newline

    char output[512];
    if (first == 0xFF) {
        infer_output_name(argv[1], output, sizeof(output), 1);
        FILE *out = fopen(output, "w");
        if (!out) {
            perror("Failed to create output file");
            fclose(in);
            return 1;
        }
        decode_csv(in, out);
        fclose(out);
        printf("Decoded as CSV to %s\n", output);
    } else if (first == 0x00) {
        infer_output_name(argv[1], output, sizeof(output), 0);
        FILE *out = fopen(output, "w");
        if (!out) {
            perror("Failed to create output file");
            fclose(in);
            return 1;
        }
        decode_txt(in, out);
        fclose(out);
        printf("Decoded as TXT to %s\n", output);
    } else {
        fprintf(stderr, "Unknown format: first byte is 0x%02X\n", first);
        fclose(in);
        return 1;
    }

    fclose(in);
    return 0;
}
