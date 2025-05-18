#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        case '\0': return 0b1111;
        default:
            fprintf(stderr, "Invalid character: '%c'\n", c);
            exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.txt outputName\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) {
        perror("Failed to open input file");
        return 1;
    }

    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s.txtd", argv[2]);

    FILE *out = fopen(output_filename, "wb");
    if (!out) {
        perror("Failed to create output file");
        fclose(in);
        return 1;
    }

    unsigned char buffer = 0;
    int half = 0;
    int ch;

    while ((ch = fgetc(in)) != EOF) {
        unsigned char val = encode_char((char)ch);
        if (half == 0) {
            buffer = val << 4;
            half = 1;
        } else {
            buffer |= val;
            fwrite(&buffer, 1, 1, out);
            half = 0;
        }
    }

    if (half == 0) {
        buffer = 0b1111 << 4;
        fwrite(&buffer, 1, 1, out);
    } else {
        buffer |= 0b1111;
        fwrite(&buffer, 1, 1, out);
    }

    fclose(in);
    fclose(out);
    printf("Encoded to %s\n", output_filename);
    return 0;
}
