#include <stdio.h>
#include <stdlib.h>

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

void decode(const char *input_filename, const char *output_filename) {
    FILE *in = fopen(input_filename, "rb");
    if (!in) {
        perror("Failed to open input file");
        exit(EXIT_FAILURE);
    }

    FILE *out = fopen(output_filename, "w");
    if (!out) {
        perror("Failed to create output file");
        fclose(in);
        exit(EXIT_FAILURE);
    }

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
    printf("Decoded to %s\n", output_filename);
}
