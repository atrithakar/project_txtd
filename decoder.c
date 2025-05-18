#include <stdio.h>
#include <stdlib.h>

// Function to convert 4-bit value back to character
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
        case 0b1010: return '.';         // Decimal Symbol
        case 0b1011: return ' ';         // Space Symbol
        case 0b1100: return '\t';        // Tab Symbol
        case 0b1101: return '\n';        // Newline Symbol
        case 0b1111: return '\0';        // EOF Symbol (special handling)
        default:
            fprintf(stderr, "Invalid nibble in file: 0x%X\n", nibble);
            exit(EXIT_FAILURE);
    }
}

int main() {
    FILE *in = fopen("output.txtd", "rb");
    if (!in) {
        perror("Failed to open output.txtd");
        return 1;
    }

    unsigned char byte;
    while (fread(&byte, 1, 1, in) == 1) {
        unsigned char high = (byte >> 4) & 0x0F;
        unsigned char low  = byte & 0x0F;

        // Stop decoding if EOF nibble is hit
        if (high == 0b1111) break;
        printf("%c", decode_nibble(high));

        if (low == 0b1111) break;
        printf("%c", decode_nibble(low));
    }

    fclose(in);
    // printf("X");
    return 0;
}
