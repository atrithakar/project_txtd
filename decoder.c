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
        case 0b1010: return '.';
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

        // Decode and print both nibbles
        printf("%c", decode_nibble(high));

        // For odd number of characters, the last nibble may be padding — so we stop reading if EOF is hit
        if (!feof(in)) {
            printf("%c", decode_nibble(low));
        }
    }

    printf("\n");
    fclose(in);
    return 0;
}
