#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to map digit characters to 4-bit values
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
        default:
            fprintf(stderr, "Invalid character: %c\n", c);
            exit(EXIT_FAILURE);
    }
}

int main() {
    char input[100000];  // for iteration 1, buffer limit is fine
    printf("Enter numeric input (digits and '.'): ");
    scanf("%s", input);

    FILE *out = fopen("output.txtd", "wb");
    if (!out) {
        perror("Failed to open output.txtd");
        return 1;
    }

    int len = strlen(input);
    unsigned char buffer = 0;
    int half = 0;  // 0 means high nibble empty, 1 means low nibble filled

    for (int i = 0; i < len; ++i) {
        unsigned char val = encode_char(input[i]);

        if (half == 0) {
            buffer = val << 4;  // store in high nibble
            half = 1;
        } else {
            buffer |= val;      // fill low nibble
            fwrite(&buffer, 1, 1, out);  // write full byte
            half = 0;
        }
    }

    // if an odd number of digits, flush remaining nibble
    if (half == 1) {
        fwrite(&buffer, 1, 1, out);
    }

    fclose(out);
    printf("Successfully written to output.txtd\n");

    return 0;
}
