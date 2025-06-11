#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
        case ',' : return 0b1110;
        case '\0': return 0b1111;
        default:
            fprintf(stderr, "Invalid character: '%c'\n", c);
            exit(EXIT_FAILURE);
    }
}

void encode_mt(const char *input_filename, const char *output_filename) {
    FILE *in = fopen(input_filename, "rb");
    if (!in) { perror("Failed to open input file"); exit(EXIT_FAILURE); }
    fseek(in, 0, SEEK_END);
    size_t n = ftell(in);
    fseek(in, 0, SEEK_SET);

    char *buf = (char*)malloc(n);
    fread(buf, 1, n, in);
    fclose(in);

    unsigned char *codes = (unsigned char*)malloc(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        codes[i] = encode_char(buf[i]);
    }

    FILE *out = fopen(output_filename, "wb");
    unsigned char byte = 0;
    int half = 0;
    for (size_t i = 0; i < n; ++i) {
        if (half == 0) { byte = codes[i] << 4; half = 1; }
        else { byte |= codes[i]; fwrite(&byte, 1, 1, out); half = 0; }
    }
    if (half == 0) { byte = 0b1111 << 4; fwrite(&byte, 1, 1, out); }
    else { byte |= 0b1111; fwrite(&byte, 1, 1, out); }
    fclose(out);

    free(buf); free(codes);
    printf("Encoded to %s\n", output_filename);
}
