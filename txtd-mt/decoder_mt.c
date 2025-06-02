#include "decoder_mt.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

void decode_mt(const char *input_filename, const char *output_filename) {
    FILE *in = fopen(input_filename, "rb");
    if (!in) { perror("Failed to open input file"); exit(EXIT_FAILURE); }
    fseek(in, 0, SEEK_END);
    size_t n = ftell(in);
    fseek(in, 0, SEEK_SET);

    unsigned char *buf = (unsigned char*)malloc(n);
    fread(buf, 1, n, in);
    fclose(in);

    char *outbuf = (char*)malloc(n * 2);
    size_t outlen = 0;

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        unsigned char high = (buf[i] >> 4) & 0x0F;
        unsigned char low  = buf[i] & 0x0F;
        if (high != 0xF) outbuf[i*2] = decode_nibble(high);
        else outbuf[i*2] = '\0';
        if (low != 0xF) outbuf[i*2+1] = decode_nibble(low);
        else outbuf[i*2+1] = '\0';
    }

    FILE *out = fopen(output_filename, "w");
    for (size_t i = 0; i < n * 2; ++i) {
        if (outbuf[i] == '\0') break;
        fputc(outbuf[i], out);
    }
    fclose(out);

    free(buf); free(outbuf);
    printf("Decoded to %s\n", output_filename);
}
