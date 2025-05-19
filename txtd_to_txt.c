#include <stdio.h>
#include "decoder.c"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txtd> <output.txt>\n", argv[0]);
        return 1;
    }

    // Decode the input binary file to a text format
    decode(argv[1], argv[2]);

    printf("Encoding and decoding completed successfully.\n");
    return 0;
}