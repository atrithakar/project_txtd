#include <stdio.h>
#include "encoder.c"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txt> <output.txtd>\n", argv[0]);
        return 1;
    }

    // Encode the input text file to a binary format
    encode(argv[1], argv[2]);



    printf("Encoding and decoding completed successfully.\n");
    return 0;
}