#include <stdio.h>
#include <stdlib.h>
#include <openssl/sha.h>
#include <string.h>


#define BUF_SIZE 32768

// Function to compute SHA-256 hash of a file
int hash_file(const char *filename, unsigned char *out_hash) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return 1;
    }

    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    unsigned char buffer[BUF_SIZE];
    size_t bytes_read = 0;

    while ((bytes_read = fread(buffer, 1, BUF_SIZE, file)) > 0) {
        SHA256_Update(&sha256, buffer, bytes_read);
    }

    fclose(file);

    SHA256_Final(out_hash, &sha256);
    return 0;
}

void print_hash(unsigned char *hash) {
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        printf("%02x", hash[i]);
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s file1 file2\n", argv[0]);
        return 1;
    }

    unsigned char hash1[SHA256_DIGEST_LENGTH];
    unsigned char hash2[SHA256_DIGEST_LENGTH];

    if (hash_file(argv[1], hash1) != 0) {
        fprintf(stderr, "Error hashing %s\n", argv[1]);
        return 1;
    }

    if (hash_file(argv[2], hash2) != 0) {
        fprintf(stderr, "Error hashing %s\n", argv[2]);
        return 1;
    }

    printf("Hash of %s: ", argv[1]);
    print_hash(hash1);

    printf("Hash of %s: ", argv[2]);
    print_hash(hash2);

    if (memcmp(hash1, hash2, SHA256_DIGEST_LENGTH) == 0) {
        printf("Files are IDENTICAL\n");
    } else {
        printf("Files are DIFFERENT\n");
    }

    return 0;
}
