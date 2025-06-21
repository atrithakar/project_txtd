#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TARGET_SIZE_BYTES 1073741824  // 1 GiB = 1024 * 1024 * 1024 bytes
#define BUFFER_SIZE 1048576           // 1 MiB = 1024 * 1024 bytes

int main() {
    FILE *fp = fopen("random_digits.txt", "w");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    char *buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Failed to allocate buffer");
        fclose(fp);
        return 1;
    }

    srand((unsigned int)time(NULL));
    size_t written = 0;

    while (written < TARGET_SIZE_BYTES) {
        size_t to_write = (TARGET_SIZE_BYTES - written < BUFFER_SIZE) ? (TARGET_SIZE_BYTES - written) : BUFFER_SIZE;

        for (size_t i = 0; i < to_write; ++i) {
            buffer[i] = '0' + (rand() % 10);  // Random digit
        }

        fwrite(buffer, 1, to_write, fp);
        written += to_write;
    }

    printf("✅ Successfully generated 1 GiB random digit file: random_digits.txt\n");

    free(buffer);
    fclose(fp);
    return 0;
}
