#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TARGET_SIZE_BYTES 1073741824  // 1 GiB
// #define TARGET_SIZE_BYTES 1048576  // 1 GiB
#define BUFFER_SIZE 1048576           // 1 MiB

int main() {
    FILE *fp = fopen("random_data.csv", "w");
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
    size_t total_written = 0;

    // Write header first
    const char *header = "Index,Height(Inches),Weight(Pounds)\n";
    fputs(header, fp);
    total_written += strlen(header);

    unsigned int index = 1;
    while (total_written < TARGET_SIZE_BYTES) {
        size_t buffer_pos = 0;

        while (buffer_pos < BUFFER_SIZE - 50) {  // leave room for a row
            int height = 55 + rand() % 26;       // height: 55 to 80
            int weight = 100 + rand() % 151;     // weight: 100 to 250
            int written = snprintf(buffer + buffer_pos, BUFFER_SIZE - buffer_pos,
                                   "%u,%d,%d\n", index++, height, weight);
            if (written <= 0) break;
            buffer_pos += written;
        }

        size_t written = fwrite(buffer, 1, buffer_pos, fp);
        total_written += written;
    }

    printf("✅ Generated random_data.csv — size: at least 1 GiB\n");

    free(buffer);
    fclose(fp);
    return 0;
}
