#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <windows.h>

#define BUFFER_SIZE 1048576  // 1 MiB

unsigned long long parse_size(const char *input) {
    double number;
    char unit[5] = {0};
    if (sscanf(input, "%lf%4s", &number, unit) < 1) {
        fprintf(stderr, "Invalid input format.\n");
        exit(1);
    }

    for (int i = 0; unit[i]; i++) unit[i] = toupper(unit[i]);

    if (strcmp(unit, "B") == 0 || strcmp(unit, "") == 0)
        return (unsigned long long)(number);
    else if (strcmp(unit, "KB") == 0)
        return (unsigned long long)(number * 1000);
    else if (strcmp(unit, "MB") == 0)
        return (unsigned long long)(number * 1000 * 1000);
    else if (strcmp(unit, "GB") == 0)
        return (unsigned long long)(number * 1000 * 1000 * 1000);
    else if (strcmp(unit, "KIB") == 0)
        return (unsigned long long)(number * 1024);
    else if (strcmp(unit, "MIB") == 0)
        return (unsigned long long)(number * 1024 * 1024);
    else if (strcmp(unit, "GIB") == 0)
        return (unsigned long long)(number * 1024 * 1024 * 1024);
    else {
        fprintf(stderr, "Unknown unit: %s\n", unit);
        exit(1);
    }
}

unsigned long long get_free_space(const char *path) {
    ULARGE_INTEGER free_bytes_available;
    if (!GetDiskFreeSpaceEx(path, &free_bytes_available, NULL, NULL)) {
        fprintf(stderr, "Failed to get disk space info.\n");
        exit(1);
    }
    return (unsigned long long)free_bytes_available.QuadPart;
}

int main() {
    char input[50];
    printf("Enter desired file size (e.g., 1MiB, 100KB, 2GiB): ");
    if (!fgets(input, sizeof(input), stdin)) {
        fprintf(stderr, "Input error.\n");
        return 1;
    }

    input[strcspn(input, "\n")] = '\0';  // Remove newline

    unsigned long long target_size = parse_size(input);
    unsigned long long free_space = get_free_space(".");

    if (target_size > free_space) {
        fprintf(stderr, "Not enough disk space. Available: %.2f MiB, Required: %.2f MiB\n",
                free_space / (1024.0 * 1024.0), target_size / (1024.0 * 1024.0));
        return 1;
    }

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
    unsigned long long written = 0;

    while (written < target_size) {
        size_t to_write = (target_size - written < BUFFER_SIZE) ? (size_t)(target_size - written) : BUFFER_SIZE;
        for (size_t i = 0; i < to_write; ++i) {
            buffer[i] = '0' + (rand() % 10);
        }
        fwrite(buffer, 1, to_write, fp);
        written += to_write;
    }

    printf("Successfully generated %s random digit file: random_digits.txt\n", input);

    free(buffer);
    fclose(fp);
    return 0;
}
