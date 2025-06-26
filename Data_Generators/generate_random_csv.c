#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <windows.h>

#define BUFFER_SIZE 1048576  // 1 MiB
#define MAX_COLUMNS 16384    // Sanity cap for Excel CSV

unsigned long long parse_size(const char *input) {
    double number;
    char unit[5] = {0};
    if (sscanf(input, "%lf%4s", &number, unit) < 1) {
        fprintf(stderr, "Error: Invalid file size format.\n");
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
        fprintf(stderr, "Error: Unknown unit '%s'.\n", unit);
        exit(1);
    }
}

unsigned long long get_free_space(const char *path) {
    ULARGE_INTEGER free_bytes_available;
    if (!GetDiskFreeSpaceEx(path, &free_bytes_available, NULL, NULL)) {
        fprintf(stderr, "Error: Failed to get disk space info.\n");
        exit(1);
    }
    return (unsigned long long)free_bytes_available.QuadPart;
}

void random_word(char *buf, int max_len) {
    int len = 1 + rand() % max_len;
    for (int i = 0; i < len; ++i) {
        buf[i] = 'a' + rand() % 26;
    }
    buf[len] = '\0';
}

int main() {
    srand((unsigned int)time(NULL));

    // Get file size
    char size_input[50];
    printf("Enter desired file size (e.g. 10MiB, 1GB): ");
    if (!fgets(size_input, sizeof(size_input), stdin)) {
        fprintf(stderr, "Error: Failed to read input.\n");
        return 1;
    }
    size_input[strcspn(size_input, "\n")] = '\0';

    // Apply 10% buffer
    unsigned long long target_size = (unsigned long long)(parse_size(size_input) * 1.1);
    unsigned long long free_space = get_free_space(".");

    if (target_size > free_space) {
        fprintf(stderr, "Error: Not enough disk space. Available: %.2f MiB, Required: %.2f MiB\n",
                free_space / (1024.0 * 1024.0), target_size / (1024.0 * 1024.0));
        return 1;
    }

    int mode;
    printf("Select data configuration mode:\n");
    printf("  0 - Auto (3 columns, variable rows)\n");
    printf("  1 - Fixed number of rows (columns calculated)\n");
    printf("  2 - Fixed number of columns (rows calculated)\n");
    printf("Enter choice (0/1/2): ");
    if (scanf("%d", &mode) != 1 || mode < 0 || mode > 2) {
        fprintf(stderr, "Error: Invalid choice. Enter 0, 1, or 2.\n");
        return 1;
    }

    int fixed_rows = 0, fixed_cols = 3;
    const int avg_cell_size = 10; // estimated average bytes per cell with comma

    if (mode == 0) {
        fixed_cols = 3;
        fixed_rows = (int)(target_size / (fixed_cols * avg_cell_size));
        if (fixed_rows < 1) fixed_rows = 1;
        printf("Auto mode: Using 3 columns and approximately %d rows.\n", fixed_rows);
    } else if (mode == 1) {
        printf("Enter number of rows: ");
        if (scanf("%d", &fixed_rows) != 1 || fixed_rows <= 0) {
            fprintf(stderr, "Error: Invalid row count.\n");
            return 1;
        }
        fixed_cols = (int)(target_size / (fixed_rows * avg_cell_size));
        if (fixed_cols < 1) fixed_cols = 1;
        if (fixed_cols > MAX_COLUMNS) fixed_cols = MAX_COLUMNS;
        printf("Calculated column count: %d\n", fixed_cols);
    } else if (mode == 2) {
        printf("Enter number of columns: ");
        if (scanf("%d", &fixed_cols) != 1 || fixed_cols <= 0 || fixed_cols > MAX_COLUMNS) {
            fprintf(stderr, "Error: Invalid column count.\n");
            return 1;
        }
        fixed_rows = (int)(target_size / (fixed_cols * avg_cell_size));
        if (fixed_rows < 1) fixed_rows = 1;
        printf("Calculated row count: %d\n", fixed_rows);
    }

    FILE *fp = fopen("random_data.csv", "w");
    if (!fp) {
        perror("Error: Failed to open output file");
        return 1;
    }

    char *buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Error: Memory allocation failed");
        fclose(fp);
        return 1;
    }

    // Write header
    for (int i = 0; i < fixed_cols; ++i) {
        char word[21];
        random_word(word, 20);
        fprintf(fp, "%s%s", word, (i == fixed_cols - 1) ? "\n" : ",");
    }

    size_t total_written = ftell(fp);
    int rows_written = 0;

    while ((mode == 0 && total_written < target_size) ||
           ((mode == 1 || mode == 2) && rows_written < fixed_rows)) {
        size_t buffer_pos = 0;

        while (buffer_pos < BUFFER_SIZE - (fixed_cols * 12)) {
            if ((mode == 1 || mode == 2) && rows_written >= fixed_rows)
                break;

            char *row = malloc(fixed_cols * 16);
            if (!row) {
                fprintf(stderr, "Error: Failed to allocate row buffer.\n");
                free(buffer);
                fclose(fp);
                return 1;
            }

            int pos = 0;
            for (int j = 0; j < fixed_cols; ++j) {
                double val = (rand() % 10000) / 100.0 + (rand() % 10000) / 100000.0;
                pos += snprintf(row + pos, fixed_cols * 16 - pos, "%.5f%s", val, (j == fixed_cols - 1) ? "\n" : ",");
            }

            if (buffer_pos + pos >= BUFFER_SIZE) {
                free(row);
                break;
            }

            memcpy(buffer + buffer_pos, row, pos);
            buffer_pos += pos;
            rows_written++;

            free(row);
        }

        if (buffer_pos > 0)
            fwrite(buffer, 1, buffer_pos, fp);

        total_written += buffer_pos;
    }

    if (mode == 0) {
        printf("File has approximately %d rows and 3 columns.\n", rows_written);
    } else if (mode == 1) {
        printf("File has %d rows and approximately %d columns.\n", fixed_rows, fixed_cols);
    } else {
        printf("File has %d columns and %d rows.\n", fixed_cols, fixed_rows);
    }

    printf("CSV file 'random_data.csv' generated successfully. Size: %.2f MiB\n",
           total_written / (1024.0 * 1024.0));

    free(buffer);
    fclose(fp);
    return 0;
}
