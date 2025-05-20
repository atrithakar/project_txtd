#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// Same decode_nibble as in decoder.c
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
        case 0b1110: return ','; // will be replaced by delimiter
        case 0b1111: return '\0';
        default: return 0xFF;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <folder>\n", argv[0]);
        return 1;
    }

    char folder[256];
    strncpy(folder, argv[1], sizeof(folder)-1);
    folder[sizeof(folder)-1] = '\0';

    // Compose file paths
    char header_path[300], data_path[300], delim_path[300], csv_path[300];
    snprintf(header_path, sizeof(header_path), "%s/%s.header.txt", folder, folder);
    snprintf(data_path, sizeof(data_path), "%s/%s.data.txtd", folder, folder);
    snprintf(delim_path, sizeof(delim_path), "%s/%s.delimeter.txt", folder, folder);
    snprintf(csv_path, sizeof(csv_path), "%s/%s.csv", folder, folder);

    // Read delimiter
    FILE *delim_file = fopen(delim_path, "r");
    if (!delim_file) {
        perror("Failed to open delimiter file");
        return 1;
    }
    char delimiter = fgetc(delim_file);
    fclose(delim_file);

    // Read header
    FILE *header = fopen(header_path, "r");
    if (!header) {
        perror("Failed to open header file");
        return 1;
    }
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror("Failed to create output CSV file");
        fclose(header);
        return 1;
    }
    char line[8192];
    while (fgets(line, sizeof(line), header)) {
        fputs(line, csv);
    }
    fclose(header);

    // Read and decode data
    FILE *data = fopen(data_path, "rb");
    if (!data) {
        perror("Failed to open data file");
        fclose(csv);
        return 1;
    }
    unsigned char byte;
    int stop = 0;
    while (!stop && fread(&byte, 1, 1, data) == 1) {
        unsigned char nibbles[2];
        nibbles[0] = (byte & 0xF0) >> 4;
        nibbles[1] = (byte & 0x0F);

        for (int i = 0; i < 2; ++i) {
            if (nibbles[i] == 0xF) {
                stop = 1;
                break;
            }
            char ch = decode_nibble(nibbles[i]);
            if (ch == '\0') continue;
            if (ch == ',' && delimiter != ',') ch = delimiter;
            fputc(ch, csv);
        }
        if (stop) break;
    }
    fclose(data);
    fclose(csv);

    printf("CSV reconstructed at: %s\n", csv_path);
    return 0;
}
