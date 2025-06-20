# Usage

> **Note:** The usage examples below demonstrate how to use the `txtd-st` (single-threaded CPU) implementation.  
> To use the `txtd-mt` (multithreaded CPU) or `txtd-cuda` (GPU-accelerated) versions, simply add the appropriate suffix (`_mt` or `_cuda`) to the executable file names shown.  
> This assumes you have followed the [Build Guide](https://github.com/atrithakar/project_txtd/blob/main/MDs/BUILD.md) and used the recommended naming conventions.  
> 
> If you’ve used custom names while compiling, make sure to substitute the executable names accordingly when running the commands.


## Encode a TXT file
```
path/to/txt_to_txtd.exe path/to/txt_name.txt
```

## Decode a TXTD file into TXT file
```
path/to/txtd_to_txt.exe path/to/encoded_txt_folder
```

## Encode a CSV file (only numerical data in the body and allowed delimiters)
```
path/to/csv_to_txtd.exe path/to/csv_name.csv
```

## Decode a TXTD file into CSV file
```
path/to/txtd_to_csv.exe path/to/encoded_csv_folder
```

> **Note:** The output folder—containing the encoded file and all necessary metadata—will be created in the directory from which you run the command in the terminal.  
> Ensure you execute the command from your intended target directory to avoid confusion about where the output is generated.

## Example

If you want to, for example, use the multithreaded (txtd-mt) version to encode a `.txt` file, follow these steps (assuming you’ve followed the [Build Guide](https://github.com/atrithakar/project_txtd/blob/main/MDs/BUILD.md) and kept the default naming scheme):

1. Start with the standard single-threaded command:  
   `path/to/txt_to_txtd.exe path/to/txt_name.txt`

2. Add the `_mt` suffix to use the multithreaded version:  
   `path/to/txt_to_txtd_mt.exe path/to/txt_name.txt`

3. Run the updated command in your terminal.

> Similarly you can use txtd-cuda, but with `_cuda` suffix instead of `_mt` suffix