# restore_files.jl
# This script reads the concatenated file (concatenated_output.txt) and restores
# the original file and directory structure into a new folder named 'restored_directory'.

# --- Separator Constants (Must match the concatenation script) ---
const SEPARATOR_START = "\"// # FILE: "
const SEPARATOR_END   = "\";"

const INPUT_FILENAME = "concatenated_output.txt"
const RESTORE_DIR = "restored_directory"

function restore_files()
    if !isfile(INPUT_FILENAME)
        println("Error: Input file '$INPUT_FILENAME' not found in the current directory.")
        return
    end

    println("Starting restoration from '$INPUT_FILENAME'...")
    
    # 1. Read the entire concatenated content
    local content
    try
        content = read(INPUT_FILENAME, String)
    catch e
        println("Error reading input file: $e")
        return
    end

    # 2. Split the content by the unique start separator
    # The split removes SEPARATOR_START from the resulting strings.
    # The first element is usually the file header/preamble, so we might skip it in the loop.
    parts = split(content, SEPARATOR_START)
    
    # 3. Create the top-level restoration directory
    if isdir(RESTORE_DIR)
        println("Warning: Restoration directory '$RESTORE_DIR' already exists. Files might be overwritten.")
    else
        mkdir(RESTORE_DIR)
        println("Created restoration directory: '$RESTORE_DIR'")
    end

    # 4. Iterate over each part
    count = 0
    for part in parts
        # We look for the SEPARATOR_END (";) to identify if this block is valid
        # and where the path ends.
        end_marker_range = findfirst(SEPARATOR_END, part)

        if end_marker_range === nothing
            # This part does not contain the closing separator.
            # It's likely the text before the very first file (preamble).
            continue
        end

        # Extract the relative path (everything before the ";)
        # part starts immediately after SEPARATOR_START, so 1 to marker start is the path.
        path_end_index = prevind(part, end_marker_range.start)
        relative_path = strip(part[1:path_end_index])
        
        if isempty(relative_path)
            println("Warning: Found empty file path. Skipping block.")
            continue
        end

        # Extract the content (everything after the ";)
        content_start_index = nextind(part, end_marker_range.stop)
        file_content = part[content_start_index:end]

        # The concatenation script adds a newline after the separator for readability.
        # We should strip that specific leading newline so the file starts exactly as it was.
        if startswith(file_content, "\n")
            file_content = file_content[2:end]
        elseif startswith(file_content, "\r\n") # Windows safety
            file_content = file_content[3:end]
        end
        
        # Determine the full path for the new restored file
        restored_full_path = joinpath(RESTORE_DIR, relative_path)
        
        # Get the directory part of the restored path
        restored_dir = dirname(restored_full_path)
        
        # 5. Create the required sub-directories if they don't exist
        if !isdir(restored_dir)
            try
                mkpath(restored_dir) # mkpath creates all intermediate directories
                println("Created directory: $restored_dir")
            catch e
                println("Error creating directory $restored_dir: $e")
                continue
            end
        end

        # 6. Write the content to the new file
        try
            open(restored_full_path, "w") do outfile
                write(outfile, file_content)
            end
            println("Restored file: $relative_path")
            count += 1
        catch e
            println("Error writing file $relative_path: $e")
        end
    end

    println("Restoration complete! $count files restored to '$RESTORE_DIR'.")
end

# Run the function
restore_files()