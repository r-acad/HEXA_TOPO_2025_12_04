# concatenate_files.jl
# This script concatenates all files in the current directory and all its sub-directories
# into a single file named 'concatenated_output.txt'.
#
# It includes the original relative path of each file and attempts to remove
# single-line comments (#...) from the content for Julia files.

# --- Global Separator (Polyglot String Format) ---
# We use a string literal format: " // # FILE: <path> ";
# This is syntactically valid (ignored) code in both Julia and JavaScript.
const FILE_SEPARATOR_START = "\"// # FILE: "
const FILE_SEPARATOR_END   = "\";"
const OUTPUT_FILENAME = "concatenated_output.txt"

using FilePathsBase # Add FilePathsBase for robust path handling (optional but good practice)

function remove_comments(content::String)
    # A robust basic comment removal for Julia code using regex
    # This handles single-line comments starting with '#' *if* they are not inside double-quotes.
   
    # Simple regex to find a '#' not preceded by a double-quote, and capture everything after.
    # This is still a simplification (doesn't handle escaped quotes or triple quotes)
    # but is much more robust than the original logic.
   
    lines = split(content, '\n')
    cleaned_lines = String[]
   
    for line in lines
        # Check for single quotes, as simple comment removal can ruin character literals
        if occursin('\'', line)
             push!(cleaned_lines, line) # Keep line as is if it has a character literal (most complex scenarios)
             continue
        end

        # Find the first '#' not preceded by an unescaped double quote (complex to do simply)
        # Using a simpler logic: remove comments only if no double quotes are present.
        if !occursin('"', line)
            comment_start = findfirst('#', line)
           
            if comment_start !== nothing
                # comment_start is an Int here.
                # line[1:comment_start-1] gets the part of the line before the comment.
                cleaned_part = line[1:comment_start-1]
                push!(cleaned_lines, cleaned_part)
                continue # Skip the rest of the loop for this line
            end
        end

        # If no '#' or if a double quote was found, or if it was handled above:
        push!(cleaned_lines, line)
    end
   
    return join(cleaned_lines, '\n')
end


function concatenate_files()
    # Get the absolute path of this script to exclude it
    try
        global my_full_path = abspath(@__FILE__)
    catch
        println("Warning: Could not determine script path. Running from REPL? Assuming 'concatenate_files.jl' in current dir.")
        global my_full_path = abspath("concatenate_files.jl")
    end

    output_full_path = abspath(OUTPUT_FILENAME)
   
    # --- List of extensions to exclude (without the dot) ---
    excluded_extensions = Set(["txt", "vtk", "md", "1html", "vtu", "json", "1yaml", "yml"])
   
    # --- List of folder *names* to exclude ---
    excluded_folders = Set([".git", ".vscode", "1IO", "1RESULTS", "viz", "1PREPOST"])
    # We include the root folder for the exclusion checks, but walkdir handles the iteration.

    # --- Array of file *names* to exclude ---
    excluded_filenames = Set([".gitignore", "1Project.toml", "Manifest.toml", "README.md", "concatenate_code.jl", "restore_files.jl", OUTPUT_FILENAME])
   
    # Open the output file for writing
    try
        open(output_full_path, "w") do outfile
            println("Starting concatenation... Output will be in '$OUTPUT_FILENAME'")
            println("Excluding extensions: $excluded_extensions")
            println("Excluding folders: $excluded_folders")
            println("Excluding filenames: $excluded_filenames")
           
            # Walk the directory tree starting from the current folder "."
            for (root, dirs, files) in walkdir(".")
               
                # --- Filter 'dirs' in-place ---
                filter!(d -> d ∉ excluded_folders, dirs)
               
                for file in files
                    full_path = abspath(joinpath(root, file))
                    relative_path = joinpath(root, file)
                   
                    # --- Get file extension for checking ---
                    basename, ext = splitext(file)
                    cleaned_ext = lstrip(ext, '.')
                   
                    # --- Exclusion Checks ---
                    if full_path == my_full_path || full_path == output_full_path
                        println("Skipping script/output file: $full_path")
                        continue
                    end
                    if file in excluded_filenames
                        println("Skipping file by name: $full_path")
                        continue
                    end
                    if cleaned_ext in excluded_extensions
                        println("Skipping file by extension ($cleaned_ext): $full_path")
                        continue
                    end
                    # --- End Exclusion Checks ---
                   
                    println("Processing: $relative_path")
                   
                    # --- Write the unique header with the relative path ---
                    # Format: "// # FILE: <relative_path>";
                    write(outfile, FILE_SEPARATOR_START, relative_path, FILE_SEPARATOR_END, "\n")
                   
                    # --- Try to read and write the file content ---
                    try
                        file_content = read(full_path, String)
                       
                        # Remove comments only for Julia files (assuming .jl extension)
                        if cleaned_ext == "jl"
                            file_content = remove_comments(file_content)
                        end
                       
                        write(outfile, file_content)
                       
                        # Ensure there is a newline after the content before the next separator
                        if !endswith(file_content, '\n')
                            write(outfile, "\n")
                        end
                       
                    catch e
                        # This handles non-text/binary files that can't be read as String
                        write(outfile, "\n# !!! ERROR: Could not read file (possibly binary or bad encoding or comment removal failure).\n")
                        write(outfile, "# !!! Error details: $e\n")
                    end
                end
            end
        end
       
        println("Concatenation complete! All files saved to '$OUTPUT_FILENAME'")
       
    catch e
        println("An error occurred during file operations: $e")
    end
end

# Run the function
concatenate_files()