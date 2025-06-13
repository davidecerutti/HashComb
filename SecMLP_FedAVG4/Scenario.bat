#!/bin/bash

# Directory containing the files
DIR="/path/to/your/directory"
# File extension to filter files
EXT=".csv"
# Lock file path
LOCK_FILE="/path/to/lockfile.lock"
# Path to Java executable
JAVA_EXEC="/path/to/your/java_executable"
# Validation subdirectory
VALIDATION_DIR="$DIR/validation"
# Range of files to iterate (e.g., [1,5] to iterate from Name.1.csv to Name.5.csv)
RANGE_START=1
RANGE_END=5

# Create validation directory if it doesn't exist
mkdir -p "$VALIDATION_DIR"

# Loop through each file in the directory
for FILE in "$DIR"/*$EXT; do
    # Extract the number from the filename (e.g., Name.1.csv -> 1)
    FILE_NUMBER=$(echo "$FILE" | grep -oP '\.\K\d+(?=\.csv)')
    
    # Check if the file number is within the specified range
    if [ "$FILE_NUMBER" -ge "$RANGE_START" ] && [ "$FILE_NUMBER" -le "$RANGE_END" ]; then
        # Wait until the lock file does not exist
        while [ -f "$LOCK_FILE" ]; do
            echo "Waiting for lock file to be released..."
            sleep 1
        done
        
        # Move the file to the validation directory
        mv "$FILE" "$VALIDATION_DIR/"
        FILE_IN_VALIDATION="$VALIDATION_DIR/$(basename "$FILE")"
        
        # If the lock file doesn't exist, proceed with running the Java executable
        echo "Processing file: $FILE_IN_VALIDATION"
        java -jar "$JAVA_EXEC" "$FILE_IN_VALIDATION"
        
        # Move the file back to the original directory after processing
        mv "$FILE_IN_VALIDATION" "$DIR/"
        
        # Create the lock file to ensure synchronization
        touch "$LOCK_FILE"
        
        # Remove the lock file after processing
        rm -f "$LOCK_FILE"
    fi
done

echo "All files in the specified range processed."
