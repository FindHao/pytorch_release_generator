import re
import os


def clean_tags(file_path):
    # Ensure file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
        
    print(f"Processing file: {file_path}")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} lines")
    
    current_category = ""
    cleaned_lines = []
    changes_made = False
    
    for i, line in enumerate(lines):
        original_line = line
        # Check if it's a category line (starts with ##)
        if line.startswith('##'):
            current_category = line.strip('##').strip().rstrip(':').lower()
            cleaned_lines.append(line)
            continue
            
        # If line doesn't start with -, add it directly
        if not line.strip().startswith('-'):
            cleaned_lines.append(line)
            continue
            
        # Process lines with tags
        line = line.strip()
        if not line:
            cleaned_lines.append('\n')
            continue
            
        # Extract all tags within square brackets
        tags = re.findall(r'\[(.*?)\]', line)
        content = re.sub(r'\[.*?\]', '', line).strip()
        
        # If content starts with -, remove it (since we'll add it when rebuilding the line)
        content = content.lstrip('- ').strip()
        
        # Clean tags
        seen_tags = set()
        cleaned_tags = []
        
        for tag in tags:
            # Remove leading/trailing whitespace but keep internal spaces
            tag_cleaned = tag.strip()
            tag_lower = tag_cleaned.lower()
            
            # Skip empty tags
            if not tag_lower:
                continue
                
            # Skip tags that match current category
            if tag_lower == current_category:
                continue
                
            # Skip duplicate tags
            if tag_lower in seen_tags:
                continue
                
            seen_tags.add(tag_lower)
            cleaned_tags.append(f'[{tag_cleaned}]')
        
        # Rebuild line, ensuring only one -
        new_line = '- ' + ' '.join(cleaned_tags)
        if cleaned_tags:  # Add space if there are tags
            new_line += ' '
        new_line += content + '\n'
        
        if new_line != original_line:
            changes_made = True
            print(f"Line {i+1} changed:")
            print(f"From: {original_line.strip()}")
            print(f"To:   {new_line.strip()}\n")
            
        cleaned_lines.append(new_line)
    
    if not changes_made:
        print("No changes were necessary!")
        return
        
    # Write back file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    print(f"File has been updated successfully!")


# Usage example
if __name__ == "__main__":
    file_path = 'release.md'  # Make sure this is the correct file path
    clean_tags(file_path)
