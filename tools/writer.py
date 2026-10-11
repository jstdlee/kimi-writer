"""
File writing tool for creating and managing markdown files.
"""

import os
from typing import Literal
from .project import get_active_project_folder


def write_file_impl(filename: str, content: str, mode: Literal["create", "append", "overwrite"]) -> str:
    """
    Writes content to a markdown file in the active project folder.
    
    Args:
        filename: The name of the file to write
        content: The content to write
        mode: The write mode - 'create', 'append', or 'overwrite'
        
    Returns:
        Success message or error message
    """
    # Check if project folder is initialized
    project_folder = get_active_project_folder()
    if not project_folder:
        return "Error: No active project folder. Please create a project first using create_project."
    
    if (not filename or os.path.basename(filename) != filename
            or "/" in filename or "\\" in filename or ".." in filename):
        return f"Error: Invalid filename '{filename}'. Use a plain file name without path separators or '..'."

    # Ensure filename ends with .md
    if not filename.endswith('.md'):
        filename = filename + '.md'
    
    # Create full file path
    file_path = os.path.join(project_folder, filename)
    root = os.path.realpath(project_folder)
    if os.path.commonpath([root, os.path.realpath(file_path)]) != root:
        return f"Error: '{filename}' resolves outside the project folder."
    
    try:
        if mode == "create":
            # Create mode: fail if file exists
            if os.path.exists(file_path):
                return f"Error: File '{filename}' already exists. Use 'append' or 'overwrite' mode to modify it."
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully created file '{filename}' with {len(content)} characters."
        
        elif mode == "append":
            # Append mode: add to end of file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully appended {len(content)} characters to '{filename}'."
        
        elif mode == "overwrite":
            # Overwrite mode: replace entire file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully overwrote '{filename}' with {len(content)} characters."
        
        else:
            return f"Error: Invalid mode '{mode}'. Use 'create', 'append', or 'overwrite'."
    
    except Exception as e:
        return f"Error writing file '{filename}': {str(e)}"

