import os

def create_project_structure():
    # List of required files
    files = [
        "main.py",
        "chroma_utils.py",
        "db_utils.py",
        "langchain_utils.py",
        "pydantic_models.py",
        "requirements.txt"
    ]
    
    # List of required directories
    directories = [
        "chroma_db"
    ]
    
    # Create each directory (if it doesn't already exist)
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create empty files
    for file in files:
        # Open the file in 'write' mode to create if it doesn't exist
        with open(file, 'w') as f:
            # You can add any initial boilerplate content here if desired
            pass

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")
