import os

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)
shared_dir = os.path.join(project_root, 'shared')

print(f"Current directory: {current_dir}")
print(f"Backend directory: {backend_dir}")
print(f"Project root: {project_root}")
print(f"Shared directory: {shared_dir}")
print(f"Shared exists: {os.path.exists(shared_dir)}")

# List contents of shared directory
if os.path.exists(shared_dir):
    print("Shared directory contents:")
    for item in os.listdir(shared_dir):
        print(f"  - {item}")
        