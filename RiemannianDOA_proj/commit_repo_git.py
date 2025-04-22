import os
import subprocess

def git_commit_and_push():
    repo_path = os.path.dirname(os.path.abspath(__file__))
    print(repo_path)
    print("bye")
    commit_message = "Auto commit after script execution"

    try:
        # Change directory to the repo
        os.chdir(repo_path)

        # Add all changes
        subprocess.run(["git", "add", "."], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Force push to remote
        subprocess.run(["git", "push", "--force"], check=True)

        print("✅ Git commit and push completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")


def remove_empty_dirs(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            dir_to_check = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_check):  # Check if the directory is empty
                os.rmdir(dir_to_check)
                print(f"Deleted empty directory: {dir_to_check}")