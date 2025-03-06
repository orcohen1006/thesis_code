import os
import subprocess

def git_commit_and_push():
    repo_path = r"\\wildwest\users\OrCohen\thesis_repo"

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

# Run the function at the end of the script
git_commit_and_push()