import os
import time
import shutil
import zipfile
from datetime import datetime

# CONFIGURATION
SRC_DIR = r"D:\dataDoctor"
BACKUP_ROOT = r"D:\ProjectBackups\dataDoctor"
LATEST_MIRROR_DIR = os.path.join(BACKUP_ROOT, "latest")
HISTORY_DIR = os.path.join(BACKUP_ROOT, "history")
LOG_FILE = os.path.join(BACKUP_ROOT, "backup_log.txt")

# Scan every 30 seconds for changes
CHECK_INTERVAL_SECONDS = 30
# Keep maximum 50 historical backups to save space
MAX_HISTORY_COUNT = 50

# Folders and file extensions to completely ignore
IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", 
    ".idea", ".vscode", ".gemini", ".antigravitycli", ".nydra_versions",
    "uploads", "node_modules"
}

IGNORE_EXTENSIONS = {
    ".db", ".db-shm", ".db-wal", ".zip", ".tar.gz", ".log"
}

# Ignore specific files
IGNORE_FILES = {
    "auto_backup.py", "backup_log.txt"
}

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    # Write to log file
    try:
        os.makedirs(BACKUP_ROOT, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception:
        pass

def should_ignore(file_path):
    # Relative path from the source directory
    rel_path = os.path.relpath(file_path, SRC_DIR)
    parts = rel_path.split(os.sep)
    
    # Check if any parent directory is ignored
    for part in parts[:-1]:
        if part in IGNORE_DIRS:
            return True
            
    # Check if file name itself is ignored
    filename = parts[-1]
    if filename in IGNORE_FILES:
        return True
        
    # Check if extension is ignored
    ext = os.path.splitext(filename)[1].lower()
    if ext in IGNORE_EXTENSIONS:
        return True
        
    return False

def scan_files():
    """Scans the source directory and returns a dict mapping relative file paths to their modification times."""
    files_map = {}
    try:
        for root, dirs, filenames in os.walk(SRC_DIR):
            # Prune ignored directories to speed up scanning
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for filename in filenames:
                full_path = os.path.join(root, filename)
                if not should_ignore(full_path):
                    try:
                        mtime = os.path.getmtime(full_path)
                        rel_path = os.path.relpath(full_path, SRC_DIR)
                        files_map[rel_path] = mtime
                    except OSError:
                        pass
    except Exception as e:
        log(f"Error scanning files: {e}")
    return files_map

def clean_old_history():
    """Keeps only the most recent historical backups."""
    if not os.path.exists(HISTORY_DIR):
        return
    try:
        backups = [os.path.join(HISTORY_DIR, f) for f in os.listdir(HISTORY_DIR) if f.endswith(".zip")]
        # Sort by creation time (oldest first)
        backups.sort(key=os.path.getctime)
        
        while len(backups) > MAX_HISTORY_COUNT:
            oldest_backup = backups.pop(0)
            os.remove(oldest_backup)
            log(f"Removed old historical backup to save space: {os.path.basename(oldest_backup)}")
    except Exception as e:
        log(f"Error cleaning old backups: {e}")

def perform_backup(changed_files):
    """Creates a latest mirror update and a historical ZIP containing only changed files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_filename = f"backup_{timestamp}.zip"
    zip_path = os.path.join(HISTORY_DIR, zip_filename)
    
    os.makedirs(LATEST_MIRROR_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    log(f"Detected {len(changed_files)} changed/new files. Backing up...")
    
    try:
        # Create ZIP archive for modified files
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for rel_path in changed_files:
                src_file = os.path.join(SRC_DIR, rel_path)
                dest_file = os.path.join(LATEST_MIRROR_DIR, rel_path)
                
                # 1. Update the 'latest' mirror
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                try:
                    shutil.copy2(src_file, dest_file)
                except Exception as e:
                    log(f"Could not copy {rel_path} to latest mirror: {e}")
                
                # 2. Add to the ZIP archive
                try:
                    zipf.write(src_file, rel_path)
                except Exception as e:
                    log(f"Could not add {rel_path} to zip archive: {e}")
                    
        log(f"Successfully saved historical snapshot: {zip_filename}")
        
        # Clean old historical ZIPs to prevent disk bloating
        clean_old_history()
        
    except Exception as e:
        log(f"Backup failed: {e}")

def main():
    log("=========================================")
    log("Starting Real-Time Auto-Backup Service...")
    log(f"Monitoring: {SRC_DIR}")
    log(f"Mirror backup path: {LATEST_MIRROR_DIR}")
    log(f"History backup path: {HISTORY_DIR}")
    log("=========================================")
    
    # Initial scan
    known_files = scan_files()
    log(f"Initial scan complete. Monitoring {len(known_files)} files.")
    
    while True:
        try:
            time.sleep(CHECK_INTERVAL_SECONDS)
            current_files = scan_files()
            
            changed_files = []
            
            # Find new or modified files
            for rel_path, mtime in current_files.items():
                if rel_path not in known_files:
                    # New file
                    changed_files.append(rel_path)
                elif mtime > known_files[rel_path]:
                    # Modified file
                    changed_files.append(rel_path)
            
            # Find deleted files (to update our known files tracker)
            deleted_files = [path for path in known_files if path not in current_files]
            for path in deleted_files:
                log(f"File deleted in project: {path} (Removed from tracker)")
                
            if changed_files:
                perform_backup(changed_files)
                
            # Update the known state
            known_files = current_files
            
        except KeyboardInterrupt:
            log("Stopping Auto-Backup Service. Goodbye!")
            break
        except Exception as e:
            log(f"Unexpected error in monitor loop: {e}")
            time.sleep(10) # Wait a bit before retrying in case of strange filesystem errors

if __name__ == "__main__":
    main()
