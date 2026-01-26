#!/usr/bin/env python3
"""
VibeAgent Project Cleanup Script
Safely cleans up unnecessary files while preserving important data.
"""

import os
import sys
import shutil
import glob
from datetime import datetime
from pathlib import Path
import argparse

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(text, color=Colors.WHITE, bold=False):
    """Print colored text"""
    if bold:
        print(f"{Colors.BOLD}{color}{text}{Colors.ENDC}")
    else:
        print(f"{color}{text}{Colors.ENDC}")

def print_section(title):
    """Print a section header"""
    print()
    print_color(f"{'='*60}", Colors.CYAN, bold=True)
    print_color(f"  {title}", Colors.CYAN, bold=True)
    print_color(f"{'='*60}", Colors.CYAN, bold=True)
    print()

def format_size(size_bytes):
    """Format size in bytes to human-readable format"""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def get_file_size(path):
    """Get file size in human-readable format"""
    try:
        size = os.path.getsize(path)
        return format_size(size)
    except:
        return "Unknown"

def create_backup(archive_dir):
    """Create a backup of the entire project"""
    print_section("CREATING BACKUP")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"vibeagent_backup_{timestamp}"
    backup_path = os.path.join(archive_dir, backup_name)
    
    print_color(f"Creating backup at: {backup_path}", Colors.BLUE)
    print_color("This may take a moment...", Colors.BLUE)
    
    # Exclude venv and .git directories from backup
    exclude_dirs = {'venv', '.git', '__pycache__', '.pytest_cache', '.ruff_cache'}
    
    os.makedirs(backup_path, exist_ok=True)
    
    # Walk through directory and copy files
    for root, dirs, files in os.walk('.', topdown=True):
        # Remove excluded directories from traversal
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, '.')
            dest_path = os.path.join(backup_path, rel_path)
            
            # Create destination directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dest_path)
    
    # Create tarball for easy archiving
    tar_path = f"{backup_path}.tar.gz"
    shutil.make_archive(backup_path, 'gztar', backup_path)
    
    print_color(f"✓ Backup created: {tar_path}", Colors.GREEN)
    return backup_path, tar_path

def find_files(patterns, root='.'):
    """Find files matching patterns"""
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    return found_files

def get_cleanup_targets():
    """Get all files and directories to clean up"""
    targets = {
        'log_files': [],
        'db_backups': [],
        'backup_files': [],
        'cache_dirs': [],
        'root_tests': [],
        'docs_files': [],
    }
    
    # Log files (*.log)
    targets['log_files'] = find_files(['**/*.log'])
    
    # Database backups (data/*.db_**, data/test_*.db)
    db_patterns = [
        'data/*_*.db',  # Pattern for db_** variants
        'data/test_*.db',
        'data/*_test.db',
        'data/*_backup.db',
        'data/*.db.bak',
    ]
    targets['db_backups'] = find_files(db_patterns)
    
    # Backup files
    backup_patterns = [
        '**/*.py.bak',
        '**/*.broken',
        '**/*.backup',
        '**/*~',
        '**/.DS_Store',
    ]
    targets['backup_files'] = find_files(backup_patterns)
    
    # Cache directories
    cache_patterns = [
        '**/__pycache__',
        '**/.pytest_cache',
        '**/.ruff_cache',
        '**/.mypy_cache',
        '**/*.pyc',
    ]
    targets['cache_dirs'] = find_files(cache_patterns)
    
    # Root test files (move to tests/legacy/)
    test_patterns = [
        './test_*.py',
        './benchmark.py',
        './quick_test.py',
        './quick_context_test.py',
        './run_*.py',
        './verify_implementation.py',
    ]
    targets['root_tests'] = find_files(test_patterns, root='.')
    
    # Documentation files (move to docs/legacy/)
    doc_patterns = [
        './*SUMMARY.md',
        './*_PLAN.md',
        './*_ANALYSIS.md',
        './*_GUIDE.md',
        './*_RESULTS.md',
        './*_COMPLETE.md',
        './*_README.md',
    ]
    targets['docs_files'] = find_files(doc_patterns, root='.')
    
    return targets

def show_cleanup_preview(targets):
    """Show preview of what will be cleaned up"""
    print_section("CLEANUP PREVIEW")
    
    total_size = 0
    
    for category, files in targets.items():
        if not files:
            continue
            
        category_names = {
            'log_files': 'Log Files',
            'db_backups': 'Database Backups',
            'backup_files': 'Backup Files',
            'cache_dirs': 'Cache Directories',
            'root_tests': 'Root Test Files (to be moved)',
            'docs_files': 'Documentation Files (to be moved)',
        }
        
        print_color(f"\n{category_names[category]}:", Colors.YELLOW, bold=True)
        
        category_size = 0
        for file_path in sorted(files):
            try:
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    category_size += size
                    print_color(f"  - {file_path} ({get_file_size(file_path)})", Colors.WHITE)
                elif os.path.isdir(file_path):
                    # Calculate directory size
                    dir_size = 0
                    for root, dirs, files_in_dir in os.walk(file_path):
                        for f in files_in_dir:
                            fp = os.path.join(root, f)
                            try:
                                dir_size += os.path.getsize(fp)
                            except:
                                pass
                    category_size += dir_size
                    print_color(f"  - {file_path}/ (DIR, {format_size(dir_size)})", Colors.WHITE)
            except:
                print_color(f"  - {file_path} (size unknown)", Colors.WHITE)
        
        total_size += category_size
        print_color(f"  Subtotal: {format_size(category_size)}", Colors.CYAN)
    
    print_color(f"\n{'='*60}", Colors.MAGENTA)
    print_color(f"TOTAL SPACE TO BE FREED: {format_size(total_size)}", Colors.MAGENTA, bold=True)
    print_color(f"{'='*60}", Colors.MAGENTA)
    
    return total_size

def move_files_to_legacy(files, target_dir, category_name):
    """Move files to a legacy directory"""
    if not files:
        return 0, 0
    
    print_color(f"\nMoving {category_name} to {target_dir}/...", Colors.BLUE)
    
    os.makedirs(target_dir, exist_ok=True)
    moved_count = 0
    total_size = 0
    
    for file_path in files:
        if not os.path.exists(file_path):
            continue
            
        try:
            # Get relative path and create target path
            rel_path = os.path.relpath(file_path, '.')
            dest_path = os.path.join(target_dir, rel_path)
            
            # Create destination directory
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Move file/directory
            if os.path.isdir(file_path):
                shutil.move(file_path, dest_path)
                # Calculate size
                for root, dirs, files_in_dir in os.walk(dest_path):
                    for f in files_in_dir:
                        fp = os.path.join(root, f)
                        try:
                            total_size += os.path.getsize(fp)
                        except:
                            pass
            else:
                file_size = os.path.getsize(file_path)
                shutil.move(file_path, dest_path)
                total_size += file_size
            
            print_color(f"  → Moved: {rel_path}", Colors.WHITE)
            moved_count += 1
            
        except Exception as e:
            print_color(f"  ✗ Failed to move {file_path}: {e}", Colors.RED)
    
    print_color(f"✓ Moved {moved_count} items ({format_size(total_size)})", Colors.GREEN)
    return moved_count, total_size

def remove_files(files, category_name):
    """Remove files permanently"""
    if not files:
        return 0, 0
    
    print_color(f"\nRemoving {category_name}...", Colors.BLUE)
    
    removed_count = 0
    total_size = 0
    
    for file_path in files:
        if not os.path.exists(file_path):
            continue
            
        try:
            # Calculate size before removal
            if os.path.isdir(file_path):
                dir_size = 0
                for root, dirs, files_in_dir in os.walk(file_path):
                    for f in files_in_dir:
                        fp = os.path.join(root, f)
                        try:
                            dir_size += os.path.getsize(fp)
                        except:
                            pass
                total_size += dir_size
                shutil.rmtree(file_path)
            else:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                os.remove(file_path)
            
            rel_path = os.path.relpath(file_path, '.')
            print_color(f"  ✓ Removed: {rel_path}", Colors.WHITE)
            removed_count += 1
            
        except Exception as e:
            print_color(f"  ✗ Failed to remove {file_path}: {e}", Colors.RED)
    
    print_color(f"✓ Removed {removed_count} items ({format_size(total_size)})", Colors.GREEN)
    return removed_count, total_size

def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description='Clean up VibeAgent project files')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Run without interactive confirmation')
    args = parser.parse_args()
    
    print_section("VIBEAGENT PROJECT CLEANUP")
    
    # Check if we're in the right directory
    if not os.path.exists('main.py') or not os.path.exists('vibeagent.py'):
        print_color("Error: Please run this script from the VibeAgent project root directory.", Colors.RED)
        sys.exit(1)
    
    # Create archives directory
    archive_dir = os.path.join('.', 'archives')
    os.makedirs(archive_dir, exist_ok=True)
    
    # Get cleanup targets
    targets = get_cleanup_targets()
    
    # Show preview
    show_cleanup_preview(targets)
    
    # Ask for confirmation unless --force is used
    if not args.force:
        print()
        print_color("IMPORTANT FILES THAT WILL BE PRESERVED:", Colors.GREEN, bold=True)
        print_color("  - README.md", Colors.GREEN)
        print_color("  - main.py", Colors.GREEN)
        print_color("  - vibeagent.py", Colors.GREEN)
        print_color("  - requirements.txt", Colors.GREEN)
        print_color("  - data/main.db (if exists)", Colors.GREEN)
        print_color("  - All source code in core/, api/, skills/, etc.", Colors.GREEN)
        print()
        
        try:
            response = input("Do you want to proceed with cleanup? (yes/no): ").strip().lower()
            if response != 'yes':
                print_color("\nCleanup cancelled.", Colors.YELLOW)
                sys.exit(0)
        except EOFError:
            print_color("\nNo terminal detected. Use --force flag to run non-interactively.", Colors.YELLOW)
            sys.exit(1)
    else:
        print_color("\nRunning in non-interactive mode (--force)", Colors.BLUE)
    
    # Create backup
    try:
        backup_path, tar_path = create_backup(archive_dir)
    except Exception as e:
        print_color(f"Failed to create backup: {e}", Colors.RED)
        if args.force:
            print_color("Continuing without backup due to --force flag...", Colors.YELLOW)
            backup_path = None
            tar_path = None
        else:
            response = input("Continue without backup? (yes/no): ").strip().lower()
            if response != 'yes':
                sys.exit(1)
            backup_path = None
            tar_path = None
    
    # Perform cleanup
    print_section("PERFORMING CLEANUP")
    
    stats = {
        'removed_files': 0,
        'removed_size': 0,
        'moved_files': 0,
        'moved_size': 0,
    }
    
    # Move root test files to tests/legacy/
    if targets['root_tests']:
        count, size = move_files_to_legacy(
            targets['root_tests'], 
            'tests/legacy', 
            'Root Test Files'
        )
        stats['moved_files'] += count
        stats['moved_size'] += size
    
    # Move documentation files to docs/legacy/
    if targets['docs_files']:
        count, size = move_files_to_legacy(
            targets['docs_files'],
            'docs/legacy',
            'Documentation Files'
        )
        stats['moved_files'] += count
        stats['moved_size'] += size
    
    # Remove log files
    if targets['log_files']:
        count, size = remove_files(targets['log_files'], 'Log Files')
        stats['removed_files'] += count
        stats['removed_size'] += size
    
    # Remove database backups
    if targets['db_backups']:
        count, size = remove_files(targets['db_backups'], 'Database Backups')
        stats['removed_files'] += count
        stats['removed_size'] += size
    
    # Remove backup files
    if targets['backup_files']:
        count, size = remove_files(targets['backup_files'], 'Backup Files')
        stats['removed_files'] += count
        stats['removed_size'] += size
    
    # Remove cache directories
    if targets['cache_dirs']:
        count, size = remove_files(targets['cache_dirs'], 'Cache Directories')
        stats['removed_files'] += count
        stats['removed_size'] += size
    
    # Show summary
    print_section("CLEANUP COMPLETE")
    
    print_color("\nSUMMARY:", Colors.GREEN, bold=True)
    print_color(f"  Files removed: {stats['removed_files']}", Colors.WHITE)
    print_color(f"  Space freed: {format_size(stats['removed_size'])}", Colors.WHITE)
    print_color(f"  Files organized: {stats['moved_files']}", Colors.WHITE)
    print_color(f"  Space in organized files: {format_size(stats['moved_size'])}", Colors.WHITE)
    
    if backup_path:
        print_color(f"\n  Backup created: {tar_path}", Colors.WHITE)
    
    # Create cleanup report
    report_path = os.path.join(archive_dir, f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, 'w') as f:
        f.write("VibeAgent Project Cleanup Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("REMOVED FILES:\n")
        f.write("-"*60 + "\n")
        for category, files in targets.items():
            if category in ['root_tests', 'docs_files']:
                continue
            if files:
                f.write(f"\n{category}:\n")
                for file_path in sorted(files):
                    f.write(f"  - {file_path}\n")
        
        f.write("\n\nORGANIZED FILES:\n")
        f.write("-"*60 + "\n")
        for category, files in targets.items():
            if category in ['root_tests', 'docs_files'] and files:
                f.write(f"\n{category}:\n")
                for file_path in sorted(files):
                    f.write(f"  - {file_path}\n")
        
        f.write(f"\n\nSTATISTICS:\n")
        f.write(f"  Files removed: {stats['removed_files']}\n")
        f.write(f"  Space freed: {format_size(stats['removed_size'])}\n")
        f.write(f"  Files organized: {stats['moved_files']}\n")
        f.write(f"  Space in organized files: {format_size(stats['moved_size'])}\n")
        
        if backup_path:
            f.write(f"\nBackup location: {tar_path}\n")
    
    print_color(f"\n  Cleanup report: {report_path}", Colors.WHITE)
    
    print_color("\n✓ Project cleanup completed successfully!", Colors.GREEN, bold=True)
    print_color("  Your project is now clean and organized.", Colors.GREEN)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_color("\n\nCleanup interrupted by user.", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_color(f"\nUnexpected error: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1)
