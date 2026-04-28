#!/usr/bin/env python3
import os
import glob
import argparse
from datetime import datetime
import shutil


def get_timestamp_from_filename(filename):
    """Extract timestamp from filename."""
    # Extract timestamp pattern: YYYYMMDD_HHMMSS
    parts = filename.split('_')
    if len(parts) >= 3:
        try:
            return f"{parts[-2]}_{parts[-1].split('.')[0]}"
        except:
            return None
    return None

def get_all_timestamps(directory):
    """Get all unique timestamps from files in the directory."""
    files = glob.glob(os.path.join(directory, "*.npy"))
    timestamps = set()
    for file in files:
        timestamp = get_timestamp_from_filename(os.path.basename(file))
        if timestamp:
            timestamps.add(timestamp)
    return sorted(list(timestamps))

def get_files_for_timestamp(directory, timestamp):
    """Get all files associated with a specific timestamp."""
    pattern = os.path.join(directory, f"*_{timestamp}.npy")
    return glob.glob(pattern)

def backup_files(directory, timestamp):
    """Create a backup of files before deletion."""
    backup_dir = os.path.join(directory, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    files = get_files_for_timestamp(directory, timestamp)
    for file in files:
        backup_path = os.path.join(backup_dir, os.path.basename(file))
        shutil.copy2(file, backup_path)

def main():
    parser = argparse.ArgumentParser(description='Clean up simulation data files')
    parser.add_argument('--directory', default='simulation_data',
                      help='Directory containing simulation data files')
    parser.add_argument('--keep', nargs='+',
                      help='Timestamps to keep (all others will be deleted)')
    parser.add_argument('--remove', nargs='+',
                      help='Timestamps to remove')
    parser.add_argument('--list', action='store_true',
                      help='List all available timestamps')
    parser.add_argument('--backup', action='store_true',
                      help='Create backup before deletion')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    # Ensure directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    # Get all timestamps
    all_timestamps = get_all_timestamps(args.directory)
    
    if args.list:
        print("\nAvailable timestamps:")
        for ts in all_timestamps:
            print(f"  {ts}")
        return
    
    # Determine which timestamps to remove
    timestamps_to_remove = set()
    if args.keep:
        timestamps_to_remove = set(all_timestamps) - set(args.keep)
    elif args.remove:
        timestamps_to_remove = set(args.remove)
    else:
        print("Error: Must specify either --keep or --remove")
        return
    
    # Validate timestamps
    invalid_timestamps = timestamps_to_remove - set(all_timestamps)
    if invalid_timestamps:
        print(f"Error: Invalid timestamps specified: {invalid_timestamps}")
        return
    
    # Process deletions
    for timestamp in timestamps_to_remove:
        files = get_files_for_timestamp(args.directory, timestamp)
        if args.backup:
            backup_files(args.directory, timestamp)
        
        if args.dry_run:
            print(f"\nWould delete files for timestamp {timestamp}:")
            for file in files:
                print(f"  {os.path.basename(file)}")
        else:
            print(f"\nDeleting files for timestamp {timestamp}:")
            for file in files:
                print(f"  {os.path.basename(file)}")
                os.remove(file)

if __name__ == "__main__":
    main() 