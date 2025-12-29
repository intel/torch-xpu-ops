import pandas as pd
import argparse
import sys
import shutil
import os

def replace_time_columns_inplace(main_file, source_file, backup=True):
    """
    Replace 'time(us)' and 'E2E total time(us)' in main_file with values from source_file
    when all other columns match exactly.
    
    Args:
        main_file: Main CSV file to be updated (modified in place)
        source_file: Reference CSV file providing new time values
        backup: Whether to create a backup of the original file (default: True)
    """
    try:
        # Create backup if requested
        if backup:
            backup_file = os.path.splitext(main_file)[0] + '_backup.csv'
            shutil.copy2(main_file, backup_file)
            print(f"✅ Created backup: {backup_file}")
        
        # Read CSV files with semicolon delimiter
        df_main = pd.read_csv(main_file, sep=';', dtype=str)  # dtype=str to preserve exact string matching
        df_source = pd.read_csv(source_file, sep=';', dtype=str)
        
        print(f"📁 Files loaded successfully:")
        print(f"  Main file ({main_file}): {len(df_main)} rows")
        print(f"  Source file ({source_file}): {len(df_source)} rows")
        
        # Strip whitespace from column names (in case of extra spaces)
        df_main.columns = df_main.columns.str.strip()
        df_source.columns = df_source.columns.str.strip()
        
        # Define the two time columns to replace
        time_cols = ['time(us)', 'E2E total time(us)']
        
        # Validate required columns exist
        missing_time_main = [col for col in time_cols if col not in df_main.columns]
        missing_time_source = [col for col in time_cols if col not in df_source.columns]
        
        if missing_time_main:
            raise ValueError(f"Missing time columns in {main_file}: {missing_time_main}")
        if missing_time_source:
            raise ValueError(f"Missing time columns in {source_file}: {missing_time_source}")
        
        # Key columns = all columns except the two time columns
        key_columns = [col for col in df_main.columns if col not in time_cols]
        # Ensure key columns exist in source too (order doesn't matter, but names must match)
        missing_keys_source = [col for col in key_columns if col not in df_source.columns]
        if missing_keys_source:
            raise ValueError(f"Missing key columns in {source_file}: {missing_keys_source}")
        
        print(f"\n🔑 Using {len(key_columns)} key columns for matching:")
        print("  " + ", ".join(key_columns[:5]) + ("..." if len(key_columns) > 5 else ""))
        if len(key_columns) > 5:
            print(f"  (total {len(key_columns)} keys)")

        # Build lookup dictionary from source: {tuple_of_keys: (time, e2e_time)}
        source_dict = {}
        duplicated_keys = 0
        for idx, row in df_source.iterrows():
            key = tuple(row[col].strip() if isinstance(row[col], str) else row[col] for col in key_columns)
            time_val = row['time(us)']
            e2e_val = row['E2E total time(us)']
            
            if key in source_dict:
                duplicated_keys += 1
                # Keep first occurrence (or warn — you may prefer to raise error)
                # For now, just note duplicates
            else:
                source_dict[key] = (time_val, e2e_val)
        
        if duplicated_keys > 0:
            print(f"⚠️  Warning: {duplicated_keys} duplicate key(s) found in source file. First occurrence used.")

        # Replace in main
        replaced_count = 0
        for idx in range(len(df_main)):
            key = tuple(
                df_main.iloc[idx][col].strip() if isinstance(df_main.iloc[idx][col], str) else df_main.iloc[idx][col]
                for col in key_columns
            )
            if key in source_dict:
                old_time = df_main.at[idx, 'time(us)']
                old_e2e = df_main.at[idx, 'E2E total time(us)']
                new_time, new_e2e = source_dict[key]
                
                df_main.at[idx, 'time(us)'] = new_time
                df_main.at[idx, 'E2E total time(us)'] = new_e2e
                replaced_count += 1
                
                if replaced_count <= 3:  # log first 3
                    print(f"🔄 Row {idx}: time({old_time} → {new_time}), E2E({old_e2e} → {new_e2e})")

        # Save back with semicolon delimiter
        df_main.to_csv(main_file, sep=';', index=False)
        
        print(f"\n✅ Processing completed!")
        print(f"Total rows: {len(df_main)}")
        print(f"Rows replaced: {replaced_count}")
        print(f"Rows unchanged: {len(df_main) - replaced_count}")
        print(f"File '{main_file}' updated in place.")
        
        if backup:
            print(f"\nℹ️  Backup saved as: {backup_file}")
            print(f"   To restore: copy \"{backup_file}\" \"{main_file}\"")
        
        return df_main

    except FileNotFoundError as e:
        print(f"❌ Error: File not found — {e}")
        if backup and os.path.exists(backup_file):
            print("🔄 Attempting to restore from backup...")
            shutil.copy2(backup_file, main_file)
        return None
    except Exception as e:
        print(f"�� Error during processing: {e}")
        if backup and 'backup_file' in locals() and os.path.exists(backup_file):
            print("🔄 Restoring from backup due to error...")
            shutil.copy2(backup_file, main_file)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Replace "time(us)" and "E2E total time(us)" in main CSV using exact match on all other columns.\n'
                    'Files must be semicolon-separated (;).'
    )
    parser.add_argument('main_file', help='Main CSV file to update (modified in place)')
    parser.add_argument('source_file', help='Source CSV file (provides new time values)')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup (⚠️ risky!)')
    
    args = parser.parse_args()

    print("=" * 70)
    print("⏱️  Time Columns In-Place Replacement Tool (Semicolon CSV)")
    print("=" * 70)

    result = replace_time_columns_inplace(
        main_file=args.main_file,
        source_file=args.source_file,
        backup=not args.no_backup
    )

    if result is not None:
        # Show sample
        sample_cols = ['case_name', 'op_name', 'shape', 'time(us)', 'E2E total time(us)']
        display_cols = [col for col in sample_cols if col in result.columns]
        if display_cols:
            print("\n🔍 Sample of updated data (first 3 rows):")
            print(result[display_cols].head(3).to_string(index=False))
        print("\n✅ Done.")
    else:
        print("\n❌ Operation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
