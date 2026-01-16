from datetime import datetime
import os
import json
from threading import RLock
from typing import Dict, Optional, Union

class HistoryManager:
    def __init__(self, root_dir):
        # Use a list to store history entries, each entry is a dict type
        self.root_dir = root_dir
        self.history_file = os.path.join(root_dir, "code_history.json")
        self.history = []
        # Ensure the directory exists
        os.makedirs(root_dir, exist_ok=True)
        self.load_from_file()  # Load existing history during initialization

    def append(self, code, source=None):
        """
        Add a history record and automatically save it to the file
        :param code: Code or content to be added
        :param source: Record source (optional)
        """
        try:
            # Create a new history entry (dictionary)
            new_entry = {
                "code": code,
                "source": source,
                "timestamp": datetime.now().isoformat() # Add timestamp
            }
            # Add the new entry to the internal list
            self.history.append(new_entry)
            # Save the updated list to the file
            self._save_to_file()
        except Exception as e:
            pass

    def clear(self):
        """Clear all history records"""
        self.history.clear()
        self._save_to_file()

    def get_history(self): # Keep unchanged, return the internal list
        """Get the original list of history records"""
        return self.history

    def get_formatted_history(self): # Keep unchanged or can be optimized
        """Get formatted history records as a string (for display)"""
        # Can choose to return only code, or include source and timestamp
        # Here returns a format with more information
        formatted_parts = []
        for entry in self.history:
            header = f"\n--- Record from [{entry.get('source', 'Unknown')}] at {entry.get('timestamp', 'Unknown Time')} ---\n"
            formatted_parts.append(header)
            formatted_parts.append(entry["code"])
        return "".join(formatted_parts)

    def _save_to_file(self):
        """Write history records to file"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            pass

    def load_from_file(self):
        """Load history records from JSON file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            self.history = []
        except Exception as e:
            self.history = [] # Reset to a safe state when an error occurs


class StorageManager:
    def __init__(self, root_dir, output_dir):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self.meta_file = os.path.join(root_dir, "meta.json")
        os.makedirs(self._root_dir, exist_ok=True)

        # Internal dual zones: cache zone (temporary) + storage zone (official/persistent)
        self._meta_cache: Dict[str, dict] = {}  # Cache zone (temporary, unconfirmed)
        self._meta_storage: Dict[str, dict] = {}  # Storage zone (persistent)
        self._lock = RLock()  # Thread-safe lock

        # Initialize file and load storage zone data
        self._init_meta_file()
        self._load_storage_from_file()

    # ------------------------------ Internal Utility Methods ------------------------------
    def _init_meta_file(self):
        """Initialize storage zone metadata file"""
        if not os.path.exists(self.meta_file):
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_storage_from_file(self):
        """Load storage zone data from file"""
        with self._lock:
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self._meta_storage = json.load(f)
            except json.JSONDecodeError as e:
                self._meta_storage = {}
            except Exception as e:
                self._meta_storage = {}

    def _save_storage_to_file(self):
        """Persist storage zone data to file"""
        with self._lock:
            try:
                with open(self.meta_file, "w", encoding="utf-8") as f:
                    json.dump(self._meta_storage, f, indent=2)
            except Exception as e:
                raise

    def _get_file_full_path(self, filename: str) -> str:
        """Internal utility: Get the full physical path of the file"""
        return os.path.join(self._root_dir, filename)

    # ------------------------------ Original Public Interfaces (Keep Compatibility, Support Override by Same Name) ------------------------------
    def register_file(self, filename, description, created_by, file_type, temp: bool = True):
        """
        Register file metadata (automatic override for same filename)
        :param filename: Filename (unique identifier)
        :param description: File description
        :param created_by: Creator
        :param file_type: File type (e.g., text/image, etc.)
        :param temp: True = store in cache zone (temporary), False = directly store in storage zone (compatible with old logic)
        :return: Filename
        """
        with self._lock:
            meta = {
                "filename": filename,
                "description": description,
                "created_by": created_by,
                "file_type": file_type
            }

            if temp:
                # Cache zone: direct override for same filename (dictionary key re-assignment = override)
                old_meta = self._meta_cache.get(filename)
                self._meta_cache[filename] = meta
            else:
                # Storage zone: direct override for same filename, auto-persist after override
                old_meta = self._meta_storage.get(filename)
                self._meta_storage[filename] = meta
                self._save_storage_to_file()

        return filename

    def read_log(self, include_temp: bool = False):
        """
        Read metadata log
        :param include_temp: True = include cache zone data, False = only read storage zone (compatible with old logic)
        :return: Metadata dictionary (copy, external modification does not affect internal data)
        """
        with self._lock:
            if include_temp:
                # Mixed reading: storage zone + cache zone (cache zone overrides same-name data, naturally prioritizes latest data)
                all_data = self._meta_storage.copy()
                all_data.update(self._meta_cache)  # update method itself overrides same-name keys
                return all_data.copy()
            else:
                # Only read storage zone (compatible with old logic)
                return self._meta_storage.copy()

    def clear_all_records(self, also_delete_files=False, clear_temp: bool = True):
        """
        Clear all records
        :param also_delete_files: Whether to delete physical files at the same time (default False)
        :param clear_temp: True = clear cache zone (default, compatible with old logic), False = only clear storage zone
        :return: None
        """
        with self._lock:
            # 1. Clear cache zone (optional)
            if clear_temp:
                self._meta_cache.clear()

            # 2. Clear storage zone + optionally delete physical files (compatible with old logic)
            if also_delete_files:
                for filename in self._meta_storage:
                    file_path = self._get_file_full_path(filename)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            pass

            # Clear storage zone and persist
            self._meta_storage.clear()
            self._save_storage_to_file()

    # ------------------------------ New: Core Function for Deleting Physical Files + Corresponding Metadata ------------------------------
    def _delete_file_unsafe(self, filename: str, delete_physical: bool, remove_meta: bool):
        if delete_physical:
            file_path = os.path.join(self._output_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass 

        if remove_meta:
            self._meta_cache.pop(filename, None)
            if filename in self._meta_storage:
                del self._meta_storage[filename]
                self._save_storage_to_file()

    def delete_file(self, filename: str, delete_physical: bool = True, remove_meta: bool = True):
        with self._lock:
            self._delete_file_unsafe(filename, delete_physical, remove_meta)

    def batch_delete_files(self, filenames: list, delete_physical: bool = True, remove_meta: bool = True):
        if not filenames:
            return
        with self._lock:
            for filename in filenames:
                self._delete_file_unsafe(filename, delete_physical, remove_meta)

    # ------------------------------ New Public Interfaces (Keep Compatibility, Support Override by Same Name) ------------------------------
    def commit_cache(self, filenames: Union[list, str, None] = None):
        """
        Commit cache zone files to storage zone (persist, automatically override storage zone data for same filenames)
        :param filenames: Specify files to commit → None = all, str = single file, list = multiple files
        :return: None
        """
        with self._lock:
            # Determine the list of files to be committed
            if filenames is None:
                commit_files = list(self._meta_cache.keys())
            elif isinstance(filenames, str):
                commit_files = [filenames] if filenames in self._meta_cache else []
            else:
                commit_files = [f for f in filenames if f in self._meta_cache]

            if not commit_files:
                return

            # Sync to storage zone: direct override for same filenames
            for filename in commit_files:
                old_storage_meta = self._meta_storage.get(filename)
                self._meta_storage[filename] = self._meta_cache[filename].copy()

            # Remove committed cache data
            for filename in commit_files:
                del self._meta_cache[filename]

            # Persist storage zone
            self._save_storage_to_file()

    def clear_cache(self):
        """
        Clear only cache zone (does not affect storage zone)
        :return: None
        """
        with self._lock:
            cache_size = len(self._meta_cache)
            self._meta_cache.clear()

    def read_cache(self):
        """
        Read only all data in temporary zone (cache zone)
        :return: Cache zone metadata dictionary (copy, external modification does not affect internal cache)
        """
        with self._lock:
            # Return a copy to avoid external modification of internal cache
            return self._meta_cache.copy()
