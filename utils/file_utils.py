import h5py
import numpy as np

class FileUtils:

    @staticmethod
    def read_h5_file(file_path, dtype=None):
        """
        Reads .h5 files and returns a numpy array.
        """
        
        # Open the .h5 file in read mode
        with h5py.File(file_path, "r") as f:
            # Get the first key from the .h5 file
            first_key = list(f.keys())[0]
            
            # Check if a specific data type is provided
            if dtype is None:
                return np.array(f[first_key])
            else:
                return np.array(f[first_key], dtype=dtype)
