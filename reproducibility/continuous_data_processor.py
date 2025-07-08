import numpy as np
import obspy
from obspy import UTCDateTime
import h5py
import pandas as pd
import os
from glob import glob
from tqdm import tqdm


class ContinuousDataPreprocessor:
    """
    Preprocesses continuous MSEED data into HDF5 format compatible with KFold framework.
    Following STEAD structure: single HDF5 file with both earthquake and noise traces.
    """
    
    def __init__(self, 
                 mseed_dir,
                 catalog_csv,
                 output_hdf5_path,
                 output_metadata_csv_path,
                 window_length=30,
                 sampling_rate=100,
                 freqmin=1,
                 freqmax=20,
                 padding=3):
        """
        Args:
            mseed_dir: Directory containing MSEED files (one per station)
            catalog_csv: CSV file with earthquake catalog (p_time, s_time, station columns)
            output_hdf5_path: Path for output HDF5 file
            output_metadata_csv_path: Path for output metadata CSV
            window_length: Window length in seconds (default 30)
            sampling_rate: Target sampling rate (default 100 Hz)
            freqmin: Minimum frequency for bandpass filter
            freqmax: Maximum frequency for bandpass filter
            padding: Padding in seconds for filtering (default 3)
        """
        self.mseed_dir = mseed_dir
        self.catalog_csv = catalog_csv
        self.output_hdf5_path = output_hdf5_path
        self.output_metadata_csv_path = output_metadata_csv_path
        self.window_length = window_length
        self.sampling_rate = sampling_rate
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.padding = padding
        self.padded_window_length = window_length + 2 * padding
        
        # Load catalog
        self.catalog = pd.read_csv(catalog_csv)
        if 'p_arrival_time' in self.catalog.columns:
            self.catalog['p_arrival_time'] = pd.to_datetime(self.catalog['p_arrival_time'])
        if 's_arrival_time' in self.catalog.columns:
            self.catalog['s_arrival_time'] = pd.to_datetime(self.catalog['s_arrival_time'])
        
    def process_all_stations(self):
        """Process all MSEED files in the directory."""
        mseed_files = glob(os.path.join(self.mseed_dir, "*.mseed"))
        
        all_metadata = []
        trace_counter = 0
        
        with h5py.File(self.output_hdf5_path, 'w') as h5f:
            # Create 'data' group
            data_group = h5f.create_group('data')
            
            for mseed_file in tqdm(mseed_files, desc="Processing stations"):
                station_metadata, trace_counter = self.process_station(
                    mseed_file, data_group, trace_counter
                )
                all_metadata.extend(station_metadata)
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(self.output_metadata_csv_path, index=False)
        print(f"Processed {len(all_metadata)} windows total")
        
    def process_station(self, mseed_file, h5_data_group, trace_counter):
        """Process a single station's MSEED file."""
        try:
            stream = obspy.read(mseed_file)
        except Exception as e:
            print(f"Error reading {mseed_file}: {e}")
            return [], trace_counter
            
        # Get station name
        station_name = stream[0].stats.station
        network = stream[0].stats.network
        
        # Merge traces and resample
        stream = stream.merge(fill_value=np.nan)
        
        # Ensure all traces have same sampling rate
        for tr in stream:
            if tr.stats.sampling_rate != self.sampling_rate:
                tr.resample(self.sampling_rate)
        
        metadata = []
        
        # Get Z, N, E components
        z_trace = stream.select(channel="*Z")
        n_trace = stream.select(channel="*N") 
        e_trace = stream.select(channel="*E")
        
        if not (z_trace and n_trace and e_trace):
            # Try alternative channel names
            z_trace = stream.select(channel="*HZ")
            n_trace = stream.select(channel="*HN")
            e_trace = stream.select(channel="*HE")
            
        if not (z_trace and n_trace and e_trace):
            print(f"Missing components for station {station_name}")
            return [], trace_counter
            
        # Use first trace of each component
        z_trace = z_trace[0]
        n_trace = n_trace[0]
        e_trace = e_trace[0]
        
        # Find common time range
        start_time = max(z_trace.stats.starttime, n_trace.stats.starttime, e_trace.stats.starttime)
        end_time = min(z_trace.stats.endtime, n_trace.stats.endtime, e_trace.stats.endtime)
        
        # Trim to common time
        z_trace.trim(start_time, end_time)
        n_trace.trim(start_time, end_time)
        e_trace.trim(start_time, end_time)
        
        # Create windows
        current_time = start_time
        window_step = self.window_length  # No overlap
        
        while current_time + self.padded_window_length <= end_time:
            window_end = current_time + self.padded_window_length
            
            # Extract window data
            z_win = z_trace.slice(current_time, window_end).data
            n_win = n_trace.slice(current_time, window_end).data
            e_win = e_trace.slice(current_time, window_end).data
            
            # Check if window is valid (no gaps/NaN values)
            if self._is_window_valid(z_win, n_win, e_win):

                z_processed = self._preprocess_trace(z_win)
                n_processed = self._preprocess_trace(n_win)
                e_processed = self._preprocess_trace(e_win)
                
                # Stack as (timesteps, channels) - matching STEAD format
                window_data = np.stack([e_processed, n_processed, z_processed], axis=-1)
                
                # Check for earthquakes in this window
                label, p_sample, s_sample = self._check_earthquake_in_window(
                    station_name, current_time + self.padding, self.window_length
                )
                
                # Create trace name
                trace_name = f"{network}.{station_name}.{current_time.strftime('%Y%m%d_%H%M%S')}"
                
                # Save to HDF5
                h5_data_group.create_dataset(trace_name, data=window_data.astype(np.float32))
                
                # Add metadata
                metadata.append({
                    'trace_name': trace_name,
                    'station_name': station_name,
                    'network': network,
                    'trace_start_time': (current_time + self.padding).isoformat(),
                    'label': label,
                    'p_arrival': p_sample if p_sample is not None else np.nan,
                    's_arrival': s_sample if s_sample is not None else np.nan,
                    'source_id': trace_name if label == 'no' else f"eq_{trace_counter}",
                    'trace_category': 'earthquake_local' if label == 'eq' else 'noise',
                    'snr_db': '[0.0 0.0 0.0]',  # Placeholder, can be calculated if needed
                    'crop_offset': 0  # Will be assigned by KFold framework
                })
                
                trace_counter += 1
                
            current_time += window_step
            
        return metadata, trace_counter
    
    def _is_window_valid(self, z_data, n_data, e_data):
        """Check if window data is valid (no NaN or gaps)."""
        if len(z_data) != len(n_data) or len(z_data) != len(e_data):
            return False
            
        expected_samples = int(self.padded_window_length * self.sampling_rate)
        if len(z_data) != expected_samples:
            return False
            
        if np.any(np.isnan(z_data)) or np.any(np.isnan(n_data)) or np.any(np.isnan(e_data)):
            return False
            
        if np.all(z_data == 0) or np.all(n_data == 0) or np.all(e_data == 0):
            return False
            
        return True
    
    def _preprocess_trace(self, data):
        """Apply bandpass filter in frequency domain and remove padding. Then demean and normalize"""

        f = np.fft.fftfreq(len(data), d=1/self.sampling_rate)
        xw = np.fft.fft(data)
        mask = (np.abs(f) < self.freqmin) | (np.abs(f) > self.freqmax)
        xw[mask] = 0
        filtered = np.real(np.fft.ifft(xw)).astype(np.float32)
        
        # Remove padding
        samples_to_crop = int(self.padding * self.sampling_rate)
        processed = filtered[samples_to_crop:-samples_to_crop]
        
        # Demean and normalize
        processed -= np.mean(processed)
        norm = np.sqrt(np.sum(np.square(processed)))
        processed = processed / (1e-10 + norm)
        
        return processed
    
    def _check_earthquake_in_window(self, station_name, window_start, window_length):
        """Check if there's an earthquake in the current window."""
        window_end = window_start + window_length
        
        # Filter catalog for this station
        station_events = self.catalog[self.catalog['station'] == station_name]
        
        for _, event in station_events.iterrows():
            if pd.notna(event.get('p_arrival_time')):
                p_time = UTCDateTime(event['p_arrival_time'])
                if window_start <= p_time <= window_end:
                    # Calculate sample position
                    p_sample = int((p_time - window_start) * self.sampling_rate)
                    
                    # Get S arrival if available
                    s_sample = None
                    if pd.notna(event.get('s_arrival_time')):
                        s_time = UTCDateTime(event['s_arrival_time'])
                        if window_start <= s_time <= window_end:
                            s_sample = int((s_time - window_start) * self.sampling_rate)
                    
                    return 'eq', p_sample, s_sample
        
        return 'no', None, None


def create_continuous_dataset(mseed_dir, catalog_csv, output_dir):
    """
    Create continuous dataset files.
    
    Args:
        mseed_dir: Directory containing MSEED files
        catalog_csv: Path to earthquake catalog CSV
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_hdf5 = os.path.join(output_dir, "continuous_waveforms.hdf5")
    output_metadata = os.path.join(output_dir, "continuous_metadata.csv")
    
    preprocessor = ContinuousDataPreprocessor(
        mseed_dir=mseed_dir,
        catalog_csv=catalog_csv,
        output_hdf5_path=output_hdf5,
        output_metadata_csv_path=output_metadata
    )
    
    preprocessor.process_all_stations()
    
    return output_hdf5, output_metadata

