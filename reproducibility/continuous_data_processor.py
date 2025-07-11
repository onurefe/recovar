import numpy as np
import obspy
from obspy import UTCDateTime
import h5py
import pandas as pd
import os
from obspy.signal import detrend

class ContinuousDataPreprocessor:
    """
    Preprocesses continuous MSEED data into HDF5 format compatible with KFold framework.
    Following STEAD structure: single HDF5 file with both earthquake and noise traces.
    """
    
    def __init__(self, 
                 catalog_csv,
                 output_hdf5_path,
                 output_metadata_csv_path,
                 window_length=60,
                 sampling_rate=100,
                 freqmin=1,
                 freqmax=20
                ):
        """
        Args:
            catalog_csv: CSV file with earthquake catalog (p_time, s_time, station columns)
            output_hdf5_path: Path for output HDF5 file
            output_metadata_csv_path: Path for output metadata CSV
            window_length: Window length in seconds (default 60 to match STEAD)
            sampling_rate: Target sampling rate (default 100 Hz)
            freqmin: Minimum frequency for bandpass filter
            freqmax: Maximum frequency for bandpass filter
        """
        self.catalog_csv = catalog_csv
        self.output_hdf5_path = output_hdf5_path
        self.output_metadata_csv_path = output_metadata_csv_path
        self.window_length = window_length
        self.sampling_rate = sampling_rate
        self.freqmin = freqmin
        self.freqmax = freqmax
        

        # Load catalog (IMPORTANT: Change format to fit your metadata.)
        self.catalog = pd.read_csv(catalog_csv)
        if 'p_arrival_time' in self.catalog.columns:
            self.catalog['p_arrival_time'] = pd.to_datetime(self.catalog['p_arrival_time'], format='ISO8601')
        if 's_arrival_time' in self.catalog.columns:
            self.catalog['s_arrival_time'] = pd.to_datetime(self.catalog['s_arrival_time'], format='ISO8601')
        
        self.trace_counter = 0
        
    def process_station(self, station_dir):
        """
        Process all MSEED files in a station directory.
        
        Args:
            station_dir: Directory containing MSEED files for one station
        """
        try:
            stream = obspy.read(os.path.join(station_dir, "*.mseed"))
        except Exception as e:
            print(f"Error reading MSEED files in {station_dir}: {e}")
            return
        
        if len(stream) == 0:
            return
        for tr in stream: #Ensure everything is float before merging to prevent errors
            tr.data = tr.data.astype(np.float32)
        
        try:
            stream = stream.merge(fill_value=np.nan)
        except Exception as e:
            print(f"Error merging streams: {e}")
            return
            
        station_name = stream[0].stats.station
        network = stream[0].stats.network
        
        for tr in stream:
            if tr.stats.sampling_rate != self.sampling_rate:
                tr.resample(self.sampling_rate)
        
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
            return
            
        # Use first trace of each component -> might need to check this again. May be redundant.
        z_trace = z_trace[0]
        n_trace = n_trace[0]
        e_trace = e_trace[0]
        
        start_time = max(z_trace.stats.starttime, n_trace.stats.starttime, e_trace.stats.starttime)
        end_time = min(z_trace.stats.endtime, n_trace.stats.endtime, e_trace.stats.endtime)
        
        # Trim to common time
        z_trace.trim(start_time, end_time)
        n_trace.trim(start_time, end_time)
        e_trace.trim(start_time, end_time)
        
        metadata = []
        current_time = start_time
        
        with h5py.File(self.output_hdf5_path, 'a') as h5f:
            if 'data' not in h5f:
                data_group = h5f.create_group('data')
            else:
                data_group = h5f['data']
                
            while current_time + self.window_length <= end_time:
                window_end = current_time + self.window_length

                expected_samples = int(self.window_length * self.sampling_rate)
                z_win = z_trace.slice(current_time, window_end).data[:expected_samples]
                n_win = n_trace.slice(current_time, window_end).data[:expected_samples]
                e_win = e_trace.slice(current_time, window_end).data[:expected_samples]
                
                if self._is_window_valid(z_win, n_win, e_win):
                    # Process traces separately 
                    z_processed = self._preprocess_trace(z_win)
                    n_processed = self._preprocess_trace(n_win)
                    e_processed = self._preprocess_trace(e_win)
                    
                    # Stack as (timesteps, channels) to be compatible with stead
                    window_data = np.stack([e_processed, n_processed, z_processed], axis=-1)
                    
                    # Check for earthquakes
                    label, p_sample, s_sample = self._check_earthquake_in_window(
                        station_name, current_time, self.window_length
                    )
                    
                    # Create trace name
                    trace_name = f"{network}.{station_name}.{current_time.strftime('%Y%m%d_%H%M%S')}"
                    
                    # Save to HDF5
                    data_group.create_dataset(trace_name, data=window_data.astype(np.float32))
                    
                    # Add metadata
                    metadata.append({
                        'trace_name': trace_name,
                        'station_name': station_name,
                        'network': network,
                        'trace_start_time': current_time.isoformat(),
                        'label': label,
                        'p_arrival_sample': p_sample if p_sample is not None else np.nan,
                        's_arrival_sample': s_sample if s_sample is not None else np.nan,
                        'source_id': trace_name if label == 'no' else f"eq_{self.trace_counter}",
                        'trace_category': 'earthquake_local' if label == 'eq' else 'noise',
                    })
                    
                    self.trace_counter += 1
                    
                current_time += self.window_length
        
        if metadata:
            metadata_df = pd.DataFrame(metadata)
            if os.path.exists(self.output_metadata_csv_path):
                metadata_df.to_csv(self.output_metadata_csv_path, mode='a', header=False, index=False)
            else:
                metadata_df.to_csv(self.output_metadata_csv_path, index=False)
            
            print(f"Processed {len(metadata)} windows from station {station_name}")
    
    def _is_window_valid(self, z_data, n_data, e_data):
        """Check if window data is valid (no NaN or gaps)."""
        for window_data in [z_data, n_data, e_data]:
            if np.ma.is_masked(window_data):
                if window_data.mask.any():
                    return False
                # If it's masked but has no actual masked values, extract the data
                window_data = window_data.data
            
            if np.isnan(window_data).any():
                return False
            
            if np.all(window_data == 0):
                return False
            
            if np.isinf(window_data).any():
                return False
        
        return True
    def _preprocess_trace(self, data):
        """Apply bandpass filter in frequency domain with demeaning."""
        
        # Demean and linear detrend before filtering
        data = data - np.mean(data)
        data = detrend(data, type='linear')
        
        f = np.fft.fftfreq(len(data), d=1/self.sampling_rate)
        xw = np.fft.fft(data)
        mask = (np.abs(f) < self.freqmin) | (np.abs(f) > self.freqmax)
        xw[mask] = 0
        filtered = np.real(np.fft.ifft(xw)).astype(np.float32)

        return filtered
    
    def _check_earthquake_in_window(self, station_name, window_start, window_length):
        """Check if there's an earthquake in the current window."""
        window_end = window_start + window_length
        
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