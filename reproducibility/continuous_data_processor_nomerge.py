import numpy as np
import obspy
from obspy import UTCDateTime
import h5py
import pandas as pd
import os
from scipy.signal import detrend


class ContinuousDataPreprocessor:
    def __init__(self,
                 catalog_csv,
                 output_hdf5_path,
                 output_metadata_csv_path,
                 window_length=60,
                 sampling_rate=100,
                 freqmin=1,
                 freqmax=20
                ):
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
        # Group files by time range (assumes filename pattern: NET.STA..CHA__STARTTIME__ENDTIME.mseed)
        import glob
        mseed_files = glob.glob(os.path.join(station_dir, "*.mseed"))

        if len(mseed_files) == 0:
            print(f"No mseed files found in {station_dir}")
            return

        # Group files by their time signature (everything except channel)
        file_groups = {}
        for filepath in mseed_files:
            filename = os.path.basename(filepath)
            # Extract time signature by removing channel identifier
            # Pattern: NET.STA..CHA__TIME__TIME.mseed -> group by NET.STA..CHA__TIME__TIME
            parts = filename.split('..')
            if len(parts) >= 2:
                # Replace channel with wildcard for grouping
                time_sig = parts[1]  # e.g., "HHN__20191007T235958410000Z__20191009T000003530000Z.mseed"
                time_sig = time_sig[3:]  # Remove channel identifier (HHN -> "")
                group_key = parts[0] + time_sig  # e.g., "KO.SLVT__20191007T235958410000Z__20191009T000003530000Z.mseed"

                if group_key not in file_groups:
                    file_groups[group_key] = []
                file_groups[group_key].append(filepath)

        print(f"Found {len(file_groups)} time-matched file groups")

        metadata = []
        with h5py.File(self.output_hdf5_path, 'a') as h5f:
            if 'data' not in h5f:
                data_group = h5f.create_group('data')
            else:
                data_group = h5f['data']

            # Process each group of time-matched files
            for group_key, file_list in file_groups.items():
                try:
                    stream = obspy.Stream()
                    for filepath in file_list:
                        stream += obspy.read(filepath)

                    if len(stream) == 0:
                        continue

                    for tr in stream:
                        tr.data = tr.data.astype(np.float32)
                        if tr.stats.sampling_rate != self.sampling_rate:
                            tr.resample(self.sampling_rate)

                    station_name = stream[0].stats.station
                    network = stream[0].stats.network

                    z_stream = stream.select(channel="*Z")
                    n_stream = stream.select(channel="*N")
                    e_stream = stream.select(channel="*E")

                    if not (z_stream and n_stream and e_stream):
                        z_stream = stream.select(channel="*HZ")
                        n_stream = stream.select(channel="*HN")
                        e_stream = stream.select(channel="*HE")

                    if not (z_stream and n_stream and e_stream):
                        print(f"  Skipping group {group_key}: missing Z/N/E channels")
                        continue

                    # Use first trace from each channel (should be the only one in this group)
                    z_trace = z_stream[0]
                    n_trace = n_stream[0]
                    e_trace = e_stream[0]

                    common_start = max(z_trace.stats.starttime, n_trace.stats.starttime, e_trace.stats.starttime)
                    common_end = min(z_trace.stats.endtime, n_trace.stats.endtime, e_trace.stats.endtime)

                    if common_end - common_start < self.window_length:
                        print(f"  Skipping group: overlap too short ({common_end - common_start}s)")
                        continue

                    z_trimmed = z_trace.copy().trim(common_start, common_end)
                    n_trimmed = n_trace.copy().trim(common_start, common_end)
                    e_trimmed = e_trace.copy().trim(common_start, common_end)

                    segment_metadata = self._process_segment(
                        z_trimmed, n_trimmed, e_trimmed,
                        station_name, network, data_group
                    )
                    metadata.extend(segment_metadata)

                except Exception as e:
                    print(f"  Error processing group {group_key}: {e}")
                    continue

        if metadata:
            metadata_df = pd.DataFrame(metadata)
            if os.path.exists(self.output_metadata_csv_path):
                metadata_df.to_csv(self.output_metadata_csv_path, mode='a', header=False, index=False)
            else:
                metadata_df.to_csv(self.output_metadata_csv_path, index=False)

            print(f"Processed {len(metadata)} windows from station {station_name}")

    def _find_matching_trace(self, stream, start_time, end_time):
        for trace in stream:
            if trace.stats.starttime <= end_time and trace.stats.endtime >= start_time:
                return trace
        return None

    def _process_segment(self, z_trace, n_trace, e_trace, station_name, network, data_group):
        metadata = []

        start_time = z_trace.stats.starttime
        end_time = z_trace.stats.endtime
        current_time = start_time

        while current_time + self.window_length <= end_time:
            window_end = current_time + self.window_length

            expected_samples = int(self.window_length * self.sampling_rate)
            z_win = z_trace.slice(current_time, window_end).data[:expected_samples]
            n_win = n_trace.slice(current_time, window_end).data[:expected_samples]
            e_win = e_trace.slice(current_time, window_end).data[:expected_samples]

            if self._is_window_valid(z_win, n_win, e_win):
                z_processed = self._preprocess_trace(z_win)
                n_processed = self._preprocess_trace(n_win)
                e_processed = self._preprocess_trace(e_win)

                # Stack as: (timesteps, channels)
                window_data = np.stack([e_processed, n_processed, z_processed], axis=-1)

                label, p_sample, s_sample = self._check_earthquake_in_window(
                    station_name, current_time, self.window_length
                )

                trace_name = f"{network}.{station_name}.{current_time.strftime('%Y%m%d_%H%M%S')}"

                data_group.create_dataset(trace_name, data=window_data.astype(np.float32))

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

        return metadata

    def _is_window_valid(self, z_data, n_data, e_data):
        for window_data in [z_data, n_data, e_data]:
            if np.ma.is_masked(window_data):
                if window_data.mask.any():
                    return False
                window_data = window_data.data

            if np.all(window_data == 0):
                return False

            if np.isinf(window_data).any():
                return False

        return True

    def _preprocess_trace(self, data):
        data = data - np.mean(data)
        data = detrend(data, type='linear')

        f = np.fft.fftfreq(len(data), d=1/self.sampling_rate)
        xw = np.fft.fft(data)
        mask = (np.abs(f) < self.freqmin) | (np.abs(f) > self.freqmax)
        xw[mask] = 0
        filtered = np.real(np.fft.ifft(xw)).astype(np.float32)

        return filtered

    def _check_earthquake_in_window(self, station_name, window_start, window_length):
        window_end = window_start + window_length

        station_events = self.catalog[self.catalog['station'] == station_name]

        for _, event in station_events.iterrows():
            if pd.notna(event.get('p_arrival_time')):
                p_time = UTCDateTime(event['p_arrival_time'])
                if window_start <= p_time <= window_end:
                    p_sample = int((p_time - window_start) * self.sampling_rate)

                    s_sample = None
                    if pd.notna(event.get('s_arrival_time')):
                        s_time = UTCDateTime(event['s_arrival_time'])
                        if window_start <= s_time <= window_end:
                            s_sample = int((s_time - window_start) * self.sampling_rate)

                    return 'eq', p_sample, s_sample

        return 'no', None, None
