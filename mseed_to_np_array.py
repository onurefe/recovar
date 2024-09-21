import os
import numpy as np
import logging
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.exceptions import FDSNNoDataException

# -------------------------------
# Configuration Parameters
# -------------------------------
class Config:
    # Magnitude range for earthquake selection
    MAG_MIN = 3.5
    MAG_MAX = 6.5

    # Sampling and window parameters
    WINDOW_SEC = 30
    SAMPLING_RATE = 100  # Hz
    NUM_POINTS = WINDOW_SEC * SAMPLING_RATE  # 3000

    # Data parameters
    CHANNELS = ['BHN', 'BHE', 'BHZ']
    DATA_DIR = 'mseed_files'
    OUTPUT_FILE = 'seismic_data.npy'
    NETWORK = "IU"
    STATION = "*"  # Wildcard to select all stations
    LOCATION = ""

    # Earthquake search parameters
    NUM_SAMPLES = 1024  # Total number of samples to process

    # Time window around the earthquake origin time
    TIME_BEFORE = 5  # Seconds before origin
    TIME_AFTER = TIME_BEFORE + WINDOW_SEC  # Total window


# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


# -------------------------------
# Function Definitions
# -------------------------------
def search_earthquakes(client, start_time, end_time, mag_min, mag_max, limit):
    """
    Searches for earthquake events within specified magnitude and time range.

    Parameters:
        client (Client): ObsPy FDSN client.
        start_time (UTCDateTime): Start time for the search.
        end_time (UTCDateTime): End time for the search.
        mag_min (float): Minimum magnitude.
        mag_max (float): Maximum magnitude.
        limit (int): Maximum number of events to retrieve.

    Returns:
        list: List of earthquake events.
    """
    try:
        events = client.get_events(
            starttime=start_time,
            endtime=end_time,
            minmagnitude=mag_min,
            maxmagnitude=mag_max,
            orderby="time-asc",
            limit=limit
        )
        logging.info(f"Found {len(events)} earthquake events.")
        return events
    except Exception as e:
        logging.error(f"Error fetching earthquake events: {e}")
        return []


def download_waveform(client, event, network, station, location, channels, start_time, end_time):
    """
    Downloads waveform data for a specific earthquake event.

    Parameters:
        client (Client): ObsPy FDSN client.
        event: Earthquake event object.
        network (str): Network code.
        station (str): Station code.
        location (str): Location code.
        channels (list): List of channel codes.
        start_time (UTCDateTime): Start time for waveform data.
        end_time (UTCDateTime): End time for waveform data.

    Returns:
        Stream or None: ObsPy Stream object containing waveform data, or None if download fails.
    """
    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel="BH*",
            starttime=start_time,
            endtime=end_time,
            attach_response=True
        )
        logging.debug(f"Downloaded waveform data for event at {event.origins[0].time}")
        return st
    except FDSNNoDataException:
        logging.warning(f"No data available for event at {event.origins[0].time}")
        return None
    except Exception as e:
        logging.error(f"Error downloading waveform for event at {event.origins[0].time}: {e}")
        return None


def preprocess_waveform(stream, channels, sampling_rate, start_time, end_time, num_points):
    """
    Processes waveform data: selects channels, resamples, trims, and pads/trims data.

    Parameters:
        stream (Stream): ObsPy Stream object containing waveform data.
        channels (list): List of desired channel codes.
        sampling_rate (int): Desired sampling rate in Hz.
        start_time (UTCDateTime): Start time for trimming.
        end_time (UTCDateTime): End time for trimming.
        num_points (int): Number of data points per channel.

    Returns:
        numpy.ndarray or None: Processed data array of shape (num_points, num_channels), or None if processing fails.
    """
    try:
        # Select required channels
        st_filtered = Stream()
        for channel in channels:
            st_filtered += stream.select(channel=channel)

        if len(st_filtered) != len(channels):
            logging.warning(f"Incomplete channels. Expected {len(channels)}, got {len(st_filtered)}.")
            return None

        # Merge streams if necessary
        st_filtered.merge(method=1, fill_value='interpolate')

        # Resample to desired sampling rate
        st_filtered.resample(sampling_rate)

        # Trim to exact window
        st_filtered.trim(starttime=start_time, endtime=end_time)

        # Ensure each trace has exactly num_points
        for tr in st_filtered:
            if len(tr.data) < num_points:
                tr.data = np.pad(tr.data, (0, num_points - len(tr.data)), 'constant')
            elif len(tr.data) > num_points:
                tr.data = tr.data[:num_points]

        # Stack channels into a single array with shape (num_points, num_channels)
        data = np.vstack([st_filtered.select(channel=ch)[0].data for ch in channels]).T
        return data
    except Exception as e:
        logging.error(f"Error preprocessing waveform: {e}")
        return None


def stack_channels(data_list, expected_shape=(None, 3000, 3)):
    """
    Converts a list of data arrays into a single NumPy array.

    Parameters:
        data_list (list): List of NumPy arrays to stack.
        expected_shape (tuple): Expected shape of the final array.

    Returns:
        numpy.ndarray: Stacked data array.
    """
    try:
        seismic_data = np.array(data_list)
        if expected_shape[0] is not None and seismic_data.shape[0] > expected_shape[0]:
            seismic_data = seismic_data[:expected_shape[0]]
        logging.info(f"Final seismic data shape: {seismic_data.shape}")
        return seismic_data
    except Exception as e:
        logging.error(f"Error stacking channels: {e}")
        return None


def save_seismic_data(seismic_data, output_file):
    """
    Saves seismic data to a NumPy file.

    Parameters:
        seismic_data (numpy.ndarray): Seismic data array to save.
        output_file (str): Path to the output file.

    Returns:
        None
    """
    try:
        np.save(output_file, seismic_data)
        logging.info(f"Saved seismic data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving seismic data: {e}")


# -------------------------------
# Main Execution Flow
# -------------------------------
def main():
    """
    Main function to orchestrate the workflow.
    """
    # Setup logging
    setup_logging()

    # Ensure data directory exists
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    # Initialize FDSN client
    client = Client("IRIS")

    # Define search time range
    end_time = UTCDateTime.now()
    start_time = end_time - 365 * 24 * 60 * 60  # 1 year ago

    # Search for earthquake events
    events = search_earthquakes(
        client=client,
        start_time=start_time,
        end_time=end_time,
        mag_min=Config.MAG_MIN,
        mag_max=Config.MAG_MAX,
        limit=Config.NUM_SAMPLES
    )

    if not events:
        logging.error("No earthquake events found. Exiting.")
        return

    # Initialize list to hold processed data
    data_list = []

    # Iterate over each event
    for idx, event in enumerate(events, start=1):
        origin_time = event.origins[0].time
        start_waveform = origin_time - Config.TIME_BEFORE
        end_waveform = origin_time + Config.TIME_AFTER

        logging.debug(f"Processing event {idx}: Origin time {origin_time}")

        # Download waveform data
        stream = download_waveform(
            client=client,
            event=event,
            network=Config.NETWORK,
            station=Config.STATION,
            location=Config.LOCATION,
            channels=Config.CHANNELS,
            start_time=start_waveform,
            end_time=end_waveform
        )

        if stream is None:
            continue  # Skip to the next event if download failed

        # Preprocess waveform data
        processed_data = preprocess_waveform(
            stream=stream,
            channels=Config.CHANNELS,
            sampling_rate=Config.SAMPLING_RATE,
            start_time=start_waveform,
            end_time=end_waveform,
            num_points=Config.NUM_POINTS
        )

        if processed_data is None:
            logging.warning(f"Skipping event {idx}: Preprocessing failed.")
            continue

        # Append processed data to the list
        data_list.append(processed_data)
        logging.info(f"Processed {len(data_list)}/{Config.NUM_SAMPLES} events.")

        # Check if the desired number of samples is reached
        if len(data_list) >= Config.NUM_SAMPLES:
            logging.info("Reached the desired number of samples. Stopping.")
            break

    # Stack channels into a single NumPy array
    seismic_data = stack_channels(data_list, expected_shape=(Config.NUM_SAMPLES, Config.NUM_POINTS, len(Config.CHANNELS)))

    if seismic_data is None or seismic_data.size == 0:
        logging.error("No seismic data was processed successfully. Exiting.")
        return

    # Save the seismic data to a file
    save_seismic_data(seismic_data, Config.OUTPUT_FILE)


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
