import h5py
import pandas as pd
import numpy as np
import json
from pathlib import Path

def merge_datasets(dataset_configs, output_name, output_dir="/home/ege/recovar/benchmarks/output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_metadata_path = output_dir / f"{output_name}_metadata.csv"
    output_waveforms_path = output_dir / f"{output_name}_waveforms.hdf5"
    
    all_metadata = []
    
    for i, config in enumerate(dataset_configs):
        print(f"Processing {config['name']}...")
        metadata = pd.read_csv(config['metadata_path'])
        metadata['source_dataset'] = config['name']
        metadata['trace_name'] = metadata['trace_name'].astype(str) + f"_from_{config['name']}"
        all_metadata.append(metadata)
    
    print("Merging metadata...")
    merged_metadata = pd.concat(all_metadata, ignore_index=True)
    merged_metadata.to_csv(output_metadata_path, index=False)
    print(f"Merged metadata: {len(merged_metadata)} rows")
    print(f"Earthquake percentage: {(merged_metadata['trace_category'] == 'earthquake_local').mean():.1%}")
    
    print("Merging waveforms...")
    with h5py.File(output_waveforms_path, 'w') as output_h5:
        output_group = output_h5.create_group('data')
        
        for config in dataset_configs:
            print(f"  Copying waveforms from {config['name']}...")
            with h5py.File(config['waveforms_path'], 'r') as input_h5:
                data_group = input_h5['data']
                trace_names = list(data_group.keys())
                
                for idx, trace_name in enumerate(trace_names):
                    if idx % 20000 == 0:
                        print(f"    Progress: {idx}/{len(trace_names)}")
                        output_h5.flush()
                    
                    new_trace_name = f"{trace_name}_from_{config['name']}"
                    data = data_group[trace_name][:]
                    output_group.create_dataset(
                        new_trace_name, 
                        data=data, 
                        dtype=np.float32,
                        compression='gzip',  
                        compression_opts=1   
                    )
    print(f"  Metadata: {output_metadata_path}")
    print(f"  Waveforms: {output_waveforms_path}")
    print(f"  Total samples: {len(merged_metadata)}")
    
    return str(output_metadata_path), str(output_waveforms_path)

def merge_all_custom_datasets(output_name):
    with open('/home/ege/recovar/reproducibility/settings.json', 'r') as f:
        settings = json.load(f)
    
    dataset_configs = []
    for name, config in settings['CUSTOM_DATASETS'].items():
        dataset_configs.append({
            'name': name,
            'metadata_path': config['metadata'],
            'waveforms_path': config['waveforms']
        })
    
    return merge_datasets(dataset_configs, output_name)

if __name__ == "__main__":
    output_name = "MERGED_fixed"
    
    print("Available datasets:")
    with open('/home/ege/recovar/reproducibility/settings.json', 'r') as f:
        settings = json.load(f)
    
    dataset_names = list(settings['CUSTOM_DATASETS'].keys())
    for name in dataset_names:
        print(f"  - {name}")
    
    print(f"\nMerging all datasets into: {output_name}")
    metadata_path, waveforms_path = merge_all_custom_datasets(output_name)
    
    settings['CUSTOM_DATASETS'][output_name] = {
        'metadata': metadata_path,
        'waveforms': waveforms_path
    }
    
    with open('/home/ege/recovar/reproducibility/settings.json', 'w') as f:
        json.dump(settings, f, indent=4)