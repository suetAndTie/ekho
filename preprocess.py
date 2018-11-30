'''
Adapted from
https://github.com/r9y9/deepvoice3_pytorch/blob/master/preprocess.py
'''
import os
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from csv import writer
from config import config
from importlib import import_module
from tqdm import tqdm



def add_text(csv_writer, path, file_type):
    """Recursively iterate to fill up dataframe
    """
    for item in path.iterdir():
        if item.is_dir():
            add_text(csv_writer, item, file_type)
        elif item.is_file():
            if item.suffix != '.txt':
                continue

            # Read file and add to dataframe
            with item.open() as f:
                for line in f:
                    fn, text = line.split(' ', 1)
                    speaker_id = fn.split('-')[0] # speaker is first part of file name
                    path = str(Path(item.parent, fn + file_type))
                    row = (path, speaker_id, text.strip()) # strip \n from text
                    csv_writer.writerow(row)

def preprocess(mod, in_dir, out_dir, num_workers):
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'data.csv'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(','.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    frame_shift_ms = config.hop_size / config.sample_rate * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='asr', type=str, help="Dataset to use (eg 'asr')")
    parser.add_argument('--data_dir', default='data/LibriSpeech', type=str, help="Directory containing the dataset")
    parser.add_argument('--out_dir', default='data/processed', type=str, help="Output directory of processed data")
    parser.add_argument('--file_type', default='.flac', type=str, help='Type of audio file')
    args = parser.parse_args()

    num_workers = cpu_count()
    data_path = Path(args.data_dir)

    # Make output folder if needed
    os.makedirs(args.out_dir, exist_ok=True)

    # Make text.csv (file_path, text)
    output_fn = os.path.join(args.out_dir, 'text.csv')
    with open(output_fn, 'w') as csvfile:
        csv_writer = writer(csvfile)
        add_text(csv_writer, data_path, args.file_type)

    module = import_module('util.'+args.dataset)
    preprocess(module, args.data_dir, args.out_dir, num_workers)

