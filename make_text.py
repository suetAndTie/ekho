import os
import argparse
from pathlib import Path
from csv import writer


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
                line = f.readline()
                while line:
                    fn, text = line.split(' ', 1)
                    fn = str(Path(item.parent, fn + file_type))
                    row = (fn, text.strip()) # strip \n from text
                    csv_writer.writerow(row)
                    line = f.readline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/LibriSpeech', help="Directory containing the dataset")
    parser.add_argument('--file_type', default='.flac', help='Type of audio file')
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    
    output_fn = os.path.join(args.data_dir, 'text.csv')
    
    with open(output_fn, 'w') as csvfile:
        csv_writer = writer(csvfile)
        add_text(csv_writer, data_path, args.file_type)

