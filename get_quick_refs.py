import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some parameters')
    parser.add_argument('--save_path', type=str, help = 'enter the path where the runs will be saved')
    # parser.add_argument('--skip', type=bool, help = 'skips running the model again')

    args = parser.parse_args()

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = 'runs'

    path = save_path
    folders = os.listdir(path)
    rows = []
    for folder in folders:
        csv_path = os.path.join(path,folder,'quick_ref.csv')
        if os.path.exists(csv_path):
            rows.append(pd.read_csv(csv_path,index_col=0))
    quick_refs = pd.concat(rows)
    quick_refs.to_csv(os.path.join(path,'quick_refs.csv'))

if __name__ == "__main__":
    main()