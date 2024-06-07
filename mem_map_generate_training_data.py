# based on 
# https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    
    dataset_shape = list(data_list[0].shape)
    total_size_last_axis = dataset_shape[-1] * len(data_list)

    # Create the final array with the shape for concatenation along the last axis
    final_shape = dataset_shape[:-1] + [total_size_last_axis]
    memmap_array = np.memmap("test.dat", dtype=np.float64, mode='w+', shape=tuple(final_shape))
    
    
   
   

    print("\rStep 1 Starting", end="", flush=True)
    current_position = 0
    for chunk in data_list:
        chunk_size = chunk.shape[-1]
        memmap_array[..., current_position:current_position + chunk_size] = chunk[...]
        current_position += chunk_size
    
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
   


    temp = data[min_t + x_offsets, ...].shape
   
    x_array = np.memmap("temp_x.dat", dtype=np.float64, mode='w+', shape=(max_t - min_t, temp[0], temp[1], temp[2]))
    y_array = np.memmap("temp_y.dat", dtype=np.float64, mode='w+', shape=(max_t - min_t, temp[0], temp[1], temp[2]))
    
    i = 0
    total = max_t - min_t
    for t in range(min_t, max_t):
        print(f"\rStep 2 Starting: {i} / {total}", end="", flush=True)
        x_array[i] = data[t + x_offsets, ...]
        y_array[i] = data[t + y_offsets, ...]
        i+= 1
        
        
       
    
    num_samples = x_array.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_array_train, y_array_train = x_array[:num_train], y_array[:num_train]

    # val
    x_array_val, y_array_val = (
        x_array[num_train: num_train + num_val],
        y_array[num_train: num_train + num_val],
    )

    # test
    x_array_test, y_array_test = x_array[-num_test:], y_array[-num_test:]


    for cat in ["train", "val", "test"]:
        # _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _x1, _y1 = locals()["x_array_" + cat], locals()["y_array_" + cat]
        print("\rSaving ", cat, "x: ", _x1.shape, "y:", _y1.shape, end="", flush=True)
        
       
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x1,
            y=_y1,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        
   


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)

    print("\rFile Opened", end="", flush=True)
    
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
   
    generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    


def main(args):
    # print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
