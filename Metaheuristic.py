import numpy as np
import pandas as pd
import time
import os
from numba import cuda, jit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import concurrent.futures
import random
import psutil
import gc

def plot_strip_single(my_df,bin_width):
    y_pos = max(my_df['y-start'])
    layer = max(my_df['level'])
    strip_height = y_pos + max(my_df.loc[my_df['level'] == layer, 'Height'])

    counter2 = 0

    fig, ax = plt.subplots()
    xx = my_df['x-start'].values
    yy = my_df['y-start'].values
    random.seed(42)
    for y in my_df['Height']:
        hex_color = '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        ax.add_patch(
            Rectangle((xx[counter2], yy[counter2]), my_df['Length'].loc[counter2], my_df['Height'].loc[counter2], color=hex_color))
        counter2 += 1
    plt.axis('equal')
    ax.plot([0, bin_width], [0, 0], color='black', linewidth=3)
    ax.plot([0, 0], [0, strip_height], color='black', linewidth=3)
    ax.plot([bin_width, bin_width], [0, strip_height], color='black', linewidth=3)
    plt.xlim([-1, bin_width + 1])
    plt.ylim([-1, strip_height + 1])
    plt.show()


def next_fit_packing(dff_test, bin_width):
    # data input and setup
    num_rect = len(dff_test)
    # input_rect = np.random.randint(1,5, size=(num_rect,2))
    # input_rect = np.array([[2,3],[2,4],[3,3],[5,1],[2,2],[4,3]])
    input_rect = dff_test[['Length', 'Height']].to_numpy()
    total_shapes = len(input_rect)
    data = {'Length': input_rect[:, 0],
            'Height': input_rect[:, 1],
            'level': np.zeros(len(input_rect)) + num_rect,
            'x-start': np.zeros(len(input_rect)),
            'y-start': np.zeros(len(input_rect))}
    my_df = pd.DataFrame(data)

    # variables initialisation
    counter = 0
    # bin_width = 8
    layer = 0
    y_pos = 0
    x_pos = 0

    # formulation
    for x in input_rect:
        if bin_width - x_pos >= input_rect[counter, 0]:
            my_df.loc[[counter], 'x-start'] = x_pos
            my_df.loc[[counter], 'y-start'] = y_pos
            my_df.loc[[counter], 'level'] = layer
            x_pos += input_rect[counter, 0]
            counter += 1
        else:
            x_pos = 0
            y_pos += max(my_df.loc[my_df['level'] == layer, 'Height'])
            layer += 1
            my_df.loc[[counter], 'level'] = layer
            my_df.loc[[counter], 'x-start'] = x_pos
            my_df.loc[[counter], 'y-start'] = y_pos
            x_pos += input_rect[counter, 0]
            counter += 1
    return my_df

def best_fit_packing(dff_test, bin_width):
    num_rect = len(dff_test)
    input_rect = dff_test[['Length', 'Height']].to_numpy()
    total_shapes = len(input_rect)
    data = {'Length': input_rect[:, 0],
            'Height': input_rect[:, 1],
            'level': np.zeros(len(input_rect)) + num_rect,
            'x-start': np.zeros(len(input_rect)),
            'y-start': np.zeros(len(input_rect))}
    my_df = pd.DataFrame(data)
    return_data = {'level': range(0,len(input_rect)),
                   'space remaining': np.zeros(len(input_rect)),
                   'vertical space': np.zeros(len(input_rect))}
    return_df = pd.DataFrame(return_data)

    # variables initialisation
    counter = 0
    #bin_width = 8
    layer = 0
    y_pos = 0
    x_pos = 0

    # formulation
    for x in input_rect:

        possible_indices_width = np.where(return_df['space remaining'] >= input_rect[counter, 0])[0]
        possible_indices_height = np.where(return_df['vertical space'] >= input_rect[counter, 1])[0]
        possible_indices = np.intersect1d(possible_indices_width, possible_indices_height)

        if possible_indices.size > 0:

            best_fit = min(return_df.loc[possible_indices, 'space remaining'])

            best_fit_indices_width = np.where(return_df['space remaining'] == best_fit)[0]
            best_possible_indices = np.intersect1d(best_fit_indices_width, possible_indices_height)
            return_to_layer = best_possible_indices[0]

            ##### update rectangle to previous layer position #####
            space_remaining = return_df.loc[return_df['level'] == return_to_layer, 'space remaining'].values
            x_pos_return = bin_width - space_remaining[0]
            #print(x_pos_return)
            return_df.loc[return_df['level'] == return_to_layer, 'space remaining'] = bin_width - x_pos_return - input_rect[counter, 0]  # update remaining space
            my_df.loc[[counter], 'x-start'] = x_pos_return

            y_pos_return = my_df.loc[my_df['level'] == return_to_layer, 'y-start'].values
            my_df.loc[[counter], 'y-start'] = y_pos_return[0]
            my_df.loc[[counter], 'level'] = return_to_layer
            counter += 1

        elif bin_width - x_pos >= input_rect[counter, 0]:
            my_df.loc[[counter], 'x-start'] = x_pos
            my_df.loc[[counter], 'y-start'] = y_pos
            my_df.loc[[counter], 'level'] = layer
            x_pos += input_rect[counter, 0]
            counter += 1
        else:
            return_df.loc[return_df['level'] == layer, 'space remaining'] = bin_width - x_pos
            x_pos = 0
            y_pos += max(my_df.loc[my_df['level'] == layer, 'Height'])
            return_df.loc[return_df['level'] == layer, 'vertical space'] = max(my_df.loc[my_df['level'] == layer, 'Height'])
            layer += 1
            my_df.loc[[counter], 'level'] = layer
            my_df.loc[[counter], 'x-start'] = x_pos
            my_df.loc[[counter], 'y-start'] = y_pos
            x_pos += input_rect[counter, 0]
            counter += 1
    #print(my_df)
    return my_df

def order_randomiser(test_df):
    # Shuffle the DataFrame rows
    shuffled_df = test_df.sample(frac=1).reset_index(drop=True)

    return shuffled_df

def sort_maxdimension_width(dff_test):
    input_rect = dff_test[['Length', 'Height']].to_numpy()
    counter3 = 0
    for x in input_rect:
        if dff_test.at[counter3,'Length'] < dff_test.at[counter3,'Height']:
            temp = dff_test.at[counter3,'Length']
            dff_test.at[counter3, 'Length'] = dff_test.at[counter3,'Height']
            dff_test.at[counter3, 'Height'] = temp
        counter3 += 1

    return(dff_test.sort_values(
                by = "Length",
                ascending=False))

def sort_maxdimension_height(dff_test):
    input_rect = dff_test[['Length', 'Height']].to_numpy()
    counter3 = 0
    for x in input_rect:
        if dff_test.at[counter3,'Height'] < dff_test.at[counter3,'Length']:
            temp = dff_test.at[counter3,'Height']
            dff_test.at[counter3, 'Height'] = dff_test.at[counter3,'Length']
            dff_test.at[counter3, 'Length'] = temp
        counter3 += 1

    return(dff_test.sort_values(
                by = "Height",
                ascending=False))

def sort_width(dff_test):
    input_rect = dff_test[['Length', 'Height']].to_numpy()
    return (dff_test.sort_values(
        by="Length",
        ascending=False))

def sort_height(dff_test):
    input_rect = dff_test[['Length', 'Height']].to_numpy()
    return (dff_test.sort_values(
        by="Height",
        ascending=False))

def shelf_expansion_v2(current_df):
    expanded_df = current_df.copy()
    largest_dim = max(current_df['Height'].max(),current_df['Length'].max())

    layer_largest_dims = current_df.groupby('level')['Height'].max()
    shelf_spaces = largest_dim - layer_largest_dims
    y_start_increments = shelf_spaces.cumsum().shift(fill_value=0)

    expanded_df['increment'] = expanded_df['level'].map(y_start_increments)
    expanded_df['y-start'] += expanded_df['increment']
    expanded_df.drop(columns='increment', inplace=True)
    return expanded_df


def shelf_compression_v2(expanded_df):
    compressed_df = expanded_df.copy()
    largest_dim = max(expanded_df['Height'].max(), expanded_df['Length'].max())
    #total_layers = int(expanded_df['level'].max())
    #layers_array = np.arange(total_layers + 1)

    layer_largest_dims = expanded_df.groupby('level')['Height'].max()
    shelf_spaces = largest_dim - layer_largest_dims

    y_start_decrements = shelf_spaces.cumsum().shift(fill_value=0)
    '''
    for level, decrement in y_start_decrements.items():
        if decrement > 0:
            compressed_df.loc[compressed_df['level'] == level, 'y-start'] -= decrement
    '''
    compressed_df['decrement'] = compressed_df['level'].map(y_start_decrements)
    compressed_df['y-start'] -= compressed_df['decrement']
    compressed_df.drop(columns='decrement', inplace=True)
    return compressed_df

def height(my_df):
    max_heights = my_df.groupby('level')['Height'].max()
    strip_height = max_heights.sum()
    return strip_height

def monitor_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"Memory Usage: {memory.percent}% | CPU Usage: {cpu}%")

def run_simulated_annealing(x_encum_starting_value):
    ##### Simulated Annealing Function Framework #####

    bin_width = 200
    # Fixed predetermined parameters
    e_max, a_max, r_max = 10, 6, 6
    T, beta, alpha = 15, 0.97, 1.5

    # Initialize control variables
    t, e, a, r = 1, 1, 0, 0

    #Plotting vairables
    reheating, cooling, acc = 0, 0, 0
    heights, temps = [], []

    # 0. call best_fit_packing function for initial feasible sol.
    #x_current = shelf_expansion_v2(x_encum_starting_value)
    x_current = x_encum_starting_value
    x_encum = x_current.copy()
    heights.append(height(x_current))

    while e != e_max:
        if r == r_max:
            T = alpha * T
            e += 1
            a = 0
            r = 0
            reheating += 1
        elif a == a_max:
            T = beta * T
            e += 1
            a = 0
            r = 0
            cooling += 1
        else:
            #expanded_df = shelf_expansion_v2(x_current)
            expanded_df = x_current.copy()

            ########
            success = 0
            while success != 1:
                rand_outer_idx = np.random.randint(len(expanded_df))
                rand_outer_rect = expanded_df.iloc[rand_outer_idx]
                rand_outer_rect_length = rand_outer_rect['Length']
                rand_outer_rect_height = rand_outer_rect['Height']
                outer_layer = rand_outer_rect['level']

                outer_rects = expanded_df[expanded_df['level'] == outer_layer]
                inner_rects = expanded_df[expanded_df['level'] != outer_layer]

                rand_inner_idx = np.random.randint(len(inner_rects))
                rand_inner_rect = inner_rects.iloc[rand_inner_idx]
                rand_inner_rect_length = rand_inner_rect['Length']
                rand_inner_rect_height = rand_inner_rect['Height']

                outer_space = bin_width - outer_rects['Length'].sum()
                outer_space_open = rand_outer_rect_length + outer_space

                inner_layer = rand_inner_rect['level']
                inner_layer_rects = expanded_df[expanded_df['level'] == inner_layer]
                inner_space = bin_width - inner_layer_rects['Length'].sum()
                inner_space_open = rand_inner_rect_length + inner_space

                if (inner_space_open >= min(rand_outer_rect_length, rand_outer_rect_height) and
                        outer_space_open >= min(rand_inner_rect_length, rand_inner_rect_height)):
                    success = 1
                    if inner_space_open >= max(rand_outer_rect_length, rand_outer_rect_height):
                        x_start_inner = rand_inner_rect['x-start']
                        inner_shifted = inner_layer_rects[inner_layer_rects['x-start'] > x_start_inner].copy()
                        inner_shifted['x-start'] -= rand_inner_rect_length
                        expanded_df.loc[inner_shifted.index] = inner_shifted
                        new_inner_rect = rand_outer_rect.copy()
                        new_inner_rect['x-start'] = bin_width - inner_space_open
                        new_inner_rect['level'] = inner_layer
                        new_inner_rect['y-start'] = rand_inner_rect['y-start']
                        if np.random.randint(2) == 1:
                            new_inner_rect['Length'], new_inner_rect[
                                'Height'] = rand_outer_rect_height, rand_outer_rect_length
                        expanded_df.loc[new_inner_rect.name] = new_inner_rect
                    else:
                        x_start_inner = rand_inner_rect['x-start']
                        inner_shifted = inner_layer_rects[inner_layer_rects['x-start'] > x_start_inner].copy()
                        inner_shifted['x-start'] -= rand_inner_rect_length
                        expanded_df.loc[inner_shifted.index] = inner_shifted

                        new_inner_rect = rand_outer_rect.copy()
                        new_inner_rect['x-start'] = bin_width - inner_space_open
                        new_inner_rect['level'] = inner_layer
                        new_inner_rect['y-start'] = rand_inner_rect['y-start']
                        new_inner_rect['Length'], new_inner_rect['Height'] = min(rand_outer_rect_height,
                                                                                 rand_outer_rect_length), max(
                            rand_outer_rect_height, rand_outer_rect_length)
                        expanded_df.loc[new_inner_rect.name] = new_inner_rect

                    if outer_space_open >= max(rand_inner_rect_length, rand_inner_rect_height):
                        x_start_outer = rand_outer_rect['x-start']
                        outer_shifted = outer_rects[outer_rects['x-start'] > x_start_outer].copy()
                        outer_shifted['x-start'] -= rand_outer_rect_length
                        expanded_df.loc[outer_shifted.index] = outer_shifted

                        new_outer_rect = rand_inner_rect.copy()
                        new_outer_rect['x-start'] = bin_width - outer_space_open
                        new_outer_rect['level'] = outer_layer
                        new_outer_rect['y-start'] = rand_outer_rect['y-start']
                        if np.random.randint(2) == 1:
                            new_outer_rect['Length'], new_outer_rect[
                                'Height'] = rand_inner_rect_height, rand_inner_rect_length
                        expanded_df.loc[new_outer_rect.name] = new_outer_rect
                    else:
                        x_start_outer = rand_outer_rect['x-start']
                        outer_shifted = outer_rects[outer_rects['x-start'] > x_start_outer].copy()
                        outer_shifted['x-start'] -= rand_outer_rect_length
                        expanded_df.loc[outer_shifted.index] = outer_shifted

                        new_outer_rect = rand_inner_rect.copy()
                        new_outer_rect['x-start'] = bin_width - outer_space_open
                        new_outer_rect['level'] = outer_layer
                        new_outer_rect['y-start'] = rand_outer_rect['y-start']
                        new_outer_rect['Length'], new_outer_rect['Height'] = min(rand_inner_rect_height,
                                                                                 rand_inner_rect_length), max(
                            rand_inner_rect_height, rand_inner_rect_length)
                        expanded_df.loc[new_outer_rect.name] = new_outer_rect

            #x_neigh = shelf_compression_v2(expanded_df)
            x_neigh = expanded_df.copy()
            acc += 1
            if height(x_neigh) < height(x_current):
                x_current = x_neigh
                a += 1
                t += 1
                heights.append(height(x_current))  # store current solution height
                temps.append(T)
                if height(x_current) < height(x_encum):
                    x_encum = x_current
            elif np.exp(-(height(x_neigh) - height(x_current)) / T) >= random.random():
                x_current = x_neigh
                a += 1
                t += 1
                heights.append(height(x_current))  # store current solution height
                temps.append(T)
            else:
                r += 1
                t += 1
    gc.collect()  # Clean up memory after each batch
    monitor_resources()  # Monitor resource usage

    return height(x_encum)


#start = time.time()

#default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\T.xlsx"
#test = pd.read_excel(default_path, sheet_name="T1a")

# List of 10 different starting values for x_encum
#x_encum_starting_values = [
    # Add the first 4 initial values explicitly
#    shelf_expansion_v2(best_fit_packing(sort_height(test), 200)),
#    shelf_expansion_v2(best_fit_packing(sort_width(test), 200)),
#    shelf_expansion_v2(best_fit_packing(sort_maxdimension_height(test), 200)),
#    shelf_expansion_v2(best_fit_packing(sort_maxdimension_width(test), 200))
#] + [
    # Use list comprehension for the remaining 6 random values
#    shelf_expansion_v2(best_fit_packing(order_randomiser(test), 200)) for _ in range(6)
#]

# Run the algorithm in parallel
def run_all_simulations_in_parallel():
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(run_simulated_annealing, x_encum) for x_encum in x_encum_starting_values]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'A simulation generated an exception: {exc}')
    return results


# Execute the parallel runs
#results = run_all_simulations_in_parallel()


#end = time.time()

# Print the results
#for result in results:
 #   print(result)

#print(end-start)

#####################################
start = time.time()
default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\T.xlsx"
sheet_names = ['T1a', 'T1b', 'T1c', 'T1d', 'T1e', 'T2a', 'T2b', 'T2c', 'T2d', 'T2e',
               'T3a', 'T3b', 'T3c', 'T3d', 'T3e', 'T4a', 'T4b', 'T4c', 'T4d', 'T4e']

def run_all_simulations_in_parallel(test):

    def run_simulated_annealing(x_encum_starting_value):
        ##### Simulated Annealing Function Framework #####

        bin_width = 200
        # Fixed predetermined parameters
        e_max, a_max, r_max = 10, 15, 50
        T, beta, alpha = 15, 0.98, 1.5

        # Initialize control variables
        t, e, a, r = 1, 1, 0, 0

        # Plotting vairables
        reheating, cooling, acc = 0, 0, 0
        heights, temps = [], []

        # 0. call best_fit_packing function for initial feasible sol.
        # x_current = shelf_expansion_v2(x_encum_starting_value)
        x_current = x_encum_starting_value
        x_encum = x_current.copy()
        heights.append(height(x_current))

        while e != e_max:
            if r == r_max:
                T = alpha * T
                e += 1
                a = 0
                r = 0
                reheating += 1
            elif a == a_max:
                T = beta * T
                e += 1
                a = 0
                r = 0
                cooling += 1
            else:
                # expanded_df = shelf_expansion_v2(x_current)
                expanded_df = x_current.copy()

                ########
                success = 0
                while success != 1:
                    rand_outer_idx = np.random.randint(len(expanded_df))
                    rand_outer_rect = expanded_df.iloc[rand_outer_idx]
                    rand_outer_rect_length = rand_outer_rect['Length']
                    rand_outer_rect_height = rand_outer_rect['Height']
                    outer_layer = rand_outer_rect['level']

                    outer_rects = expanded_df[expanded_df['level'] == outer_layer]
                    inner_rects = expanded_df[expanded_df['level'] != outer_layer]

                    rand_inner_idx = np.random.randint(len(inner_rects))
                    rand_inner_rect = inner_rects.iloc[rand_inner_idx]
                    rand_inner_rect_length = rand_inner_rect['Length']
                    rand_inner_rect_height = rand_inner_rect['Height']

                    outer_space = bin_width - outer_rects['Length'].sum()
                    outer_space_open = rand_outer_rect_length + outer_space

                    inner_layer = rand_inner_rect['level']
                    inner_layer_rects = expanded_df[expanded_df['level'] == inner_layer]
                    inner_space = bin_width - inner_layer_rects['Length'].sum()
                    inner_space_open = rand_inner_rect_length + inner_space

                    if (inner_space_open >= min(rand_outer_rect_length, rand_outer_rect_height) and
                            outer_space_open >= min(rand_inner_rect_length, rand_inner_rect_height)):
                        success = 1
                        if inner_space_open >= max(rand_outer_rect_length, rand_outer_rect_height):
                            x_start_inner = rand_inner_rect['x-start']
                            inner_shifted = inner_layer_rects[inner_layer_rects['x-start'] > x_start_inner].copy()
                            inner_shifted['x-start'] -= rand_inner_rect_length
                            expanded_df.loc[inner_shifted.index] = inner_shifted
                            new_inner_rect = rand_outer_rect.copy()
                            new_inner_rect['x-start'] = bin_width - inner_space_open
                            new_inner_rect['level'] = inner_layer
                            new_inner_rect['y-start'] = rand_inner_rect['y-start']
                            if np.random.randint(2) == 1:
                                new_inner_rect['Length'], new_inner_rect[
                                    'Height'] = rand_outer_rect_height, rand_outer_rect_length
                            expanded_df.loc[new_inner_rect.name] = new_inner_rect
                        else:
                            x_start_inner = rand_inner_rect['x-start']
                            inner_shifted = inner_layer_rects[inner_layer_rects['x-start'] > x_start_inner].copy()
                            inner_shifted['x-start'] -= rand_inner_rect_length
                            expanded_df.loc[inner_shifted.index] = inner_shifted

                            new_inner_rect = rand_outer_rect.copy()
                            new_inner_rect['x-start'] = bin_width - inner_space_open
                            new_inner_rect['level'] = inner_layer
                            new_inner_rect['y-start'] = rand_inner_rect['y-start']
                            new_inner_rect['Length'], new_inner_rect['Height'] = min(rand_outer_rect_height,
                                                                                     rand_outer_rect_length), max(
                                rand_outer_rect_height, rand_outer_rect_length)
                            expanded_df.loc[new_inner_rect.name] = new_inner_rect

                        if outer_space_open >= max(rand_inner_rect_length, rand_inner_rect_height):
                            x_start_outer = rand_outer_rect['x-start']
                            outer_shifted = outer_rects[outer_rects['x-start'] > x_start_outer].copy()
                            outer_shifted['x-start'] -= rand_outer_rect_length
                            expanded_df.loc[outer_shifted.index] = outer_shifted

                            new_outer_rect = rand_inner_rect.copy()
                            new_outer_rect['x-start'] = bin_width - outer_space_open
                            new_outer_rect['level'] = outer_layer
                            new_outer_rect['y-start'] = rand_outer_rect['y-start']
                            if np.random.randint(2) == 1:
                                new_outer_rect['Length'], new_outer_rect[
                                    'Height'] = rand_inner_rect_height, rand_inner_rect_length
                            expanded_df.loc[new_outer_rect.name] = new_outer_rect
                        else:
                            x_start_outer = rand_outer_rect['x-start']
                            outer_shifted = outer_rects[outer_rects['x-start'] > x_start_outer].copy()
                            outer_shifted['x-start'] -= rand_outer_rect_length
                            expanded_df.loc[outer_shifted.index] = outer_shifted

                            new_outer_rect = rand_inner_rect.copy()
                            new_outer_rect['x-start'] = bin_width - outer_space_open
                            new_outer_rect['level'] = outer_layer
                            new_outer_rect['y-start'] = rand_outer_rect['y-start']
                            new_outer_rect['Length'], new_outer_rect['Height'] = min(rand_inner_rect_height,
                                                                                     rand_inner_rect_length), max(
                                rand_inner_rect_height, rand_inner_rect_length)
                            expanded_df.loc[new_outer_rect.name] = new_outer_rect

                # x_neigh = shelf_compression_v2(expanded_df)
                x_neigh = expanded_df.copy()
                acc += 1
                if height(x_neigh) < height(x_current):
                    x_current = x_neigh
                    a += 1
                    t += 1
                    heights.append(height(x_current))  # store current solution height
                    temps.append(T)
                    if height(x_current) < height(x_encum):
                        x_encum = x_current
                elif np.exp(-(height(x_neigh) - height(x_current)) / T) >= random.random():
                    x_current = x_neigh
                    a += 1
                    t += 1
                    heights.append(height(x_current))  # store current solution height
                    temps.append(T)
                else:
                    r += 1
                    t += 1
        gc.collect()  # Clean up memory after each batch
        #monitor_resources()  # Monitor resource usage

        return height(x_encum)

    x_encum_starting_values = [
                                  # Add the first 4 initial values explicitly
                                  shelf_expansion_v2(best_fit_packing(sort_height(test), 200)),
                                  shelf_expansion_v2(best_fit_packing(sort_width(test), 200)),
                                  shelf_expansion_v2(best_fit_packing(sort_maxdimension_height(test), 200)),
                                  shelf_expansion_v2(best_fit_packing(sort_maxdimension_width(test), 200))
                              ] + [
                                  # Use list comprehension for the remaining 6 random values
                                  shelf_expansion_v2(best_fit_packing(order_randomiser(test), 200)) for _ in range(6)
                              ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_simulated_annealing, x_encum) for x_encum in x_encum_starting_values]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'A simulation generated an exception: {exc}')
    return results

def process_sheet(sheet_name):
    df = pd.read_excel(default_path, sheet_name=sheet_name)
    results = run_all_simulations_in_parallel(df)
    return sheet_name, results

all_results = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_sheet = {executor.submit(process_sheet, sheet): sheet for sheet in sheet_names}
    for future in concurrent.futures.as_completed(future_to_sheet):
        sheet_name, results = future.result()
        all_results[sheet_name] = results

print(all_results)

# Assuming 'all_results' is the dictionary with sheet names as keys and results as values
results_data = []

# Create a list of tuples with the sheet name and the corresponding results
for sheet_name, results in all_results.items():
    for result in results:
        results_data.append((sheet_name, result))

# Convert the list to a DataFrame
df_results = pd.DataFrame(results_data, columns=['Sheet Name', 'Result'])

# Save the DataFrame to a single Excel sheet
output_file_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\MetaHeuristicResults\T.xlsx"
df_results.to_excel(output_file_path, index=False, sheet_name='T')

print(f'Results saved to {output_file_path}')
end = time.time()
print(end-start)