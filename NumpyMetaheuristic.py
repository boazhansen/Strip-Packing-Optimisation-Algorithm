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

    layer_largest_dims = expanded_df.groupby('level')['Height'].max()
    shelf_spaces = largest_dim - layer_largest_dims

    y_start_decrements = shelf_spaces.cumsum().shift(fill_value=0)

    compressed_df['decrement'] = compressed_df['level'].map(y_start_decrements)
    compressed_df['y-start'] -= compressed_df['decrement']
    compressed_df.drop(columns='decrement', inplace=True)
    return compressed_df


def shelf_compression_numpy(expanded_array):
    # Copy the input array to avoid modifying the original
    compressed_array = np.copy(expanded_array)

    # Column indices based on the dataframe structure:
    # Assuming column 0 = 'Length', column 1 = 'Height', column 2 = 'level', column 3 = 'x-start', column 4 = 'y-start'
    HEIGHT_COL = 1
    LENGTH_COL = 0
    LEVEL_COL = 2
    Y_START_COL = 4

    # Find the largest dimension in the entire array (similar to max of 'Height' and 'Length' in dataframe)
    largest_dim = max(np.max(expanded_array[:, HEIGHT_COL]), np.max(expanded_array[:, LENGTH_COL]))

    # Get the unique levels and their corresponding max heights
    unique_levels, level_indices = np.unique(expanded_array[:, LEVEL_COL], return_inverse=True)
    layer_largest_dims = np.array(
        [np.max(expanded_array[level_indices == i, HEIGHT_COL]) for i in range(len(unique_levels))])

    # Calculate the shelf spaces (largest dimension - max height per layer)
    shelf_spaces = largest_dim - layer_largest_dims

    # Calculate the cumulative y-start decrement
    y_start_decrements = np.cumsum(shelf_spaces)
    y_start_decrements = np.insert(y_start_decrements[:-1], 0, 0)  # Shift the values by 1

    # Apply the decrements based on the level of each item
    compressed_array[:, Y_START_COL] -= y_start_decrements[level_indices]

    return compressed_array

def height_old(my_df):
    max_heights = my_df.groupby('level')['Height'].max()
    strip_height = max_heights.sum()
    return strip_height

def height_np_level(my_arr):
    levels = np.unique(my_arr[:, 2])  # Get unique levels
    max_heights = np.array([my_arr[my_arr[:, 2] == level, 1].max() for level in levels])
    strip_height = max_heights.sum()
    return strip_height

def height(arr):
    x_encum_array = arr.copy()
    layers = int(np.max(x_encum_array[:, 2]))  # Max of 'level' column

    for current_layer in range(1, layers + 1):
        rects_current_layer = x_encum_array[x_encum_array[:, 2] == current_layer]

        for current_rect in rects_current_layer:
            current_rect_index = np.where((x_encum_array == current_rect).all(axis=1))[0][0]

            # x-search space conditions
            lowerbound = current_rect[3]  # x-start
            upperbound = lowerbound + current_rect[0]  # x-start + Length

            condition1 = (x_encum_array[:, 3] >= lowerbound) & (x_encum_array[:, 3] < upperbound)
            condition2 = (x_encum_array[:, 3] + x_encum_array[:, 0] > lowerbound) & (
                        x_encum_array[:, 3] + x_encum_array[:, 0] <= upperbound)
            condition3 = (x_encum_array[:, 3] <= lowerbound) & (x_encum_array[:, 3] + x_encum_array[:, 0] >= upperbound)

            x_search = x_encum_array[condition1 | condition2 | condition3]

            heightbound = current_rect[4]  # y-start
            condition4 = x_search[:, 4] < heightbound
            y_search = x_search[condition4]

            if y_search.size > 0:
                y_end = y_search[:, 4] + y_search[:, 1]  # y-start + Height
                new_y_start = np.max(y_end)
            else:
                new_y_start = 0

            x_encum_array[current_rect_index, 4] = new_y_start

    # Calculate y-end (y-start + Height) for each row
    y_end = x_encum_array[:, 4] + x_encum_array[:, 1]

    # Find the maximum value of y-end
    strip_height = np.max(y_end)

    return strip_height


def monitor_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"Memory Usage: {memory.percent}% | CPU Usage: {cpu}%")

def exchange_operator(x_encum_starting_value, bin_width):

    expanded_np = x_encum_starting_value  # Instead of expanding, we copy

    ########
    success = 0
    while success != 1:
        # print(expanded_np)
        #rand_outer_idx, rand_inner_idx = np.random.choice(len(expanded_np), size=2, replace=False)

        rand_outer_idx = np.random.randint(len(expanded_np))
        #print(rand_outer_idx)
        rand_outer_rect = expanded_np[rand_outer_idx]
        #print(rand_outer_rect)
        rand_outer_rect_length = rand_outer_rect[0]  # Assuming Length is the first column
        rand_outer_rect_height = rand_outer_rect[1]  # Assuming Height is the second column
        outer_layer = rand_outer_rect[2]  # Assuming 'level' is the third column

        # Select outer rects from the same level
        outer_rects = expanded_np[expanded_np[:, 2] == outer_layer]

        ##############
        # Get the global indices of inner_rects in expanded_np
        inner_rects_mask = expanded_np[:, 2] != outer_layer
        inner_rects = expanded_np[expanded_np[:, 2] != outer_layer]  # This is the filtered subset of rectangles
        inner_rects_indices = np.where(inner_rects_mask)[0]  # These are the original indices in expanded_np
        #print(inner_rects)
        #3print(inner_rects_indices)
        # Sample a random index from inner_rects_indices, which holds global indices of inner_rects
        rand_inner_idx = np.random.choice(inner_rects_indices)
        #print(rand_inner_idx)
        # Use this global index to access the correct rectangle in expanded_np
        rand_inner_rect = expanded_np[rand_inner_idx]
        #print(rand_inner_rect)
        ##############

        rand_inner_rect_length = rand_inner_rect[0]
        rand_inner_rect_height = rand_inner_rect[1]

        outer_space = bin_width - np.sum(outer_rects[:, 0])  # Sum of 'Length' of outer rects
        outer_space_open = rand_outer_rect_length + outer_space

        inner_layer = rand_inner_rect[2]
        inner_layer_rects = expanded_np[expanded_np[:, 2] == inner_layer]
        inner_space = bin_width - np.sum(inner_layer_rects[:, 0])  # Sum of 'Length' of inner rects
        inner_space_open = rand_inner_rect_length + inner_space

        if (inner_space_open >= min(rand_outer_rect_length, rand_outer_rect_height) and
                outer_space_open >= min(rand_inner_rect_length, rand_inner_rect_height)):
            success = 1
            if inner_space_open >= max(rand_outer_rect_length, rand_outer_rect_height):
                #print('1st')
                x_start_inner = rand_inner_rect[3]  # Assuming 'x-start' is the 4th column

                # Define the criteria for filtering
                criteria = (expanded_np[:, 2] == inner_layer) & (expanded_np[:, 3] > x_start_inner)
                # Filter the rows based on criteria
                filtered_rows = expanded_np[criteria]
                # Subtract rand_inner_rect_length from the 3rd column of filtered rows
                filtered_rows[:, 3] -= rand_inner_rect_length
                # Update the original array with modified values
                expanded_np[criteria] = filtered_rows

                new_inner_rect = rand_outer_rect.copy()
                new_inner_rect[3] = bin_width - inner_space_open  # 'x-start'
                new_inner_rect[2] = inner_layer  # Change 'level'
                new_inner_rect[4] = rand_inner_rect[4]  # 'y-start'

                if np.random.randint(2) == 1:
                    #print('Problem1?')
                    new_inner_rect[0], new_inner_rect[1] = rand_outer_rect_height, rand_outer_rect_length
                expanded_np[rand_inner_idx] = new_inner_rect
            else:
                #print('2nd')
                x_start_inner = rand_inner_rect[3]

                # Define the criteria for filtering
                criteria = (expanded_np[:, 2] == inner_layer) & (expanded_np[:, 3] > x_start_inner)
                # Filter the rows based on criteria
                filtered_rows = expanded_np[criteria]
                # Subtract rand_inner_rect_length from the 3rd column of filtered rows
                filtered_rows[:, 3] -= rand_inner_rect_length
                # Update the original array with modified values
                expanded_np[criteria] = filtered_rows

                new_inner_rect = rand_outer_rect.copy()
                new_inner_rect[3] = bin_width - inner_space_open  # 'x-start'
                new_inner_rect[2] = inner_layer  # Change 'level'
                new_inner_rect[4] = rand_inner_rect[4]  # 'y-start'
                new_inner_rect[0], new_inner_rect[1] = min(rand_outer_rect_height, rand_outer_rect_length), \
                    max(rand_outer_rect_height, rand_outer_rect_length)
                expanded_np[rand_inner_idx] = new_inner_rect

            if outer_space_open >= max(rand_inner_rect_length, rand_inner_rect_height):
                #print('3rd')
                x_start_outer = rand_outer_rect[3]

                # Define the criteria for filtering
                criteria = (expanded_np[:, 2] == outer_layer) & (
                        expanded_np[:, 3] > x_start_outer)
                # Filter the rows based on criteria
                filtered_rows = expanded_np[criteria]
                # Subtract rand_inner_rect_length from the 3rd column of filtered rows
                filtered_rows[:, 3] -= rand_outer_rect_length
                # Update the original array with modified values
                expanded_np[criteria] = filtered_rows

                new_outer_rect = rand_inner_rect.copy()
                new_outer_rect[3] = bin_width - outer_space_open
                new_outer_rect[2] = outer_layer
                new_outer_rect[4] = rand_outer_rect[4]

                if np.random.randint(2) == 1:
                    #print('Problem3?')
                    new_outer_rect[0], new_outer_rect[1] = rand_inner_rect_height, rand_inner_rect_length
                else:
                    new_outer_rect[0], new_outer_rect[1] = rand_inner_rect_length, rand_inner_rect_height

                expanded_np[rand_outer_idx] = new_outer_rect
            else:
                #print('4th')
                x_start_outer = rand_outer_rect[3]

                # Define the criteria for filtering
                criteria = (expanded_np[:, 2] == outer_layer) & (
                        expanded_np[:, 3] > x_start_outer)
                # Filter the rows based on criteria
                filtered_rows = expanded_np[criteria]
                # Subtract rand_inner_rect_length from the 3rd column of filtered rows
                filtered_rows[:, 3] -= rand_outer_rect_length
                # Update the original array with modified values
                expanded_np[criteria] = filtered_rows

                new_outer_rect = rand_inner_rect.copy()
                new_outer_rect[3] = bin_width - outer_space_open
                new_outer_rect[2] = outer_layer
                new_outer_rect[4] = rand_outer_rect[4]
                new_outer_rect[0], new_outer_rect[1] = min(rand_inner_rect_height, rand_inner_rect_length), \
                    max(rand_inner_rect_height, rand_inner_rect_length)
                expanded_np[rand_outer_idx] = new_outer_rect

    x_neigh = expanded_np.copy()

    return x_neigh


def run_simulated_annealing_numpy(x_encum_starting_value):
    ##### Simulated Annealing Function Framework #####

    bin_width = 200
    # Fixed predetermined parameters
    e_max, a_max, r_max = 500, 5, 90
    T, beta, alpha = 20, 0.97, 1.2

    # Initialize control variables
    t, e, a, r = 1, 1, 0, 0

    # Plotting variables
    reheating, cooling, acc = 0, 0, 0
    #heights, temps = [], []

    # Convert to NumPy if not already done
    x_current = x_encum_starting_value.copy()
    x_encum = x_current.copy()

    # Function to get height of the configuration
    #temps.append(T)
    #heights.append(height(x_current))

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
            # Simulated annealing core logic

            expanded_np = x_current.copy()  # Instead of expanding, we copy

            ########
            x_neigh = exchange_operator(expanded_np.copy(), 1250)
            ########
            #x_neigh = expanded_np.copy()
            height_x_neigh = height(x_neigh.copy())
            height_x_current = height(x_current.copy())
            acc += 1
            if height_x_neigh < height_x_current:
                x_current = x_neigh
                a += 1
                t += 1
                #heights.append(height_x_current)
                #temps.append(T)
                if height_x_current < height(x_encum.copy()):
                    x_encum = x_current.copy()
                    #print('yes')
            elif np.exp(-(height_x_neigh - height_x_current) / T) >= random.random():
                x_current = x_neigh.copy()
                a += 1
                t += 1
                #heights.append(height_x_current)
                #temps.append(T)
            else:
                r += 1
                t += 1

    gc.collect()  # Clean up memory
    #monitor_resources()  # Monitor resource usage
    #strip_height = height(x_encum)

    #return x_encum, strip_height, heights,
    return height(x_encum)


#start = time.time()

#default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\T.xlsx"
#test = pd.read_excel(default_path, sheet_name="T1a")


#Numpy adaptations
#BFDH = shelf_expansion_v2(best_fit_packing(order_randomiser(test), 200))
#print(shelf_compression_v2(BFDH))
#print(height_old(BFDH))
#BFDH_numpy = BFDH.to_numpy()
#Meta = run_simulated_annealing_numpy(BFDH_numpy)
#print(shelf_compression_numpy(Meta))
#Meta_comp = shelf_compression_v2(Meta)
#print(Meta_comp)
#end = time.time()
#print(end-start)
#columns = ['Length', 'Height', 'level', 'x-start', 'y-start']
#Meta_comp = shelf_compression_numpy(Meta)
#Meta_df = pd.DataFrame(Meta_comp, columns=columns)
#print(Meta_df)
#print(height_old(Meta_df))
#plot_strip_single(shelf_compression_v2(BFDH), 200)
#plot_strip_single(Meta_df,200)




default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\PrONestDatasets.xlsx"
#test = pd.read_excel(default_path, sheet_name="T1a")
#sheet_names = ['T1a', 'T1b', 'T1c', 'T1d', 'T1e', 'T2a', 'T2b', 'T2c', 'T2d', 'T2e']
#sheet_names = ['T3a', 'T3b', 'T3c', 'T3d', 'T3e', 'T4a', 'T4b', 'T4c', 'T4d', 'T4e']
#sheet_names = ['N3a', 'N3b', 'N3c', 'N3d', 'N3e', 'N4a', 'N4b', 'N4c', 'N4d', 'N4e']
#sheet_names = ['C1a', 'C1b', 'C1c']
#sheet_names = ['BKW02', 'BKW03']
sheet_names = ['test2',	'test3', 'test5', 'test9', 'test10', 'test11']


# List of 10 different starting values for x_encum
#x_encum_starting_values = [
    # Add the first 4 initial values explicitly
#    shelf_expansion_v2(best_fit_packing(sort_height(test), 200)).to_numpy(),
#    shelf_expansion_v2(best_fit_packing(sort_width(test), 200)).to_numpy(),
#    shelf_expansion_v2(best_fit_packing(sort_maxdimension_height(test), 200)).to_numpy(),
#    shelf_expansion_v2(best_fit_packing(sort_maxdimension_width(test), 200)).to_numpy()
#] + [
    # Use list comprehension for the remaining 6 random values
#    shelf_expansion_v2(best_fit_packing(order_randomiser(test), 200)).to_numpy() for _ in range(40)
#]

def run_all_simulations_in_parallel(test):
    x_encum_starting_values = [
                                  # Add the first 4 initial values explicitly
                                  shelf_expansion_v2(best_fit_packing(sort_height(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_width(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_maxdimension_height(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_maxdimension_width(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_height(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_width(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_maxdimension_height(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(best_fit_packing(sort_maxdimension_width(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(next_fit_packing(sort_height(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(next_fit_packing(sort_width(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(next_fit_packing(sort_maxdimension_height(test), 1250)).to_numpy(),
                                  shelf_expansion_v2(next_fit_packing(sort_maxdimension_width(test), 1250)).to_numpy()
                              ] + [
                                  # Use list comprehension for the remaining 6 random values
                                  shelf_expansion_v2(best_fit_packing(order_randomiser(test), 1250)).to_numpy() for _ in
                                  range(20)
                              ]

    #def run_all_SA_in_parallel():
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_simulated_annealing_numpy, x_encum) for x_encum in x_encum_starting_values]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'A simulation generated an exception: {exc}')
    return results

start = time.time()

def process_sheet(sheet_name):
    df = pd.read_excel(default_path, sheet_name=sheet_name)
    results = run_all_simulations_in_parallel(df)
    return sheet_name, results

all_results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    future_to_sheet = {executor.submit(process_sheet, sheet): sheet for sheet in sheet_names}
    for future in concurrent.futures.as_completed(future_to_sheet):
        sheet_name, results = future.result()
        all_results[sheet_name] = results

#print(all_results)

# Assuming 'all_results' is the dictionary with sheet names as keys and results as values
results_data = []

# Create a list of tuples with the sheet name and the corresponding results
for sheet_name, results in all_results.items():
    for result in results:
        results_data.append((sheet_name, result))

# Convert the list to a DataFrame
df_results = pd.DataFrame(results_data, columns=['Sheet Name', 'Result'])

# Save the DataFrame to a single Excel sheet
output_file_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\MetaHeuristicResults\ProNestResults.xlsx"
df_results.to_excel(output_file_path, index=False, sheet_name='T23591011')

#start = time.time()
# Execute the parallel runs
#results = run_all_SA_in_parallel()
end = time.time()


print(end-start)
# Print the results
#for result in results:
#    print(result)



#start = time.time()
# Run the algorithm in parallel
#def run_all_simulations_in_parallel():
#    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#        futures = [executor.submit(run_simulated_annealing_numpy, x_encum) for x_encum in x_encum_starting_values]
#        results = []
#        all_strip_heights = []
#        min_strip_height = float('inf')
#        best_solution = None

#        for future in concurrent.futures.as_completed(futures):
#            try:
#                x_encum, strip_height, heights, temps = future.result()
#                #print(f'Strip height: {strip_height}')  # Print each strip height
#                results.append((x_encum, strip_height, heights, temps))
#                all_strip_heights.append(strip_height)

#                if strip_height < min_strip_height:
#                    min_strip_height = strip_height
#                    best_solution = (x_encum, strip_height, heights, temps)
#            except Exception as exc:
#                print(f'A simulation generated an exception: {exc}')

        #print(f'Lowest Strip Height: {min_strip_height}')
#        return best_solution, results, all_strip_heights


# Execute the parallel runs and retrieve the best solution
#best_solution, all_results, all_strip_heights = run_all_simulations_in_parallel()

# Extract the best solution details
#x_encum_best, strip_height_best, heights_best, temps_best = best_solution

#end = time.time()
#print(end-start)

##### Plot nest plot #####
#columns = ['Length', 'Height', 'level', 'x-start', 'y-start']
#x_encum_best_comp = shelf_compression_numpy(x_encum_best)
#x_encum_best_df = pd.DataFrame(x_encum_best_comp, columns=columns)
#plot_strip_single(x_encum_best_df,200)
# Print the results
#print(f'Best Strip height: {strip_height_best}')
#for strip_height in all_strip_heights:
#    print(strip_height)

#x_axis = range(len(temps_best))  # This will create a sequence like [0, 1, 2, 3, ..., len(temps_best)-1]

# Create a figure and axis
#fig, ax1 = plt.subplots()

# Plot temps_best on ax1 (primary y-axis)
#ax1.plot(x_axis, temps_best, color='r', label='Temps')
#ax1.set_xlabel('X-axis (Integer Sequence)')
#ax1.set_ylabel('Temperature', color='r')
#ax1.tick_params(axis='y', labelcolor='r')

# Create a second y-axis (heights_best)
#ax2 = ax1.twinx()  # This shares the same x-axis as ax1
#ax2.plot(x_axis, heights_best, color='b', label='Heights')
#ax2.set_ylabel('Height', color='b')
#ax2.tick_params(axis='y', labelcolor='b')

# Add titles and a grid
#plt.title('Temps and Heights vs X-axis')
#ax1.grid(True)

# Show the plot
#plt.show()


