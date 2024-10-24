import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import time
import concurrent.futures
import os
from GUIFunctions import plot_strip_single, shelf_expansion_v2, total_rect_area

def all_heuristics_parallel(dff_test, bin_width):
    # Define the heuristics in a dictionary with function references
    heuristics = {
        'NFDH': next_fit_packing(sort_height(dff_test), bin_width),
        'FFDH': first_fit_packing(sort_height(dff_test), bin_width),
        'BFDH': best_fit_packing(sort_height(dff_test), bin_width),
        'NFDW': next_fit_packing(sort_width(dff_test), bin_width),
        'FFDW': first_fit_packing(sort_width(dff_test), bin_width),
        'BFDW': best_fit_packing(sort_width(dff_test), bin_width),
        'NFDH_processed': next_fit_packing(sort_maxdimension_height(dff_test), bin_width),
        'FFDH_processed': first_fit_packing(sort_maxdimension_height(dff_test), bin_width),
        'BFDH_processed': best_fit_packing(sort_maxdimension_height(dff_test), bin_width),
        'NFDW_processed': next_fit_packing(sort_maxdimension_width(dff_test), bin_width),
        'FFDW_processed': first_fit_packing(sort_maxdimension_width(dff_test), bin_width),
        'BFDW_processed': best_fit_packing(sort_maxdimension_width(dff_test), bin_width)
    }

    # Function to run each heuristic and calculate height
    def run_heuristic(name):
        arrangement = heuristics[name]
        height_value = height_dataframe(arrangement)  # Evaluate the height
        return name, arrangement, height_value

    # Store heights and find the best arrangement
    heights = []
    best_name = None
    best_arrangement = None
    best_height = float('inf')

    # Using ThreadPoolExecutor to run the heuristics in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_heuristic, name): name for name in heuristics}

        results = {}
        for future in concurrent.futures.as_completed(futures):
            name, arrangement, height_value = future.result()  # Get the result of the heuristic
            results[name] = arrangement
            heights.append(height_value)

            # Check if this arrangement has the lowest height
            if height_value < best_height:
                best_height = height_value
                best_name = name
                best_arrangement = arrangement

    # Return the best arrangement, all heights, and the best height
    return best_arrangement, heights, best_height

def height_dataframe(my_df):
    max_heights = my_df.groupby('level')['Height'].max()
    strip_height = max_heights.sum()
    return strip_height

def height(my_arr):
    levels = np.unique(my_arr[:, 2])  # Get unique levels
    max_heights = np.array([my_arr[my_arr[:, 2] == level, 1].max() for level in levels])
    strip_height = max_heights.sum()
    return strip_height

def np_to_df(x_encum_best):
    columns = ['Length', 'Height', 'level', 'x-start', 'y-start']
    x_encum_best_comp = shelf_compression_numpy(x_encum_best)
    x_encum_best_df = pd.DataFrame(x_encum_best_comp, columns=columns)
    return x_encum_best_df

def np_to_df_plane(x_encum_best):
    columns = ['Length', 'Height', 'level', 'x-start', 'y-start']
    x_encum_best_comp = super_compression_numpy(x_encum_best)
    x_encum_best_df = pd.DataFrame(x_encum_best_comp, columns=columns)
    return x_encum_best_df

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

def first_fit_packing(dff_test, bin_width):
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
            return_to_layer = possible_indices[0]

            ##### update rectangle to previous layer position #####
            space_remaining = return_df.loc[return_df['level'] == return_to_layer, 'space remaining'].values
            x_pos_return = bin_width - space_remaining[0]
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

def super_compression_numpy(arr):
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

            x_encum_array[current_rect_index, 4] = new_y_start  # Update y-start

    return x_encum_array

def height_sc_np(arr):
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

def run_simulated_annealing_numpy(x_encum_starting_value, bin_width, T, alpha, beta, e_max, a_max, r_max):
    ##### Simulated Annealing Function Framework #####

    #bin_width = 200
    # Fixed predetermined parameters
    #e_max, a_max, r_max = 500, 8, 40
    #T, beta, alpha = 100, 0.99, 2

    # Initialize control variables
    t, e, a, r = 1, 1, 0, 0

    # Plotting variables
    reheating, cooling, acc = 0, 0, 0
    heights, temps = [], []

    # Convert to NumPy if not already done
    x_current = x_encum_starting_value.copy()
    x_encum = x_current.copy()

    # Function to get height of the configuration
    temps.append(T)
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
            # Simulated annealing core logic

            expanded_np = x_current.copy()  # Instead of expanding, we copy

            ########
            x_neigh = exchange_operator(expanded_np.copy(), bin_width)
            ########
            #x_neigh = expanded_np.copy()
            height_x_neigh = height(x_neigh.copy())
            height_x_current = height(x_current.copy())
            acc += 1
            if height_x_neigh < height_x_current:
                x_current = x_neigh
                a += 1
                t += 1
                heights.append(height_x_current)
                temps.append(T)
                if height_x_current < height(x_encum.copy()):
                    x_encum = x_current.copy()
                    #print('yes')
            elif np.exp(-(height_x_neigh - height_x_current) / T) >= random.random():
                x_current = x_neigh.copy()
                a += 1
                t += 1
                heights.append(height_x_current)
                temps.append(T)
            else:
                r += 1
                t += 1

    #gc.collect()  # Clean up memory
    #monitor_resources()  # Monitor resource usage
    strip_height = height(x_encum)

    return x_encum, strip_height, heights, temps

def run_simulated_annealing_numpy_plane(x_encum_starting_value, bin_width, T, alpha, beta, e_max, a_max, r_max):
    ##### Simulated Annealing Function Framework #####

    # Initialize control variables
    t, e, a, r = 1, 1, 0, 0

    # Plotting variables
    reheating, cooling, acc = 0, 0, 0
    heights, temps = [], []

    # Convert to NumPy if not already done
    x_current = x_encum_starting_value.copy()
    x_encum = x_current.copy()

    # Function to get height of the configuration
    temps.append(T)
    heights.append(height_sc_np(x_current))

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
            x_neigh = exchange_operator(expanded_np.copy(), bin_width)
            ########
            #x_neigh = expanded_np.copy()
            height_x_neigh = height_sc_np(x_neigh)
            height_x_current = height_sc_np(x_current)
            acc += 1
            if height_x_neigh < height_x_current:
                x_current = x_neigh
                a += 1
                t += 1
                heights.append(height_x_current)
                temps.append(T)
                if height_x_current < height_sc_np(x_encum):
                    x_encum = x_current
                    #print('yes')
            elif np.exp(-(height_x_neigh - height_x_current) / T) >= random.random():
                x_current = x_neigh
                a += 1
                t += 1
                heights.append(height_x_current)
                temps.append(T)
            else:
                r += 1
                t += 1

    #gc.collect()  # Clean up memory
    #monitor_resources()  # Monitor resource usage
    strip_height = height_sc_np(x_encum)

    return x_encum, strip_height, heights, temps


st.set_page_config(page_title='SPP Interface', layout='wide')
st.title("Strip Packing Optimiser")
st.subheader("DSS Interface")

upload_file = st.file_uploader('Upload your CSV File')

# Default DataFrame if no file is uploaded
default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\testdata2.xlsx"
default_df = pd.read_excel(default_path, sheet_name="testdata2")
#default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\T.xlsx"
#default_df = pd.read_excel(default_path, sheet_name="T1a")

#before file is uploaded, to avoid error
if upload_file:
    test = pd.read_excel(upload_file)
else:
    test = default_df

rect_area = total_rect_area(test)
def heuristicmainpage():
    st.header("Heuristic - Mainpage")
    st.write("#### Parameter Initialisation")
    # Custom Bin Width
    bin_width = st.number_input('Enter strip width', min_value=0, value=16)

    if st.button('Run Algorithm'):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write('#### Solution Nest Plot')
            start = time.time()
            best_arrangement, heights, best_height = all_heuristics_parallel(test, bin_width)
            end = time.time()
            plot_strip_single(best_arrangement, bin_width, best_height)
            st.success('All functions have finished running!')
            st.session_state['height_array_heuristics'] = heights
            st.session_state['output_table_heuristic'] = best_arrangement

        with col2:
            st.write('#### Performance Metrics')

            st.write(f"##### Strip Height: {best_height}")
            run_time = "{:.2f}".format(end - start)
            st.write(f"##### Execution Time: {run_time}s")
            plot_area = bin_width * best_height
            st.write(f"##### Plot Area: {plot_area}")
            plot_density = "{:.2f}".format(100 * rect_area / plot_area)
            st.write(f"##### Plot Density: {plot_density}%")
            num_heur = 12
            st.write(f"##### Competing Heuristics: {num_heur}")
def heuristicplots():
    if 'height_array_heuristics' in st.session_state:
        st.write("#### Strip Heights of Heuristics")
        heights_heuristics = st.session_state['height_array_heuristics']
        st.bar_chart(data=heights_heuristics)
    else:
        st.write("No data found in session state.")

def heuristictable():
    col1, col2 = st.columns([1,2])
    with col2:
        st.write("#### Packing Arrangement Data Table")
        if 'output_table_heuristic' in st.session_state:
            tabled_data = st.session_state['output_table_heuristic']
            st.dataframe(tabled_data)
        else:
            st.write("No data found in session state.")
    with col1:
        st.write("#### User Inputted Data")
        st.dataframe(test)
def metaheuristicmainpage():

    st.write("#### Parameter Initialisation")
    bin_width = st.number_input('Enter strip width', min_value=0, value=200)
    ##### User input of starting solutions #####
    x_encum_starting_values = []

    st.write("##### Starting Solutions Selection:")
    col001, col002, col003, col004, col005 = st.columns([1.55, 0.5, 2, 0.4, 2])
    with col001:
        st.write("###### No Heuristic")
    with col003:
        st.write("###### Standard Sorting Heuristics")
    with col005:
        st.write("###### Pre-Processed Heuristics")
    col01, col02, col03, col04, col05, col06, col07 = st.columns([3, 1, 2, 2, 0.8, 2, 2])
    with col01:
        rand_start = st.number_input("Random",
                                      min_value=0,
                                      max_value=50,
                                      value=30,
                                      step=1)
        for _ in range(rand_start):
            x_encum_starting_values.append(shelf_expansion_v2(best_fit_packing(order_randomiser(test), bin_width)).to_numpy())
    with col03:
        if st.checkbox("NFDH"):
            x_encum_starting_values.append(shelf_expansion_v2(next_fit_packing(sort_height(test), bin_width)).to_numpy())
        if st.checkbox("FFDH"):
            x_encum_starting_values.append(
                shelf_expansion_v2(first_fit_packing(sort_height(test), bin_width)).to_numpy())
        if st.checkbox("BFDH"):
            x_encum_starting_values.append(
                shelf_expansion_v2(best_fit_packing(sort_height(test), bin_width)).to_numpy())
    with col04:
        if st.checkbox("NFDW"):
            x_encum_starting_values.append(shelf_expansion_v2(next_fit_packing(sort_width(test), bin_width)).to_numpy())
        if st.checkbox("FFDW"):
            x_encum_starting_values.append(shelf_expansion_v2(first_fit_packing(sort_width(test), bin_width)).to_numpy())
        if st.checkbox("BFDW"):
            x_encum_starting_values.append(shelf_expansion_v2(best_fit_packing(sort_width(test), bin_width)).to_numpy())
    with col06:
        if st.checkbox("NFDH*"):
            x_encum_starting_values.append(shelf_expansion_v2(next_fit_packing(sort_maxdimension_height(test), bin_width)).to_numpy())
        if st.checkbox("FFDH*"):
            x_encum_starting_values.append(
                shelf_expansion_v2(first_fit_packing(sort_maxdimension_height(test), bin_width)).to_numpy())
        if st.checkbox("BFDH*"):
            x_encum_starting_values.append(
                shelf_expansion_v2(best_fit_packing(sort_maxdimension_height(test), bin_width)).to_numpy())
    with col07:
        if st.checkbox("NFDW*"):
            x_encum_starting_values.append(shelf_expansion_v2(next_fit_packing(sort_maxdimension_width(test), bin_width)).to_numpy())
        if st.checkbox("FFDW*"):
            x_encum_starting_values.append(
                shelf_expansion_v2(first_fit_packing(sort_maxdimension_width(test), bin_width)).to_numpy())
        if st.checkbox("BFDW*"):
            x_encum_starting_values.append(
                shelf_expansion_v2(best_fit_packing(sort_maxdimension_width(test), bin_width)).to_numpy())
   ##### User input of parameters (temps, epochs, etc.) ####
    st.write("##### Simulated Annealing Parameters:")
    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 2, 1, 2, 1, 2])
    with col1:
        T = st.number_input('Temperature', min_value=0, value=50)

    with col3:
        e_max = st.number_input('Maximum Epochs', min_value=0, value=500)
        #beta = st.number_input('Cooling Schedule (β)', min_value=0.01, value=0.95, step=0.01)

    with col5:
        a_max = st.number_input('Maximum Acceptances', min_value=0, value=5)
        #alpha = st.number_input('Reheating Schedule (α)', min_value=1.0, value=1.5, step=0.1)
        #if packing_restriction == "Plane Packing":
    with col7:
        r_max = st.number_input('Maximum Rejections', min_value=0, value=50)

    beta = st.slider("Cooling Schedule (β)", min_value=0.70, max_value=0.99, value=0.97, step=0.01)
    alpha = st.slider("Reheating Schedule (α)", min_value=1.0, max_value=3.5, value=1.2, step=0.1)

    packing_restriction = st.radio(
        "Select Packing Restrictions",
        key="visibility",
        options=["Level (Guillotine)", "Plane (Non-Guillotine)"],
    )

    if st.button('Run Algorithm'):

        with st.spinner('Generating Solution...'):
            start_time = time.time()
#            x_encum_starting_values = [
#                # Add 10 different initial values of x_encum
#                shelf_expansion_v2(best_fit_packing(order_randomiser(test), bin_width)).to_numpy() for _ in range(10)
#            ]

            if packing_restriction == "Plane (Non-Guillotine)":
                ####### Numpy #######
                def run_all_simulations_in_parallel_plane():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        futures = [
                            executor.submit(run_simulated_annealing_numpy_plane, x_encum, bin_width, T, alpha, beta, e_max, a_max,
                                            r_max) for x_encum in x_encum_starting_values]
                        results = []
                        all_strip_heights = []
                        min_strip_height = float('inf')
                        best_solution = None

                        for future in concurrent.futures.as_completed(futures):
                            try:
                                x_encum, strip_height, heights, temps = future.result()
                                # print(f'Strip height: {strip_height}')  # Print each strip height
                                results.append((x_encum, strip_height, heights, temps))
                                all_strip_heights.append(strip_height)

                                if strip_height < min_strip_height:
                                    min_strip_height = strip_height
                                    best_solution = (x_encum, strip_height, heights, temps)
                            except Exception as exc:
                                print(f'A simulation generated an exception: {exc}')
                    return best_solution, results, all_strip_heights

                best_solution, all_results, all_strip_heights = run_all_simulations_in_parallel_plane()

                # Extract the best solution details
                x_encum_best, strip_height_best, heights_best, temps_best = best_solution
                end_time = time.time()

                # Convert to dataframe:
                x_encum_best_df = np_to_df_plane(x_encum_best)

                run_time = "{:.2f}".format(end_time - start_time)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write('#### Solution Nest Plot')
                    # Return the dataframe with the minimum height
                    plot_strip_single(x_encum_best_df, bin_width, strip_height_best)
                with col2:
                    st.write('#### Performance Metrics')
                    st.write(f"##### Strip Height: {strip_height_best}")
                    st.write(f"##### Execution Time: {run_time}s")
                    plot_area = bin_width*strip_height_best
                    st.write(f"##### Plot Area: {plot_area}")
                    plot_density = "{:.2f}".format(100*rect_area/plot_area)
                    st.write(f"##### Plot Density: {plot_density}%")

            else:
                ####### Numpy #######
                def run_all_simulations_in_parallel():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        futures = [
                            executor.submit(run_simulated_annealing_numpy, x_encum, bin_width, T, alpha, beta, e_max, a_max,
                                            r_max) for x_encum in x_encum_starting_values]
                        results = []
                        all_strip_heights = []
                        min_strip_height = float('inf')
                        best_solution = None

                        for future in concurrent.futures.as_completed(futures):
                            try:
                                x_encum, strip_height, heights, temps = future.result()
                                # print(f'Strip height: {strip_height}')  # Print each strip height
                                results.append((x_encum, strip_height, heights, temps))
                                all_strip_heights.append(strip_height)

                                if strip_height < min_strip_height:
                                    min_strip_height = strip_height
                                    best_solution = (x_encum, strip_height, heights, temps)
                            except Exception as exc:
                                print(f'A simulation generated an exception: {exc}')
                    return best_solution, results, all_strip_heights

                best_solution, all_results, all_strip_heights = run_all_simulations_in_parallel()

                # Extract the best solution details
                x_encum_best, strip_height_best, heights_best, temps_best = best_solution
                end_time = time.time()

                # Convert to dataframe:
                x_encum_best_df = np_to_df(x_encum_best)

                run_time = "{:.2f}".format(end_time - start_time)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write('#### Solution Nest Plot')
                    # Return the dataframe with the minimum height
                    plot_strip_single(x_encum_best_df, bin_width, strip_height_best)
                with col2:
                    st.write('#### Performance Metrics')
                    st.write(f"##### Strip Height: {strip_height_best}")
                    st.write(f"##### Execution Time: {run_time}s")
                    plot_area = bin_width*strip_height_best
                    st.write(f"##### Plot Area: {plot_area}")
                    plot_density = "{:.2f}".format(100*rect_area/plot_area)
                    st.write(f"##### Plot Density: {plot_density}%")

        st.success("Solution Found!")

        # Finding Temperature and Height arrays for min height solution
        height_array = heights_best
        temp_array = temps_best
        output_table = x_encum_best_df
        # Store the selected arrays in session state
        st.session_state['output_table'] = output_table
        st.session_state['height_array'] = height_array
        st.session_state['temp_array'] = temp_array
        st.session_state['height_solutions'] = all_strip_heights
def metaheuristicplots():
    # Access stored arrays from session state
    if 'height_array' in st.session_state and 'temp_array' in st.session_state and 'height_solutions' in st.session_state:
        height_solutions = st.session_state['height_solutions']
        heights = st.session_state['height_array']
        temps = st.session_state['temp_array']

        st.bar_chart(data=height_solutions)

        # plot height, temp, and iteration
        iteration = np.arange(0, len(heights))
        # Create a figure and axis
        fig, ax1 = plt.subplots()

        # Plot the first y-axis (heights_array)
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Height', color=color)
        ax1.plot(iteration, heights, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Temperature', color=color)
        ax2.plot(iteration, temps, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Show the plot
        plt.title('Height and Temperature vs Iteration')
        st.pyplot(fig)
    else:
        st.write("No data found in session state.")

def metaheuristictable():
    col1, col2 = st.columns([1, 2])
    with col2:
        st.write("#### Packing Arrangement Data Table")
        if 'output_table' in st.session_state:
            tabled_data = st.session_state['output_table']
            st.dataframe(tabled_data)
        else:
            st.write("No data found in session state.")
    with col1:
        st.write("#### User Inputted Data")
        st.dataframe(test)


##### New Insert #####
st.sidebar.header('Options')

# Set default index for the first selectbox (0 for "Heuristic", 1 for "Metaheuristic")
option1 = st.sidebar.selectbox(
    'Choose an Algorithm Class:',
    ('Heuristic', 'Metaheuristic'),
    index=0  # Default to "Heuristic"
)

# Conditional logic for the second selectbox
if option1 == 'Metaheuristic':
    option2 = st.sidebar.selectbox(
        'Choose an Option:',
        ('Simulated Annealing', 'Additional Plots', 'Solution Table'),
        index=0  # Default to "Mainpage" for Metaheuristic
    )
else:  # Heuristic is selected
    option2 = st.sidebar.selectbox(
        'Choose a sub-option:',
        ('Mainpage', 'Additional Plots','Solution Table'),
        index=0  # Default to "Mainpage" for Heuristic
    )

# Display the selected options
#st.write(f'Selected option: {option1}')
#st.write(f'Selected sub-option: {option2}')

# Display different content based on selections
if option1 == 'Heuristic':
    if option2 == 'Mainpage':
        heuristicmainpage()

    elif option2 == 'Solution Table':
        st.header("Heuristic - Solution Table")
        #st.write("Content for Heuristic Solution Table")
        heuristictable()

    elif option2 == 'Additional Plots':
        st.header("Heuristic - Additional Plots")
        heuristicplots()

else:  # Metaheuristic
    if option2 == 'Simulated Annealing':
        st.header("Metaheuristic - Simulated Annealing")
        metaheuristicmainpage()
    elif option2 == 'Additional Plots':
        st.header("Metaheuristic - Additional Plots")
        #st.write("Content for Metaheuristic Additional Plots")
        metaheuristicplots()
        #if 'output' in st.session_state:
        #    st.write("Recalled output from Page One: ", st.session_state['output'])
        #else:
        #    st.write("No data generated yet. Please go to Page One and generate data.")

    elif option2 == 'Solution Table':
        st.header("Metaheuristic - Solution Table")
        #st.write("Content for Metaheuristic Solution Table")
        metaheuristictable()
# Create columns to control layout







