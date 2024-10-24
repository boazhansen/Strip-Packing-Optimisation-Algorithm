#import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import concurrent.futures

default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\T.xlsx"
#default_df = pd.read_excel(default_path, sheet_name="T1a")
def height(my_df):
    #outer_layer = my_df['level'].max()
    #max_y_start = my_df['y-start'].max()
    #outer_layer_height = my_df.loc[my_df['level'] == outer_layer, 'Height'].max()
    #strip_height = max_y_start + outer_layer_height
    #print(strip_height)
    max_heights = my_df.groupby('level')['Height'].max()
    strip_height = max_heights.sum()
    return strip_height


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


#BFDH = best_fit_packing(sort_maxdimension_height(default_df),16)
#plot_strip_single(BFDH,16)

#sheet_names = ['T1a', 'T1b', 'T1c', 'T1d', 'T1e', 'T2a', 'T2b', 'T2c', 'T2d', 'T2e',
#               'T3a', 'T3b', 'T3c', 'T3d', 'T3e', 'T4a', 'T4b', 'T4c', 'T4d', 'T4e']

sheet_names = ['BKW05']

def all_heuristics_parallel(dff_test, bin_width):
    # Define the heuristics in a dictionary with function references
    heuristics = {
        'NFDH': height(next_fit_packing(sort_height(dff_test), bin_width)),
        'FFDH': height(first_fit_packing(sort_height(dff_test), bin_width)),
        'BFDH': height(best_fit_packing(sort_height(dff_test), bin_width)),
        'NFDW': height(next_fit_packing(sort_width(dff_test), bin_width)),
        'FFDW': height(first_fit_packing(sort_width(dff_test), bin_width)),
        'BFDW': height(best_fit_packing(sort_width(dff_test), bin_width)),
        'NFDH_processed': height(next_fit_packing(sort_maxdimension_height(dff_test), bin_width)),
        'FFDH_processed': height(first_fit_packing(sort_maxdimension_height(dff_test), bin_width)),
        'BFDH_processed': height(best_fit_packing(sort_maxdimension_height(dff_test), bin_width)),
        'NFDW_processed': height(next_fit_packing(sort_maxdimension_width(dff_test), bin_width)),
        'FFDW_processed': height(first_fit_packing(sort_maxdimension_width(dff_test), bin_width)),
        'BFDW_processed': height(best_fit_packing(sort_maxdimension_width(dff_test), bin_width))
    }

    # Function to run each heuristic
    def run_heuristic(name):
        return (name, heuristics[name])

    # Using ThreadPoolExecutor to run the heuristics in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_heuristic, name): name for name in heuristics}

        results = {}
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()  # Get the result of the heuristic
            results[name] = result

    return results

    #min_heuristic = min(results, key=results.get)
    #min_height = results[min_heuristic]

    #return min_heuristic, min_height, results

# Usage example:
# dff_test and bin_width should be defined earlier
'''
min_heuristic, min_height, result = all_heuristics_parallel(default_df, 200)
print(result)
print(min_heuristic)
print(min_height)
'''
#FFDH = result['FFDH']
#plot_strip_single(best_fit_packing(sort_maxdimension_width(default_df),200),200)

def process_sheet(sheet_name):
    df = pd.read_excel(default_path, sheet_name=sheet_name)
    results = all_heuristics_parallel(df, 100)
    return sheet_name, results

# Process all sheets
all_results = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_sheet = {executor.submit(process_sheet, sheet): sheet for sheet in sheet_names}
    for future in concurrent.futures.as_completed(future_to_sheet):
        sheet_name, results = future.result()
        all_results[sheet_name] = results

# Create a list to store all results
results_list = []

# Flatten the results into a list of dictionaries
for sheet_name, heuristics in all_results.items():
    for heuristic_name, height_value in heuristics.items():
        results_list.append({
            'Sheet': sheet_name,
            'Heuristic': heuristic_name,
            'Height': height_value
        })

# Convert the list to a DataFrame
results_df = pd.DataFrame(results_list)

# Sort the DataFrame by Sheet and Heuristic
results_df = results_df.sort_values(['Sheet', 'Heuristic'])

# Save the results to a new Excel file
output_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\HeuristicResults\BKW5_Heuristics_Results.xlsx"
results_df.to_excel(output_path, index=False)

print(f"Results have been saved to {output_path}")