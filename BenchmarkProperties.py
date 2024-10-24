import pandas as pd
import concurrent.futures
import math

default_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\Kendall&J.xlsx"
output_path = r"C:\Users\hansenm\Dropbox\4th Year - Semester 1\Industrial Project\Layer Heurisitic First Attempt\Benchmark Properties\Kendall&J_Properties.xlsx"
#sheet_names = ['N1a', 'N1b', 'N1c', 'N1d', 'N1e', 'N2a', 'N2b', 'N2c', 'N2d', 'N2e',
#               'N3a', 'N3b', 'N3c', 'N3d', 'N3e', 'N4a', 'N4b', 'N4c', 'N4d', 'N4e']
#sheet_names = ['C1a', 'C1b', 'C1c', 'C2a', 'C2b', 'C2c',
#               'C3a', 'C3b', 'C3c', 'C4a', 'C4b', 'C4c']
#sheet_names = ['BKW01', 'BKW02', 'BKW03', 'BKW04', 'BKW05']
#sheet_names = ['test2',	'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9', 'test10', 'test11']
sheet_names = ['Kendall', 'J1', 'J2']
def analyze_sheet(df):
    max_aspect_ratio = 0
    max_area = 0
    min_area = float('inf')
    max_dimension = 0
    min_dimension = float('inf')

    for _, row in df.iterrows():
        length, height = row['Length'], row['Height']
        aspect_ratio = max(length / height, height / length)
        area = length * height
        max_dimension = max(max_dimension, length, height)
        min_dimension = min(min_dimension, length, height)

        max_aspect_ratio = max(max_aspect_ratio, aspect_ratio)
        max_area = max(max_area, area)
        min_area = min(min_area, area)

    extreme_aspect_ratio = max_aspect_ratio
    extreme_area_ratio = max_area / min_area if min_area > 0 else float('inf')
    range_of_dimensions = max_dimension / min_dimension if min_dimension > 0 else float('inf')

    return extreme_aspect_ratio, extreme_area_ratio, range_of_dimensions

def process_sheet(sheet_name):
    df = pd.read_excel(default_path, sheet_name=sheet_name)
    results = analyze_sheet(df)
    return sheet_name, results

def main():
    all_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_sheet = {executor.submit(process_sheet, sheet): sheet for sheet in sheet_names}
        for future in concurrent.futures.as_completed(future_to_sheet):
            sheet_name, results = future.result()
            all_results[sheet_name] = results

    # Create a DataFrame from the results and save to Excel
    results_df = pd.DataFrame.from_dict(all_results, orient='index',
                                        columns=['Extreme Aspect Ratio', 'Extreme Area Ratio', 'Range of Dimensions'])
    results_df.index.name = 'Sheet Name'
    results_df.to_excel(output_path)
    print(f"Analysis complete. Results saved to '{output_path}'.")

if __name__ == '__main__':
    main()
