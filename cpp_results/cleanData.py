import re
import csv
import sys
import argparse

def print_tests_parameters(filename):
    """
    Find all the different test runs in the file and print their parameters.
    """
    data_pattern = r'TestFeynmanKac> Test FeynmanKac,'
    test_counter = 0

    print(f"Extracting test parameters from {filename}...")
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(data_pattern, line.strip())
            if match:
                # Convert matched groups to appropriate types
                print(line.strip())

                test_counter += 1

    print(f"Found {test_counter} test runs in {filename}")
    return

def extract_conv_data(filename, output_csv):
    """
    Extract WoS convergence test data from the output file and convert to CSV.

    Args:
        filename: Input file path
        output_csv: Output CSV file path (optional, defaults to input_name.csv)
    """

    if output_csv is None:
        output_csv = filename.rsplit('.', 1)[0] + '.csv'

    # Pattern to match data lines
    data_pattern = r'TestFeynmanKac>\s+([0-9.e-]+),\s+([0-9.-]+),\s+([0-9.-]+),\s+([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+)'

    # CSV headers
    headers = ['delta', 'Dimension', 'WoS-result', 'expected', 'WoS-error',
               'work']

    data_rows = []

    with open(filename, 'r') as file:
        for line in file:
            match = re.match(data_pattern, line.strip())
            if match:
                # Convert matched groups to appropriate types
                row = [
                    float(match.group(1)),  # delta
                    float(match.group(2)),  # Dimension
                    float(match.group(3)),  # WoS-result
                    float(match.group(4)),  # expected
                    float(match.group(5)),  # WoS-error
                    float(match.group(6)),  # work
                ]
                data_rows.append(row)

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"Extracted {len(data_rows)} data rows to {output_csv}")
    return data_rows


def extract_cg_data(filename, output_csv):
    """
    Extract cg timing test data from the output file and convert to CSV.

    Args:
        filename: Input file path
        output_csv: Output CSV file path (optional, defaults to input_name.csv)
    """

    if output_csv is None:
        output_csv = filename.rsplit('.', 1)[0] + '.csv'

    # Pattern to match data lines
    data_pattern = r'TestFeynmanKac>\s+([0-9.e-]+),\s+([0-9.-]+),\s+([0-9.-]+),\s+([0-9.-]+),\s*([0-9.-]+)'

    # CSV headers
    headers = ['epsilon', 'dimension', 'mlmc_result', 'mc_result', 'expected', 
               'mlmc_cost', 'mc_cost', 'max_level', 'varL']

    data_rows = []

    with open(filename, 'r') as file:
        for line in file:
            match = re.match(data_pattern, line.strip())
            if match:
                # Convert matched groups to appropriate types
                row = [
                    float(match.group(1)),  # mlmc error
                    float(match.group(2)),  # mlmc time
                    float(match.group(3)),  # cg error point
                    float(match.group(4)),  # cg error l2
                    float(match.group(5)),  # cg time
                ]
                data_rows.append(row)

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"Extracted {len(data_rows)} data rows to {output_csv}")
    return data_rows


def extract_mlmc_data(filename, output_csv=None):
    """
    Extract MLMC speedup test data from the output file and convert to CSV.

    Args:
        filename: Input file path
        output_csv: Output CSV file path (optional, defaults to input_name.csv)
    """

    if output_csv is None:
        output_csv = filename.rsplit('.', 1)[0] + '.csv'

    # Pattern to match data lines
    data_pattern = r'TestFeynmanKac>\s+([0-9.e-]+),\s+(\d+),\s+([0-9.-]+),\s+([0-9.-]+),\s+([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+),\s*(\d+),\s*([0-9.-]+),'

    # CSV headers
    headers = ['epsilon', 'dimension', 'mlmc_result', 'mc_result', 'expected', 
               'mlmc_cost', 'mc_cost', 'max_level', 'varL']

    data_rows = []

    with open(filename, 'r') as file:
        for line in file:
            match = re.match(data_pattern, line.strip())
            if match:
                # Convert matched groups to appropriate types
                row = [
                    float(match.group(1)),  # epsilon
                    int(match.group(2)),    # dimension
                    float(match.group(3)),  # mlmc_result
                    float(match.group(4)),  # mc_result
                    float(match.group(5)),  # expected
                    float(match.group(6)),  # mlmc_cost
                    float(match.group(7)),  # mc_cost
                    int(match.group(8)),    # max_level
                    float(match.group(9))   # varL
                ]
                data_rows.append(row)

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"Extracted {len(data_rows)} data rows to {output_csv}")
    return data_rows


def extract_timing_data(filename, output_csv=None):
    """
    Extract timing data from the output file.

    Args:
        filename: Input file path
        output_csv: Output CSV file path (optional)
    """

    if output_csv is None:
        output_csv = filename.rsplit('.', 1)[0] + '_timings.csv'

    timing_pattern = r'Timings>\s+(\w+)\.*\s+Wall\s+(\w+)\s+=\s+([0-9.]+)'

    timing_data = []

    with open(filename, 'r') as file:
        for line in file:
            match = re.match(timing_pattern, line.strip())
            if match:
                timer_name = match.group(1)
                metric = match.group(2)
                value = float(match.group(3))
                timing_data.append([timer_name, metric, value])

    if timing_data:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timer_name', 'metric', 'value'])
            writer.writerows(timing_data)

        print(f"Extracted {len(timing_data)} timing entries to {output_csv}")

    return timing_data


if __name__ == "__main__":

    options = ["cg", "mlmc", "conv", "test_info"]
    parser = argparse.ArgumentParser(description="Extract CG and MLMC data from output files.")
    parser.add_argument("filetype", choices=options, help="Type of data to extract: 'cg', 'mlmc', or 'conv' or whether to print the run parameters 'test_info'.")
    parser.add_argument("input_file", help="Input file path containing the data to extract.")
    parser.add_argument("-o", "--output_csv", help="Output CSV file path (optional).")

    parser.parse_args()
    data_type = parser.parse_args().filetype
    input_file = parser.parse_args().input_file
    if len(sys.argv) < 2:
        print("Usage: python extract_mlmc_data.py <input_file> [output_csv]")
        sys.exit(1)

    output_file = parser.parse_args().output_csv
    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '.csv'
    if data_type not in options:
        print(f"Invalid data type '{data_type}'. Choose from {options}.")
        sys.exit(1)
    # Extract timing data
    if data_type != "test_info":
        print(f"Extracting {data_type} data from {input_file}...")
        timing_data = extract_timing_data(input_file)

    if data_type == "cg":
        # Extract CG data
        cg_data = extract_cg_data(input_file, output_file)
        print(f"Processing complete. Found {len(cg_data)} CG data points and {len(timing_data)} timing measurements.")
    elif data_type == "mlmc":
        # Extract MLMC data
        mlmc_data = extract_mlmc_data(input_file, output_file)
        print(f"Processing complete. Found {len(mlmc_data)} MLMC data points and {len(timing_data)} timing measurements.")
    elif data_type == "conv":
        # Extract MLMC data
        conv_data = extract_conv_data(input_file, output_file)
        print(f"Processing complete. Found {len(conv_data)} convergence data points and {len(timing_data)} timing measurements.")
    elif data_type == "test_info":
        # Print test parameters
        print_tests_parameters(input_file)
        print("Processing complete. No data extracted.")
    # Extract main MLMC data


