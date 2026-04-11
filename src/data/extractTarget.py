import os
def extractTarget(summary_file_path, edf_file_path):
    edf_file_name = os.path.basename(edf_file_path)
    seizure_start_time = None
    seizure_end_time = None
    with open(summary_file_path, 'r') as file:
        lines = file.readlines()
    found = False
    for line in lines:
        if "File Name: " + edf_file_name in line:
            found = True
        if found:
            if "Number of Seizures in File: 0" in line:
                return None, None  # No seizures found
            if "Seizure Start Time:" in line:
                seizure_start_time = int(line.split(": ")[1].split(" ")[0])
            if "Seizure End Time:" in line:
                seizure_end_time = int(line.split(": ")[1].split(" ")[0])
                break  # Found the needed info, exit loop
    return seizure_start_time, seizure_end_time

