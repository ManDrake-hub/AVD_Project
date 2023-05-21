import json
import pandas as pd
import csv
import numpy as np

INFRACTIONS = True
SCORE = True
SHUTDOWN_EVENT = True

merged_dict = {}
custom_dict = {}

infractions_dict_new = {"collisions_layout":"Coll with Layout", "collisions_pedestrian":"Coll with Pedestrians", "collisions_vehicle":"Coll with Vehicles", 
                        "red_light":"Red light", "stop_infraction":"Stop", "outside_route_lanes":"Outside lanes", "min_speed_infractions":"Useless Decelerations", 
                        "yield_emergency_vehicles_infractions":"Yield emergency vehicles", "scenario_timeouts":"Scenario timeouts"}
score_dict_new = {"score_composed":"Driving score", "score_route":"Route Completion", "score_penalty":"Infraction Penalty", "duration_game":"Simulation time (s)"}
shutdown_event_dict_new = {"route_dev":"Route Deviation", "vehicle_blocked":"Vehicle Blocked", "route_timeout":"Route Timeout"}

file_json = 'results/simulation_results_must.json'
file_csv = 'results/csv/simulation_results_must.csv'
file_xlsx = 'results/xlsx/simulation_results_must.xlsx'

# Load the JSON file
with open(file_json, 'r') as f:
    data = json.load(f)

# Title
global_status = data['_checkpoint']['global_record']['status']
title = [' ','Global_record']
routes = data['_checkpoint']['records']
for i in range(len(routes)):
    title.append(routes[i]['route_id'])
# Status
status = ['Status']
global_status = data['_checkpoint']['global_record']['status']
status.append(global_status)
for i in range(len(routes)):
    status.append(routes[i]['status'])
# Num infractions
num_infractions = ['N Infractions','']
for i in range(len(routes)):
    num_infractions.append(routes[i]['num_infractions'])

# Dict infractions global
infractions_dict_comp = data['_checkpoint']['global_record']['infractions']
infractions_dict = dict((k, infractions_dict_comp[k]) for k in list(infractions_dict_comp.keys())[:-3])
values = [float(val) for val in list(infractions_dict.values())]
# Percentages infractions
total = sum(values)
percentages = [val/total*100 for val in values]
for key in infractions_dict:
    infractions_dict[key] = f"{round(percentages.pop(0),1):.1f}%"

# Dict score global
score_dict = data['_checkpoint']['global_record']['scores_mean']

# Dict shutdown_event global
shutdown_event_dict = dict((k, infractions_dict_comp[k]) for k in list(infractions_dict_comp.keys())[-3:])

if INFRACTIONS:
    merged_dict.update(infractions_dict)
    custom_dict.update(infractions_dict_new)
if SCORE:
    merged_dict.update(score_dict)
    merged_dict.update({"duration_game":''})
    custom_dict.update(score_dict_new)
if SHUTDOWN_EVENT:
    merged_dict.update(shutdown_event_dict)
    custom_dict.update(shutdown_event_dict_new)

# Matrix for global
keys = np.array(list(merged_dict.keys()))
values = np.array(list(merged_dict.values()))
total_matrix = np.column_stack((keys, values))

# Update matrix to add routes
new_cols = np.zeros((total_matrix.shape[0], len(routes)))
total_matrix = np.hstack((total_matrix, new_cols))
for i in range(len(routes)):
    for j in range(len(total_matrix)):
        if total_matrix[j,0] in shutdown_event_dict: 
            shutdown_event_value = data['_checkpoint']['records'][i]['infractions'][total_matrix[j,0]]
            if len(shutdown_event_value) == 0:
                total_matrix[j][i+2] = ''
            else:
                total_matrix[j][i+2] = shutdown_event_value[0]
        elif total_matrix[j,0] in score_dict and total_matrix[j,0] != "duration_game":
            score_value = data['_checkpoint']['records'][i]['scores'][total_matrix[j,0]]
            total_matrix[j][i+2] = round(score_value,1)
        elif total_matrix[j,0] == "duration_game":
            duration_game = data['_checkpoint']['records'][i]['meta'][total_matrix[j,0]]
            total_matrix[j][i+2] = round(duration_game,1)
        else:
            infraction_value = len(data['_checkpoint']['records'][i]['infractions'][total_matrix[j,0]])
            if infraction_value == 0:
                total_matrix[j][i+2] = ''
            else:
                total_matrix[j][i+2] = round(infraction_value,1)
for row in total_matrix:
    if row[0] in shutdown_event_dict_new or score_dict_new or infractions_dict_new:
        row[0] = custom_dict[row[0]]
print(total_matrix)

# CSV
with open(file_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    writer.writerow(status)
    writer.writerow(num_infractions)
    for row in total_matrix:
        writer.writerow(row)

# CSV to xlsx
read_file = pd.read_csv (file_csv)
read_file.to_excel (file_xlsx, index = None, header=True)
