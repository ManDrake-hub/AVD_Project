import json
import pandas as pd
import csv
import numpy as np

INFRACTIONS = True
SCORE = True
SHUTDOWN_EVENT = True

# Load the JSON file
with open('results/routes_avddiem_exam_results.json', 'r') as f:
    data = json.load(f)

# Title
global_status = data['_checkpoint']['global_record']['status']
title = [' ','Global_record']
routes = data['_checkpoint']['records']
for i in range(len(routes)):
    title.append(routes[i]['route_id'])
#Status
status = ['status']
global_status = data['_checkpoint']['global_record']['status']
status.append(global_status)
for i in range(len(routes)):
    status.append(routes[i]['status'])
#Num infractions
num_infractions = ['num_infractions','']
for i in range(len(routes)):
    num_infractions.append(routes[i]['num_infractions'])

#Dict infractions global
infractions_dict_comp = data['_checkpoint']['global_record']['infractions']
infractions_dict = dict((k, infractions_dict_comp[k]) for k in list(infractions_dict_comp.keys())[:-3])
values = [float(val) for val in list(infractions_dict.values())]
# Percentages infractions
total = sum(values)
percentages = [val/total*100 for val in values]
for key in infractions_dict:
    infractions_dict[key] = f"{percentages.pop(0):.2f}%"
#Dict score global
score_dict = data['_checkpoint']['global_record']['scores_mean']
#Dict shutdown_event global
shutdown_event_dict = dict((k, infractions_dict_comp[k]) for k in list(infractions_dict_comp.keys())[-3:])

# Matrix for global
merged_dict = {}
merged_dict.update(infractions_dict)
merged_dict.update(score_dict)
merged_dict.update(shutdown_event_dict)
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
                total_matrix[j][i+2] = None
            else:
                total_matrix[j][i+2] = shutdown_event_value[0]
        elif total_matrix[j,0] in score_dict:
            score_value = data['_checkpoint']['records'][i]['scores'][total_matrix[j,0]]
            total_matrix[j][i+2] = score_value
        else:
            infraction_value = len(data['_checkpoint']['records'][i]['infractions'][total_matrix[j,0]])
            if infraction_value == 0:
                total_matrix[j][i+2] = None
            else:
                total_matrix[j][i+2] = infraction_value
print(total_matrix)

''''
if INFRACTIONS:
    add()
if SCORE:

if SHUTDOWN_EVENT:
'''
#CSV
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    writer.writerow(status)
    writer.writerow(num_infractions)
    for row in total_matrix:
        writer.writerow(row)

#CSV to xlsx
read_file = pd.read_csv (r'results.csv')
read_file.to_excel (r'results.xlsx', index = None, header=True)