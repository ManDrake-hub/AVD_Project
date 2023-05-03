import json
import pandas as pd
import csv
import numpy as np

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
infractions_dict = data['_checkpoint']['global_record']['infractions']
values = [float(val) for val in list(infractions_dict.values())]
shutdown_event_list = ["route_dev", "vehicle_blocked", "route_timeout"]

# Percentages
total = sum(values)
percentages = [val/total*100 for val in values]
for key in infractions_dict:
    infractions_dict[key] = f"{percentages.pop(0):.2f}%"

# Matrix for each route
total_matrix = np.empty([len(infractions_dict), 2+len(routes)], dtype = object)
# Global
for i, (k, v) in enumerate(infractions_dict.items()):
    total_matrix[i,0] = k
    total_matrix[i,1] = v
# Routes
for i in range(len(routes)):
    for j in range(len(total_matrix)):
        if total_matrix[j,0] in shutdown_event_list: 
            infraction_value = data['_checkpoint']['records'][i]['infractions'][total_matrix[j,0]]
            if len(infraction_value) == 0:
                total_matrix[j][i+2] = None
            else:
                total_matrix[j][i+2] = infraction_value[0]
        else:
            infraction_value = len(data['_checkpoint']['records'][i]['infractions'][total_matrix[j,0]])
            if infraction_value == 0:
                total_matrix[j][i+2] = None
            else:
                total_matrix[j][i+2] = infraction_value
print(total_matrix)

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