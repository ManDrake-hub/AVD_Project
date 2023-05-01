import os
import subprocess
import json
import glob
import random
import shutil
import numpy as np


def start_sim():
    com = ["./run_test_controller.sh"]
    process = subprocess.Popen(com, shell=True)
    try:
        process.wait()
    except Exception as e:
        remove_generated_files()
        raise e

def create_random_config(target_speed, index, eps=0.0001):
    kp = random.random() * 10 + eps
    ki = random.random() * 10 + eps
    kd = random.random() * 10 + eps

    kv = random.random() * 10 + eps
    ks = random.random() * 10 + eps
    ld = random.random() * 3 + eps

    d = {
        "sensors" : [
                {"type": "sensor.speedometer", "id": "Speed"}
        ],
        "longitudinal_control_dict" : {"K_P": kp, "K_I": ki, "K_D": kd, "dt": 0.05},
        "lateral_control_dict" : {"K_V": kv, "K_S": ks, "lookahead_distance": ld, "dt": 0.05},
        "target_speed": target_speed,
        "Visualizer_IP" : "",
        "SaveSpeedData" : "speed.txt",
        "ignore_vehicles": True
    }

    with open(f"./carla_behavior_agent/config_agent_basic_{index}.json", "w") as f:
        json.dump(d, f)

def create_child_config(parents, target_speed, index, mutation=0.01, eps=0.000001):
    def get_parent_values():
        parent = random.choice(parents)
        return parent
    
    mutate = random.random() < mutation
    kp = get_parent_values()["longitudinal_control_dict"]["K_P"] if not mutate else random.random() * 25 + eps
    ki = get_parent_values()["longitudinal_control_dict"]["K_I"] if not mutate else random.random() * 25 + eps
    kd = get_parent_values()["longitudinal_control_dict"]["K_D"] if not mutate else random.random() * 25 + eps

    kv = get_parent_values()["lateral_control_dict"]["K_V"] if not mutate else random.random() * 25 + eps
    ks = get_parent_values()["lateral_control_dict"]["K_S"] if not mutate else random.random() * 2 + eps
    ld = get_parent_values()["lateral_control_dict"]["lookahead_distance"] if not mutate else random.random() * 3 + eps

    d = {
        "sensors" : [
                {"type": "sensor.speedometer", "id": "Speed"}
        ],
        "longitudinal_control_dict" : {"K_P": kp, "K_I": ki, "K_D": kd, "dt": 0.05},
        "lateral_control_dict" : {"K_V": kv, "K_S": ks, "lookahead_distance": ld, "dt": 0.05},
        "target_speed": target_speed,
        "Visualizer_IP" : "",
        "SaveSpeedData" : "speed.txt",
        "ignore_vehicles": True
    }

    with open(f"./carla_behavior_agent/config_agent_basic_{index}.json", "w") as f:
        json.dump(d, f)

def create_config(d, index):
    with open(f"./carla_behavior_agent/config_agent_basic_{index}.json", "w") as f:
        json.dump(d, f)

def calculate_position_score(file):
    acc = 0
    with open(file, "r") as f:
        entries = f.readlines()
    for entry in entries:
        acc -= float(entry)**2
    return acc/len(entries)

def calculate_speed_score(file):
    acc = 0
    with open(file, "r") as f:
        entries = f.readlines()
    for entry in entries:
        acc -= float(entry)
    return acc/len(entries)

def has_reached(file):
    with open(file, "r") as f:
        return bool(f.read())

generated_files = []

def remove_generated_files():
    global generated_files
    for file in generated_files:
        os.remove(file)

if __name__ == "__main__":
    n_parents = 3

    # 0) Read number of vehicles
    with open("./config.json", "r") as f:
        n_vehicles = json.load(f)["n_vehicles"]

    files = glob.glob("./carla_behavior_agent/*_0.py")
    for file in files:
        for i in range(1, n_vehicles):
            file_new = file.replace("_0.py", f"_{i}.py")
            generated_files.append(file_new)
            shutil.copyfile(file, file_new)

    if os.path.exists("./carla_behavior_agent/best_scores.txt"):
        os.remove("./carla_behavior_agent/best_scores.txt")

    parents = []
    for gen in range(10):
        # 1) Clear scores and configs if present
        files = glob.glob("./GA_score/*") + \
                glob.glob('./carla_behavior_agent/config_agent_basic_*') + \
                glob.glob('./results/*')
        for f in files:
            os.remove(f)

        # 2) Create N controller configs (random or using mix of best M cars with mutation)
        if not parents:
            for i in range(n_vehicles):
                create_random_config(30, i)
        else:
            _p = [x[0] for x in parents]
            for i in range(n_parents):
                create_config(_p[i], n_vehicles-n_parents+i)
            for i in range(n_vehicles-n_parents):
                create_child_config(_p, 30, i, mutation=0.05)
        parents = []

        # 3) Start sim
        start_sim()

        # 4) Calculate score for each vehicle
        locations = glob.glob(f"./GA_score/position_error_*")
        speeds = glob.glob(f"./GA_score/speed_error_*")
        reacheds = glob.glob(f"./GA_score/reached_end_*")

        index = 0
        for location, speed, reached in zip(locations, speeds, reacheds):
            if has_reached(reached):
                score_loc = calculate_position_score(location)
                score_speed = calculate_position_score(speed)
                score = score_loc + score_speed
            else:
                score = -np.inf

            with open(f"./carla_behavior_agent/config_agent_basic_{index}.json", "r") as f:
                d = json.load(f)
            parents.append((d, score, score_loc, score_speed))
            index += 1

        # 5) Select best parents and go to step (1)
        print([x[1] for x in parents])
        parents = sorted(parents, reverse=True, key=lambda x: x[1])
        print([x[1] for x in parents])
        parents = parents[:n_parents]

        with open("./carla_behavior_agent/best_scores.txt", "a") as f:
            f.write(json.dumps(parents[0]))
            f.write("\n")

    remove_generated_files()
    with open(f"./carla_behavior_agent/config_agent_basic_best.json", "w") as f:
        json.dump(parents[0], f)