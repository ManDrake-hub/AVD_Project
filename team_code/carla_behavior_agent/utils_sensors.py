from misc import is_hero
import math
import numpy as np
import carla


def compute_magnitude_angle_with_sign(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])

    d_angle = math.degrees(math.atan2(np.cross(forward_vector, target_vector), np.dot(forward_vector, target_vector)))
    return (norm_target, d_angle)

# TODO: fare in modo che aggiungiamo al sensore in base ad uno dei qualsiasi punti che approssimano la macchina

def get_sensors_locations_fw(ego_vehicle_location, ego_vehicle_transform, location_list, sensors, max_distance=60):
    """
        :param ego_vehicle_location: ego vehicle location
        :param ego_vehicle_transform: ego vehicle transform
        :param location_list: list of tuples formatted as (actor, location)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :param max_distance: vehicles with greater distance than this will be ignored (can be set to None to include all vehicles)
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    def _compute_info(ego_location, location): return compute_magnitude_angle_with_sign(location, ego_location, ego_vehicle_transform.rotation.yaw)
    def dist(location): return location.distance(ego_vehicle_location)
    vehicle_length_step = 4.0 / 1

    ego_locs = []
    ego_fw = ego_vehicle_transform.rotation.get_forward_vector()
    for vehicle_step in [-1, 0, 1]:
        offset_x = vehicle_step*vehicle_length_step*ego_fw.x
        offset_y = vehicle_step*vehicle_length_step*ego_fw.y
        ego_location = carla.Location(x=ego_vehicle_location.x + offset_x, 
                                        y=ego_vehicle_location.y + offset_y, 
                                        z=ego_vehicle_location.z)
        ego_locs.append(ego_location)

    vehicles = []
    for vehicle, location, fw in location_list:
        if is_hero(vehicle):
            continue

        if dist(location) > max_distance:
            continue

        # https://carla.readthedocs.io/en/0.9.5/measurements/
        # vehicle_length_step = vehicle.bounding_box.extent.x / 4
        locations = []
        for ego_loc in ego_locs:
            for step in [-1, 0, 1]:
                offset_x = step*vehicle_length_step*fw.x
                offset_y = step*vehicle_length_step*fw.y

                location = carla.Location(x=location.x + offset_x, 
                                        y=location.y + offset_y, 
                                        z=location.z)
                locations.append(location)
            infos = [_compute_info(ego_loc, x) for x in locations]
        for info in infos:
            vehicles.append((vehicle, *info))
    vehicles.sort(key=lambda x: x[1])
    return _get_sensors(sensors, vehicles)

def get_sensors_transform(ego_vehicle_location, ego_vehicle_transform, transform_list, sensors, max_distance=60):
    """
        :param ego_vehicle_location: ego vehicle location
        :param ego_vehicle_transform: ego vehicle transform
        :param location_list: list of tuples formatted as (actor, location)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :param max_distance: vehicles with greater distance than this will be ignored (can be set to None to include all vehicles)
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    def _compute_info(location): return compute_magnitude_angle_with_sign(location, ego_vehicle_location, ego_vehicle_transform.rotation.yaw)

    vehicles = []
    for vehicle, transform in transform_list:
        if is_hero(vehicle):
            continue

        # https://carla.readthedocs.io/en/0.9.5/measurements/
        # vehicle_length_step = vehicle.bounding_box.extent.x / 4
        vehicle_length_step = 3.0 / 4
        locations = []
        for step in [-1, 0, 1]:
            r_vec = transform.get_forward_vector()
            offset_x = step*vehicle_length_step*r_vec.x
            offset_y = step*vehicle_length_step*r_vec.y

            location = carla.Location(x=transform.location.x + offset_x, 
                                    y=transform.location.y + offset_y, 
                                    z=transform.location.z)
            locations.append(location)
        info = min([_compute_info(x) for x in locations], key=lambda x: x[0])
        if info[0] > max_distance:
            continue
        vehicles.append((vehicle, *info))
    vehicles.sort(key=lambda x: x[1])
    return _get_sensors(sensors, vehicles)














def get_sensors(ego_vehicle_location, ego_vehicle_transform, vehicle_list, sensors, max_distance=60):
    """
        :param ego_vehicle_location: ego vehicle location
        :param ego_vehicle_transform: ego vehicle transform
        :param vehicle_list: list of actors (not limited to vehicles)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :param max_distance: vehicles with greater distance than this will be ignored (can be set to None to include all vehicles)
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    def dist(v): return v.get_location().distance(ego_vehicle_location)
    vehicles = [(x, *compute_magnitude_angle_with_sign(x.get_location(), ego_vehicle_location, ego_vehicle_transform.rotation.yaw)) for x in vehicle_list 
                if ((dist(x) < max_distance) if max_distance is not None else True) and not is_hero(x)]
    vehicles.sort(key=lambda x: x[1])
    return _get_sensors(sensors, vehicles)

def get_sensors_ego(ego_vehicle, vehicle_list, sensors, max_distance=60):
    """
        :param ego_vehicle: ego vehicle
        :param vehicle_list: list of actors (not limited to vehicles)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :param max_distance: vehicles with greater distance than this will be ignored (can be set to None to include all vehicles)
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    ego_vehicle_location = ego_vehicle.get_location()
    ego_vehicle_transform = ego_vehicle.get_transform()
    return get_sensors(ego_vehicle_location, ego_vehicle_transform, vehicle_list, sensors, max_distance)

def get_sensors_locations_ga(ego_vehicle_location, ego_vehicle_transform, ego_vehicle_velocity, location_list, sensors, max_distance=60, max_vehicles=1):
    """
        :param ego_vehicle_location: ego vehicle location
        :param ego_vehicle_transform: ego vehicle transform
        :param ego_vehicle_transform: ego vehicle velocity (Vector)
        :param location_list: list of tuples formatted as (actor, location)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :param max_distance: vehicles with greater distance than this will be ignored (can be set to None to include all vehicles)
        :param max_vehicles: how many vehicles to track per sensor
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    def dist(location): return location.distance(ego_vehicle_location)
    vehicles = [(vehicle, *compute_magnitude_angle_with_sign(location, ego_vehicle_location, ego_vehicle_transform.rotation.yaw)) for vehicle, location in location_list 
                if ((dist(location) < max_distance) if max_distance is not None else True) and not is_hero(vehicle)]
    vehicles.sort(key=lambda x: x[1])
    return _get_sensors_ga(ego_vehicle_velocity, sensors, vehicles, max_vehicles)

def _get_sensors(sensors, vehicle_list_with_info):
    """Do not call this"""
    sensors_result = {}
    for sensor in sensors:
        sensor_id, sensor_range_min, sensor_range_max, sensor_angle_min, sensor_angle_max = sensor
        sensors_result[sensor_id] = []
        for v in vehicle_list_with_info:
            vehicle, distance, angle = v
            if (sensor_range_min <= distance <= sensor_range_max and sensor_angle_min <= angle <= sensor_angle_max):
                sensors_result[sensor_id].append((vehicle, distance, angle))
    return sensors_result

def _get_sensors_ga(ego_vehicle_velocity, sensors, vehicle_list_with_info, max_vehicles):
    """Do not call this"""
    sensors_result = {}
    for sensor in sensors:
        sensor_id, sensor_range_min, sensor_range_max, sensor_angle_min, sensor_angle_max = sensor
        sensors_result[sensor_id] = {"active": False, "distances": [], "angles": [], "speed_deltas": []}
        for v in vehicle_list_with_info:
            vehicle, distance, angle = v
            if (sensor_range_min <= distance <= sensor_range_max and sensor_angle_min <= angle <= sensor_angle_max):
                if len(sensors_result[sensor_id]["distances"]) >= max_vehicles:
                    break
                sensors_result[sensor_id]["active"] = True
                sensors_result[sensor_id]["distances"].append((distance - sensor_range_min) / (sensor_range_max - sensor_range_min))
                sensors_result[sensor_id]["angles"].append((angle - sensor_angle_min) / (sensor_angle_max - sensor_angle_min))
                sensors_result[sensor_id]["speed_deltas"].append((vehicle.get_velocity().length() / (ego_vehicle_velocity.length() + 1e-5)) - 1.0)
                # sensors_result[sensor_id]["dunno"].append((vehicle.get_velocity().length() / (self._vehicle.get_velocity().length() + 1e-5)) - 1.0)
    return sensors_result