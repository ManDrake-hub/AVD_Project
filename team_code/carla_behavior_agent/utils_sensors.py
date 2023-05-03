from misc import compute_magnitude_angle_with_sign, is_hero


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

def get_sensors_locations(ego_vehicle_location, ego_vehicle_transform, location_list, sensors, max_distance=60):
    """
        :param ego_vehicle_location: ego vehicle location
        :param ego_vehicle_transform: ego vehicle transform
        :param location_list: list of tuples formatted as (actor, location)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :param max_distance: vehicles with greater distance than this will be ignored (can be set to None to include all vehicles)
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    def dist(location): return location.distance(ego_vehicle_location)
    vehicles = [(vehicle, *compute_magnitude_angle_with_sign(location, ego_vehicle_location, ego_vehicle_transform.rotation.yaw)) for vehicle, location in location_list 
                if ((dist(location) < max_distance) if max_distance is not None else True) and not is_hero(vehicle)]
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