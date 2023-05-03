#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import numpy as np
import carla

def draw_waypoints(world, waypoints, z=0.5, color=(255, 0, 0, 255), lifetime=1.0):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        _color = carla.Color(*color)
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=lifetime, color=_color)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def get_trafficlight_trigger_location(traffic_light):
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """
    def rotate_point(point, radians):
        """
        rotate a given point by a given angle
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)


def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be tkaen into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # We don't care about the angle, nothing else to check
    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle

"""
θ = atan2(b x a, a · b),

where θ is the signed angle between the vectors, a and b are the two vectors, x denotes the cross product of two vectors, · denotes the dot product of two vectors, and atan2 is a function that returns the angle in radians between -π and π.

In this formula, the cross product of two vectors a and b is a scalar given by a x b = |a| |b| sin(θ), where θ is the angle between the vectors. The sign of the cross product indicates the orientation of the vectors relative to each other.

So, if the z-component of the cross product a x b is positive, then the angle between the vectors is counterclockwise, and if it is negative, then the angle is clockwise.

The angle is signed, which means that it can be positive or negative. A positive angle indicates a counterclockwise rotation from the first vector to the second vector, while a negative angle indicates a clockwise rotation.
"""

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

def get_sensors(ego_vehicle, vehicle_list, sensors, max_distance=60):
    """
        :param ego_vehicle: ego vehicle
        :param vehicle_list: list of actors (not limited to vehicles)
        :param sensors: list of tuples formatted as (sensor_id: str, range_min: float, range_max: float, angle_min: float, angle_max: float)
        :return: a dict with sensor_id as key and list of tuples formatted as (actor, distance, angle) as values
    """
    ego_vehicle_loc = ego_vehicle.get_location()
    ego_vehicle_transform = ego_vehicle.get_transform()

    def dist(v): return v.get_location().distance(ego_vehicle_loc)
    vehicles = [(x, *compute_magnitude_angle_with_sign(x.get_location(), ego_vehicle_loc, ego_vehicle_transform.rotation.yaw)) for x in vehicle_list 
                if ((dist(x) < max_distance) if max_distance is not None else True) and 
                (not 'role_name' in x.attributes or ('role_name' in x.attributes and not 'hero' in x.attributes['role_name']))]
    vehicles.sort(key=lambda x: x[1][0])

    sensors_result = {}
    for sensor in sensors:
        sensor_id, sensor_range_min, sensor_range_max, sensor_angle_min, sensor_angle_max = sensor
        sensors_result[sensor_id] = []
        for v in vehicles:
            vehicle, distance, angle = v
            if (sensor_range_min <= distance <= sensor_range_max and sensor_angle_min <= angle <= sensor_angle_max):
                sensors_result[sensor_id].append((vehicle, distance, angle))
    return sensors_result

def get_sensors_ga(ego_vehicle, vehicle_list, sensors, max_distance=60, max_vehicles=2):
    # sensors: [ (id, range_min, range_max, angle_min, angle_max) ]
    # out:     {sensor_id: {"active", "distances", "angles", "speed_deltas"}}
    ego_vehicle_loc = ego_vehicle.get_location()
    ego_vehicle_transform = ego_vehicle.get_transform()

    def dist(v): return v.get_location().distance(ego_vehicle_loc)
    vehicles = [(x, *compute_magnitude_angle_with_sign(x.get_location(), ego_vehicle_loc, ego_vehicle_transform.rotation.yaw)) for x in vehicle_list 
                if dist(x) < max_distance and 
                (not 'role_name' in x.attributes or ('role_name' in x.attributes and not 'hero' in x.attributes['role_name']))]
    vehicles.sort(key=lambda x: x[1][0])

    sensors_result = {}
    for sensor in sensors:
        sensor_id, sensor_range_min, sensor_range_max, sensor_angle_min, sensor_angle_max = sensor
        sensors_result[sensor_id] = {"active": False, "distances": [], "angles": [], "speed_deltas": []}
        for v in vehicles:
            vehicle, distance, angle = v
            if (sensor_range_min <= distance <= sensor_range_max and sensor_angle_min <= angle <= sensor_angle_max):
                if len(sensors_result[sensor_id]["distances"]) >= max_vehicles:
                    break
                sensors_result[sensor_id]["active"] = True
                sensors_result[sensor_id]["distances"].append((distance - sensor_range_min) / (sensor_range_max - sensor_range_min)) # TODO: change
                sensors_result[sensor_id]["angles"].append((angle - sensor_angle_min) / (sensor_angle_max - sensor_angle_min))
                sensors_result[sensor_id]["speed_deltas"].append((vehicle.get_velocity().length() / (self._vehicle.get_velocity().length() + 1e-5)) - 1.0)
                # sensors_result[sensor_id]["dunno"].append((vehicle.get_velocity().length() / (self._vehicle.get_velocity().length() + 1e-5)) - 1.0)
    return sensors_result

def get_sensors_ga(self, ego_vehicle_loc, ego_vehicle_transform, vehicle_list, sensors, max_distance=30, max_vehicles=2):
    # sensors: [ (id, range_min, range_max, angle_min, angle_max) ]
    # out:     {sensor_id: {"active", "distances", "angles", "speed_deltas"}}

    def dist(v): return v.get_location().distance(ego_vehicle_loc)
    vehicles = [(x, compute_magnitude_angle_with_sign(x.get_location(), ego_vehicle_loc, ego_vehicle_transform.rotation.yaw)) for x in vehicle_list 
                if dist(x) < max_distance and 
                (not 'role_name' in x.attributes or ('role_name' in x.attributes and not 'hero' in x.attributes['role_name']))]

    vehicles.sort(key=lambda x: x[1][0])
    sensors_result = {}
    for sensor in sensors:
        sensor_id, sensor_range_min, sensor_range_max, sensor_angle_min, sensor_angle_max = sensor
        sensors_result[sensor_id] = {"active": False, "distances": [], "angles": [], "speed_deltas": []}
        for v in vehicles:
            vehicle, distance_angle = v
            distance, angle = distance_angle
            if (sensor_range_min <= distance <= sensor_range_max and sensor_angle_min <= angle <= sensor_angle_max):
                if len(sensors_result[sensor_id]["distances"]) >= max_vehicles:
                    break
                sensors_result[sensor_id]["active"] = True
                sensors_result[sensor_id]["distances"].append((distance - sensor_range_min) / (sensor_range_max - sensor_range_min)) # TODO: change
                sensors_result[sensor_id]["angles"].append((angle - sensor_angle_min) / (sensor_angle_max - sensor_angle_min))
                sensors_result[sensor_id]["speed_deltas"].append((vehicle.get_velocity().length() / (self._vehicle.get_velocity().length() + 1e-5)) - 1.0)
                # sensors_result[sensor_id]["dunno"].append((vehicle.get_velocity().length() / (self._vehicle.get_velocity().length() + 1e-5)) - 1.0)
    return sensors_result

def compute_magnitude_angle(target_location, current_location, orientation):
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
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)

def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0
