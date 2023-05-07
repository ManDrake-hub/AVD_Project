# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
import json
import datetime
import misc
import utils_sensors
from pynput import keyboard
import time

# TODO: import the one with the correct index
# from basic_agent import BasicAgent
# from local_planner import RoadOption
BasicAgent = __import__("basic_agent_" + __file__.replace(".py", "").split("_")[-1]).BasicAgent
RoadOption = __import__("local_planner_" + __file__.replace(".py", "").split("_")[-1]).RoadOption

from behavior_types import Cautious, Aggressive, Normal

from misc import get_speed, positive, is_within_distance, compute_distance


DEBUG_SMALL = 'small'
DEBUG_MEDIUM = 'medium'
DEBUG_LARGE = 'large'

DEBUG_TYPE = {
    DEBUG_SMALL: [0.8, 0.1],
    DEBUG_MEDIUM: [0.5, 0.15],
    DEBUG_LARGE: [0.2, 0.2],
}  # Size, height

def draw_string(world, location, string='', color=(255, 255, 255, 255), life_time=-1):
    """Utility function to draw debugging strings"""
    v_shift, _ = DEBUG_TYPE.get(DEBUG_SMALL)
    l_shift = carla.Location(z=v_shift)
    color = carla.Color(*color)
    world.debug.draw_string(location + l_shift, string, False, color, life_time)

def draw_point(world, location, point_type=DEBUG_SMALL, color=(255, 255, 255, 255), life_time=-1):
    """Utility function to draw debugging points"""
    v_shift, size = DEBUG_TYPE.get(point_type, DEBUG_SMALL)
    l_shift = carla.Location(z=v_shift)
    color = carla.Color(*color)
    world.debug.draw_point(location + l_shift, size, color, life_time)

def draw_arrow(world, location1, location2, arrow_type=DEBUG_SMALL, color=(255, 255, 255, 255), life_time=-1):
    """Utility function to draw debugging points"""
    if location1 == location2:
        draw_point(world, location1, arrow_type, color, life_time)
    v_shift, thickness = DEBUG_TYPE.get(arrow_type, DEBUG_SMALL)
    l_shift = carla.Location(z=v_shift)
    color = carla.Color(*color)
    world.debug.draw_arrow(location1 + l_shift, location2 + l_shift, thickness, thickness, color, life_time)


class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        ################################################################
        self.index = int(__file__.replace(".py", "").split("_")[-1])
        ################################################################

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self.vehicle = vehicle

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        # Load the color for the debug string on the vehicle
        with open("./config.json", "r") as f:
            d = json.load(f)
            self.color = d["colors"][self.index] if "colors" in d else (255, 0, 0, 255)

        self.finished_datetime: datetime.datetime = None
        self.finished_timeout = 30
        self.prev_offset = 0.0
        self.prev_speed = 0.0
        self.time_step = 0.05 # 20Hz

        ################################################################
        # Section for GA scoring system
        ################################################################
        with open(f"./GA_score/reached_end_{self.index}", "w") as fp:
            fp.write(str(False))
        ################################################################
        self.slow_down = False
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        try:
            if key.char == "p":
                self.slow_down = not self.slow_down
                return
        except Exception as e:
            pass

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def print_state(self, state: str, line: int):
        draw_string(self._world, self._vehicle.get_location() - carla.Location(x=(line+1)*0.75), state, self.color)

    def predict_locations(self, vehicle_list, step):
        location_list = []
        for vehicle in vehicle_list:
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            acc = vehicle.get_acceleration()
            t = step * self.time_step
            loc_pred = carla.Location(x=loc.x + vel.x * t + acc.x * t**2, 
                                        y=loc.y + vel.y * t + acc.y * t**2, 
                                        z=loc.z)
            location_list.append((vehicle, loc_pred))
        return location_list

    def _predict_ego_data_prev(self, step):
            loc = self._vehicle.get_location()
            vel = self._vehicle.get_velocity()
            acc = self._vehicle.get_acceleration()
            t = step * self.time_step
            loc_pred = carla.Location(x=loc.x + vel.x * t + acc.x * t**2, 
                                        y=loc.y + vel.y * t + acc.y * t**2, 
                                        z=loc.z)
            transform_pred = self._map.get_waypoint(loc_pred, lane_type=carla.LaneType.Any).transform
            return transform_pred, loc_pred

    def predict_ego_data(self, prev_offset, step):
        if step < 0:
            return ego_transform_pred, self._predict_ego_data_prev(step)

        ego_wp, _ = self._local_planner.get_incoming_waypoint_and_direction(step)

        ego_transform_pred = ego_wp.transform
        # ego_transform_pred.rotation.yaw = -ego_transform_pred.rotation.yaw
        ego_loc_pred = ego_transform_pred.location

        # Displace the wp to the side
        r_vec = ego_transform_pred.get_right_vector()
        offset_x = prev_offset*r_vec.x
        offset_y = prev_offset*r_vec.y

        ego_loc_pred = carla.Location(x=ego_loc_pred.x + offset_x, 
                                        y=ego_loc_pred.y + offset_y, 
                                        z=ego_loc_pred.z)
        return ego_transform_pred, ego_loc_pred

    def get_base_speed(self):
        return min([
            self._behavior.max_speed,
            self._speed_limit - self._behavior.speed_lim_dist])

    def get_lane(self, location):
        return self._map.get_waypoint(location, lane_type=carla.LaneType.Any).lane_id

    def same_lane(self, vehicle, step):
        ego_wp, _ = self._local_planner.get_incoming_waypoint_and_direction(step)
        v_loc = vehicle.get_location()
        return self.get_lane(v_loc) == self.get_lane(ego_wp.transform.location)

    def opposite_lane(self, vehicle, step):
        ego_wp, _ = self._local_planner.get_incoming_waypoint_and_direction(step)
        v_loc = vehicle.get_location()
        return self.get_lane(v_loc) == -1*self.get_lane(ego_wp.transform.location)

    def get_vehicles_on_opposite_lane(self, vehicle_list, step):
        ego_wp, _ = self._local_planner.get_incoming_waypoint_and_direction(step)

        vehicle_list_opposite = []
        for vehicle in vehicle_list:
            v_loc = vehicle.get_location()
            if self.get_lane(v_loc) == -1 * self.get_lane(ego_wp.transform.location):
                vehicle_list_opposite.append(vehicle)
        return vehicle_list_opposite

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        if self.slow_down:
            time.sleep(2)

        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ################################################################
        # Obstacle Management
        ################################################################
        # Starting info
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_velocity = self._vehicle.get_velocity()
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        # Aree di interesse
        sensors = [("right", 0, 5, 10, 170), ("left", 0, 5, -160, -20), ("front", 0, 5, -5, 5), ("ahead", 0, 25, -90, 90), ("ahead_short", 0, 10, -90, 90)]
        offsets = []
        _prev_offset = self.prev_offset
        stop_overtake = False
        speed_front = None
        speed_final = self.get_base_speed()
        
        max_steps = 20
        step = -1
        print("#######################################################")
        while step < max_steps:
            #############################
            ego_transform_pred, ego_loc_pred = ego_vehicle_transform, ego_vehicle_loc
            location_list = self.predict_locations(vehicle_list, step)
            if step != -1:
                ego_transform_pred, ego_loc_pred = self.predict_ego_data(_prev_offset, step)
                location_list = [(x, x.get_location()) for x in vehicle_list]
            #############################

            draw_point(self._world, ego_loc_pred, color=(255, 0, 0, 255) if not step == -1 else (255, 255, 0, 255), life_time=-1)
            sensors_result = utils_sensors.get_sensors_locations(ego_loc_pred, ego_transform_pred, location_list, sensors)

            if sensors_result["left"] and (self.get_lane(self.predict_ego_data(0.0, step)[1] if step != -1 else ego_vehicle_loc) != self.get_lane([x[1] for x in location_list if x[0] == sensors_result["left"][0][0]][0])):
                offset = max(0, 4.0 - sensors_result["left"][0][1])
            elif sensors_result["right"]:
                offset = -2.0
                speed_final = 45
                vehicles_ahead = [x[0] for x in sensors_result["ahead"] if x[0] not in [x[0] for x in sensors_result["right"]]]
                vehicles_ahead_opposite = self.get_vehicles_on_opposite_lane(vehicles_ahead, step)
                max_steps += 1

                if len(vehicles_ahead_opposite) > 0 and not self.prev_offset < 0:
                    stop_overtake = True
                    print("Stop overtake at:", step)
            elif sensors_result["front"]:
                offset = -2.0
                speed_final = 45
                print("front activated", step, _prev_offset)
            else:
                offset = 0.0

            if sensors_result["ahead_short"]:
                vehicle_front = [x[0] for x in sensors_result["ahead_short"] if not self.opposite_lane(x[0], step)]
                if vehicle_front:
                    vehicle_front = vehicle_front[0]
                    if speed_front is None:
                        speed_front = vehicle_front.get_velocity().length() if step < 7 else self.get_base_speed()
                        print("speed is None, set to: ", speed_front, step, vehicle_front.get_velocity().length())

            _prev_offset = offset
            if step < 7:
                offsets.append(offset)
            step += 1

        offset_final = misc.exponential_weighted_average(offsets, 0.45)
        if stop_overtake:
            speed_final = speed_front if speed_front is not None else self.get_base_speed()
            offset_final = 0.0
        print(offset_final, speed_final)
        self.prev_speed = speed_final
        self.prev_offset = offset_final

        ###########################################
        # Displace the wp to the side
        r_vec = ego_vehicle_transform.get_right_vector()
        offset_x = offset_final*r_vec.x
        offset_y = offset_final*r_vec.y

        ego_wp, _ = self._local_planner.get_incoming_waypoint_and_direction(0)
        final_loc = ego_wp.transform.location
        final_loc = carla.Location(x=final_loc.x+offset_x, y=final_loc.y+offset_y, z=final_loc.z)
        draw_point(self._world, final_loc, color=(0, 0, 255, 255), life_time=-1)
        ###########################################

        self._local_planner.set_offset(offset_final)
        self._local_planner.set_speed(speed_final)

        control = self._local_planner.run_step(debug=debug)
        return control
        ################################################################

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
