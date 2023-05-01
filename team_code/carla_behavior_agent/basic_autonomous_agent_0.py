#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
# TODO: remove this import, we are going to do one later with the index to use
# from behavior_agent import BehaviorAgent
BehaviorAgent = __import__("behavior_agent_" + __file__.replace(".py", "").split("_")[-1]).BehaviorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import importlib
from agents.navigation.local_planner import RoadOption

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

import json
from utils import Streamer
import xmltodict

def get_entry_point():
    return 'MyTeamAgent'

class MyTeamAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    # TODO: add index to inputs
    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._agent = None

        ###################################################################
        # TODO: load and add some vars that will be used later
        self.index = int(__file__.replace(".py", "").split("_")[-1])

        self._hero_actor = None
        # Load the color for the debug string on the vehicle
        with open("./config.json", "r") as f:
            d = json.load(f)
            self.color = d["colors"][self.index] if "colors" in d else (255, 0, 0, 255)
            self.color = carla.Color(*self.color)

        with open("./config.json", "r") as f:
            self.debug_local = bool(json.load(f)["debug_local"])
        ###################################################################
        
        with open(path_to_conf_file, "r") as f:
            self.configs = json.load(f)
            f.close()
        
        self.__show = len(self.configs["Visualizer_IP"]) > 0
        
        if self.__show:
            self.showServer = Streamer(self.configs["Visualizer_IP"])

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """
        return self.configs["sensors"]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self._global_plan_world_coord = global_plan_world_coord
        self._global_plan = global_plan_gps

    def draw_waypoints(self, waypoints, vertical_shift=0.1, size=0.05, persistency=10000):
        for w in waypoints:
            wp = w[0].transform.location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green
            CarlaDataProvider.get_world().debug.draw_point(wp, size=size, color=color, life_time=persistency)

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation. 
        """
        if not self._agent:
            
            # Search for the ego actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                # TODO: change role_name to search for
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == f'hero{self.index}':
                    hero_actor = actor
                    # TODO: save the hero actor
                    self._hero_actor = hero_actor

            if not hero_actor:
                return carla.VehicleControl()
            
            # TODO: load the agent imported
            self._agent = BehaviorAgent(hero_actor, opt_dict=self.configs)

            # TODO: this creates the plan that we are going to use
            plan = []
            plan = [(CarlaDataProvider.get_map().get_waypoint(x[0].location), x[1]) for x in self._global_plan_world_coord]

            ####################################################
            # TODO: this is a problem, if we draw the waypoints given to our agent,
            # they are different from the one displayed by our route_scenario

            with open("./config.json", "r") as f:
                self.show_plan = bool(json.load(f)["show_plan"])
            if self.show_plan:
                self.draw_waypoints(waypoints=plan)
            ####################################################

            self._agent.set_global_plan(plan)

            return carla.VehicleControl()

        else:

            controls = self._agent.run_step(debug=self.debug_local)
            if self.__show:
                self.showServer.send_frame("RGB", input_data["Center"][1])
                self.showServer.send_data("Controls",{ 
                "steer":controls.steer, 
                "throttle":controls.throttle, 
                "brake": controls.brake,
                })
            # TODO: Save the speed in a different file for each vehicle
            if len(self.configs["SaveSpeedData"]) > 0:
                with open("./"+self.configs["SaveSpeedData"].replace(".", f"{self.index}."),"a") as fp:
                    fp.write(str(timestamp)+";"+str(input_data["Speed"][1]["speed"] * 3.6)+";"+str(self.configs["target_speed"])+"\n")
            # TODO: Show on the vehicle, the index associated with the agent
            CarlaDataProvider.get_world().debug.draw_string(self._hero_actor.get_location(), str(self.index), color=self.color)

            ################################################################
            # Section for GA scoring system
            ################################################################
            with open(f"./GA_score/speed_error_{self.index}", "a") as fp:
                fp.write(str(abs(self.configs["target_speed"] - input_data["Speed"][1]["speed"] * 3.6))+"\n")
            ################################################################

            return controls

    def destroy(self):
        print("DESTROY")
        if self._agent:
            self._agent.reset()
            