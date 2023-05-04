#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import importlib

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

import json
from utils import Streamer

def get_entry_point():
    return 'MyTeamAgent'

class MyTeamAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    ###########################################################################################################################
    def setup(self, path_to_conf_file, index):
    ###########################################################################################################################
        """
        Setup the agent parameters
        """
        ###########################################################################################################################
        self.BasicAgent = __import__("basic_agent" + str(index)).BasicAgent
        
        self.track = Track.SENSORS
        self.index = index
        self._hero_actor = None
        with open("./config.json", "r") as f:
            d = json.load(f)
            self.color = d["colors"][self.index] if "colors" in d else (255, 0, 0, 255)
            self.color = carla.Color(*self.color)
        ###########################################################################################################################
        
        # self.client = carla.Client("127.0.0.1", 6000)
        # self.client.set_timeout(30)
        # self.traffic_manager = self.client.get_trafficmanager(8000)

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

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation. 
        """

        # for actor in CarlaDataProvider.get_world().get_actors():
        #     self.traffic_manager.ignore_vehicles_percentage(actor, 100)

        if not self._agent:
            # Search for the ego actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                # TODO: This finds the hero actor
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == f'hero{self.index}':
                    hero_actor = actor
                    self._hero_actor = hero_actor
            if not hero_actor:
                return carla.VehicleControl()
            self._agent = self.BasicAgent(hero_actor, opt_dict=self.configs, index=self.index)

            # TODO: connect the AI with the vehicle

            plan = []
            prev_wp = None
            for transform, _ in self._global_plan_world_coord:
                wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                if prev_wp:
                    plan.extend(self._agent.trace_route(prev_wp, wp))
                prev_wp = wp

            self._agent.set_global_plan(plan)

            return carla.VehicleControl()

        else:
            # Release other vehicles 
            # vehicle_list = CarlaDataProvider.get_world().get_actors().filter("*vehicle*")
            # for actor in vehicle_list:
            #     if not('role_name' in actor.attributes and actor.attributes['role_name'] == 'hero'):
            #         actor.destroy()
            controls = self._agent.run_step()
            if self.__show:
                self.showServer.send_frame("RGB", input_data["Center"][1])
                self.showServer.send_data("Controls",{ 
                "steer":controls.steer, 
                "throttle":controls.throttle, 
                "brake": controls.brake,
                })
            ###########################################################################################################################
            if len(self.configs["SaveSpeedData"]) > 0:
                with open("./"+self.configs["SaveSpeedData"].replace(".", f"{self.index}."),"a") as fp:
                    fp.write(str(timestamp)+";"+str(input_data["Speed"][1]["speed"] * 3.6)+";"+str(self.configs["target_speed"])+"\n")
                    fp.close()
            CarlaDataProvider.get_world().debug.draw_string(self._hero_actor.get_location(), str(self.index), color=self.color)
            ###########################################################################################################################
            return controls

    def destroy(self):
        print("DESTROY")
        if self._agent:
            self._agent.reset()
            