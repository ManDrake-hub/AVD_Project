#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import json
from pynput import keyboard

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.config = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # TODO: This are all params that have to be per object
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._watchdog = None
        self._agent_watchdog = None

        self._statistics_manager = statistics_manager

        with open("./config.json", "r") as f:
            self.height = json.load(f)["height"]

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

        # TODO: captures keybaord to switch telecamera
        self.camera_index = 0
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        #keyboard.add_hotkey('a', self.switch_camera_prev)
        #keyboard.add_hotkey('d', self.switch_camera_next)
        with open("./config.json", "r") as f:
            self.n_vehicles = json.load(f)["n_vehicles"]
        self.offset = 0
        self.offset_step = 0.5

    def on_press(self, key):
        try:
            if key.char == "a":
                self.camera_index -= 1
                if self.camera_index < 0:
                    self.camera_index = self.n_vehicles-1
                return

            if key.char == "d":
                self.camera_index += 1
                if self.camera_index >= self.n_vehicles:
                    self.camera_index = 0
                return

            if key.char == "w":
                self.offset -= self.offset_step
                return

            if key.char == "s":
                self.offset += self.offset_step
                return
            
            if key.char == "r":
                self.offset = 0
                return
        except Exception as e:
            pass

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None

    def load_scenario(self, config, scenario, agents, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        # TODO: this creates a wrapper around the ego vehicle given
        self._agent_wrappers = []
        with open("./config.json", "r") as f:
            n_vehicles = json.load(f)["n_vehicles"]
        for i in range(n_vehicles):
            _agent_wrapper = AgentWrapperFactory.get_wrapper(agents[i])
            self._agent_wrappers.append(_agent_wrapper)
        self.config = config
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        for i, _agent_wrapper in enumerate(self._agent_wrappers):
            _agent_wrapper.setup_sensors(self.ego_vehicles[i])

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        self._running = True

        while self._running:
            self._tick_scenario()

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._watchdog.pause()

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                # TODO: this runs the ego vehicle
                ego_actions = []
                with open("./config.json", "r") as f:
                    n_vehicles = json.load(f)["n_vehicles"]
                for i in range(n_vehicles):
                    ego_action = self._agent_wrappers[i]()
                    ego_actions.append(ego_action)
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            # TODO: this applies the control action the first ego vehicle
            # self.ego_vehicles[0].apply_control(ego_action)
            for i in range(n_vehicles):
                self.ego_vehicles[i].apply_control(ego_actions[i])

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.config,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                # This writes ego position
                self._statistics_manager.write_live_results(
                    self.config.index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            # TODO: questo stoppa lo scenario una volta raggiunto l'obiettivo
            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                # TODO: rimuovo il set a false
                # self._running = False
                pass

            # TODO: questo setta la telecamera
            ego_trans = self.ego_vehicles[self.camera_index].get_transform()
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=self.height+self.offset),
                                                          carla.Rotation(pitch=-90)))

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return False

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                self._agent_wrapper.cleanup()
                self._agent_wrapper = None

            self.analyze_scenario()

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m'+'FAILURE'+'\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'

        ResultOutputProvider(self, global_result)
