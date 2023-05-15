#!/bin/bash
export ROUTES=/home/colander/Desktop/Project/AVD_Project/userCode/routes_avddiem_exam.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/home/colander/Desktop/Project/AVD_Project/userCode/carla_behavior_agent/basic_autonomous_agent_0.py
export TEAM_CONFIG=/home/colander/Desktop/Project/AVD_Project/userCode/carla_behavior_agent/config_agent_basic_0.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=localhost
export CARLA_PORT=2000
export CARLA_TRAFFIC_MANAGER_PORT=8000
export CHECKPOINT_ENDPOINT=/home/colander/Desktop/Project/AVD_Project/userCode/results/simulation_results.json
export DEBUG_CHECKPOINT_ENDPOINT=/home/colander/Desktop/Project/AVD_Project/userCode/results/live_results.txt
export RESUME=0
export TIMEOUT=60

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--host=${CARLA_HOST} \
--port=${CARLA_PORT} \
--timeout=${TIMEOUT} \
--traffic-manager-port=${CARLA_TRAFFIC_MANAGER_PORT} 
