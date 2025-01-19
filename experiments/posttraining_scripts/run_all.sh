#!/bin/bash

# File to save the timing results
output_file="timing_results.txt"

# # unconstrained
# { time ./unconstrained-synchronous_multisensor-dsads.sh > /dev/null 2>&1; } 2>> log.txt
# { time ./unconstrained-synchronous_multisensor-rwhar.sh > /dev/null 2>&1; } 2>> log.txt

# # opportunistic
# { time ./opportunistic-asynchronous_single_sensor-dsads.sh > /dev/null 2>&1; } 2>> log.txt
# { time ./opportunistic-asynchronous_single_sensor-rwhar.sh > /dev/null 2>&1; } 2>> log.txt

# { time ./opportunistic-asynchronous_multisensor-dsads.sh > /dev/null 2>&1; } 2>> log.txt
# { time ./opportunistic-asynchronous_multisensor-rwhar.sh > /dev/null 2>&1; } 2>> log.txt

# { time ./opportunistic-asynchronous_multisensor_time_context-dsads.sh > /dev/null 2>&1; } 2>> log.txt
# { time ./opportunistic-asynchronous_multisensor_time_context-rwhar.sh > /dev/null 2>&1; } 2>> log.txt

# # conservative
{ time ./conservative-asynchronous_single_sensor-dsads.sh > /dev/null 2>&1; } 2>> log.txt
{ time ./conservative-asynchronous_single_sensor-rwhar.sh > /dev/null 2>&1; } 2>> log.txt

{ time ./conservative-asynchronous_multisensor-dsads.sh > /dev/null 2>&1; } 2>> log.txt
{ time ./conservative-asynchronous_multisensor-rwhar.sh > /dev/null 2>&1; } 2>> log.txt

{ time ./conservative-asynchronous_multisensor_time_context-dsads. > /dev/null 2>&1; } 2>> log.txt
{ time ./conservative-asynchronous_multisensor_time_context-rwhar.sh > /dev/null 2>&1; } 2>> log.txt