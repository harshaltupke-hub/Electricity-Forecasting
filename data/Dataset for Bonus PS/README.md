# TSO Optimization Support Dataset
Files:
- network_buses.csv: one optimization bus for each existing forecast zone.
- transmission_lines.csv: simplified three-zone network links and finite transfer limits.
- generator_operating_limits.csv: aggregate generation resources per zone/technology with capacities matching system_capacity_reference.csv exactly; adds optimization assumptions (minimum output, ramping, costs).
- optimization_parameters.csv: 99.9% served-load rule and objective cost coefficients.
- zone_bus_mapping.csv: exact mapping between existing forecast columns and optimization buses.
