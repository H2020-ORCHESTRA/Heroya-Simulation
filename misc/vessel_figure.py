import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Provided dictionary
vessels_data = {
    'Vessel A': {'arrival': datetime(2023, 3, 15, 15, 42), 'departure': datetime(2023, 3, 17, 16, 2)},
    'Vessel B': {'arrival': datetime(2023, 3, 15, 23, 24), 'departure': datetime(2023, 3, 16, 19, 0)},
    'Vessel C': {'arrival': datetime(2023, 3, 17, 7, 56), 'departure': datetime(2023, 3, 17, 23, 59)},
    'Vessel D': {'arrival': datetime(2023, 3, 15, 3, 21), 'departure': datetime(2023, 3, 15, 21, 37)},
    'Vessel E': {'arrival': datetime(2023, 3, 14, 2, 54), 'departure': datetime(2023, 3, 15, 3, 13)},
    'Vessel F': {'arrival': datetime(2023, 3, 18, 7, 51), 'departure': datetime(2023, 3, 18, 22, 49)},
    'Vessel G': {'arrival': datetime(2023, 3, 18, 23, 0), 'departure': datetime(2023, 3, 19, 16, 54)},
    'Vessel H': {'arrival': datetime(2023, 3, 14, 3, 29), 'departure': datetime(2023, 3, 16, 12, 12)},
    'Vessel I' : {'arrival': datetime(2023, 3, 18, 0, 2), 'departure': datetime(2023, 3, 18, 20, 41)},
    'Vessel J' : {'arrival': datetime(2023, 3, 16, 17, 43), 'departure': datetime(2023, 3, 20, 6, 56)},
}

# Group vessels with the same name
grouped_vessels = {}
for name, data in vessels_data.items():
    if name not in grouped_vessels:
        grouped_vessels[name] = {'arrivals': [], 'departures': []}
    grouped_vessels[name]['arrivals'].append(data['arrival'])
    grouped_vessels[name]['departures'].append(data['departure'])

# Plotting the timeline with horizontal bars for each group
fig, ax = plt.subplots(figsize=(10, 6))
for name, group_data in grouped_vessels.items():
    for arrival, departure in zip(group_data['arrivals'], group_data['departures']):
        ax.barh(name, width=(departure - arrival), left=arrival)

# Beautify the plot
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
plt.xlabel("Timeline")
plt.title("Vessels Arrival and Departure Timeline")

# Show the plot
plt.show()
