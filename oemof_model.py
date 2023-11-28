# ****************************************************************************
# ********** Define and optimise the energy system ******************
# ****************************************************************************

# import required packages
from oemof.tools import logger
from oemof.tools import economics
import oemof.solph as solph
import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

solver = "cbc"  # 'glpk', 'gurobi',....
debug = False  # Set number_of_timesteps to 3 to get a readable lp-file.
number_of_time_steps = 8760
solver_verbose = False  # show/hide solver output

# initiate the logger (see the API docs for more information)
logger.define_logging(
    logfile="oemof_example.log",
    screen_level=logging.INFO,
    file_level=logging.INFO,
)

logging.info("Initialize the energy system")
date_time_index = pd.date_range("1/1/2021", periods=number_of_time_steps, freq="H")
energysystem = solph.EnergySystem(timeindex=date_time_index)

###############################################################################
# Input data
###############################################################################

# Read data file
filename = os.path.join(os.getcwd(), "input_data.csv")
data = pd.read_csv(filename)

# wacc / interest rate
wacc = 0.05

# estimate epc for different technologies

# PV
capex_pv = 800  # capital costs in €/KW
opex_pv_pct = 3  # operational costs in % of capex
opex_pv = opex_pv_pct * capex_pv / 100
n_pv = 25  # lifetime in years
annuity_pv = economics.annuity(capex=capex_pv, n=n_pv, wacc=wacc)
epc_pv = annuity_pv + opex_pv

# Grid
epc_grid = 109  # grid capacity price in €/kW/a
electricity_price = 0.4  # €/kWh

##########################################################################
# Create oemof object
##########################################################################

# Create oemof object
logging.info("Create oemof objects")

# create electricity bus
b_el = solph.Bus(label="electricity")

# adding the buses to the energy system
energysystem.add(b_el)

# create excess component for the electricity bus to allow overproduction
energysystem.add(solph.components.Sink(label="excess_bel", inputs={b_el: solph.Flow(variable_costs=1000)}))

# create fixed source object representing pv power plants
energysystem.add(
    solph.components.Source(label="pv", outputs={b_el: solph.Flow(fix=data["pv"],
                                                       investment=solph.Investment(ep_costs=epc_pv, maximum=None))}))

# Create a back supply
energysystem.add(
    solph.components.Source(label='grid', outputs={b_el: solph.Flow(investment=solph.Investment(ep_costs=epc_grid),
                                                         variable_costs=electricity_price)}))


# create simple sink object representing the electrical demand
energysystem.add(
    solph.components.Sink(label="demand", inputs={b_el: solph.Flow(fix=data["demand_el"], nominal_value=1)}))

##########################################################################
# Optimise the energy system and plot the results
##########################################################################

# Optimise the energy system and plot the results
logging.info("Optimise the energy system")

# initialise the operational model
model = solph.Model(energysystem)

# if tee_switch is true solver messages will be displayed
logging.info("Solve the optimization problem")
model.solve(solver=solver, solve_kwargs={"tee": solver_verbose})
logging.info("Store the energy system with the results.")

# add results to the energy system to make it possible to store them.
energysystem.results["main"] = solph.processing.results(model)
energysystem.results["meta"] = solph.processing.meta_results(model)

# store energy system with results
energysystem.dump(dpath=None, filename=None)

# Processing the results
results = energysystem.results["main"]

# get all variables of a specific component/bus

electricity_bus = solph.views.node(results, "electricity")

print('')
print('---------------------Energy flow (kWh)------------------------------------------')
flow_df = pd.DataFrame(electricity_bus['sequences'].sum())
column_names = ['Flow']
flow_df.columns = column_names
first_value = flow_df['Flow'][0]
flow_df['Share'] = ((flow_df['Flow'] / first_value) * 100).round(2)
print(flow_df)


print('')
print('-----------------------Energy Share---------------------------')
demand = (electricity_bus['sequences'][('electricity', 'demand'), 'flow'].sum())
generation = electricity_bus['sequences'][('grid', 'electricity'), 'flow'].sum() + \
             electricity_bus['sequences'][('pv', 'electricity'), 'flow'].sum()

PV_share = (electricity_bus['sequences'][('pv', 'electricity'), 'flow'].sum()) / demand * 100
Excess = (electricity_bus['sequences'][('electricity', 'excess_bel'), 'flow'].sum()) / demand * 100
Grid_share = (electricity_bus['sequences'][('grid', 'electricity'), 'flow'].sum()) / demand * 100

print('Grid share = ' + str(round(Grid_share, 2)) + ' %')
print('PV share = ' + str(round(PV_share, 2)) + ' %')

print('')
print('--------------------Technology Size (kW)-----------------------------------------')
print(str(electricity_bus['scalars']))

print('')
print('-----------------------Total costs---------------------------')
PV_cost = (electricity_bus['scalars'][('pv', 'electricity'), 'invest']) * epc_pv
Grid_cost = (electricity_bus['sequences'][('grid', 'electricity'), 'flow'].sum()) * electricity_price + \
            (electricity_bus['scalars'][('grid', 'electricity'), 'invest']) * epc_grid
Total_cost = PV_cost + Grid_cost
LCOE = Total_cost / generation

print('PV cost = ' + str(round(PV_cost, 2)) + ' €/a')
# print('Battery Storage cost = ' + str(round(Storage_cost, 2)) + ' €/a')
print('Grid cost = ' + str(round(Grid_cost, 2)) + ' €/a')
print('Total cost = ' + str(round(Total_cost, 2)) + ' €/a')
print('LCOE = ' + str(round(LCOE, 2)) + ' €/kWh')

df = pd.DataFrame(electricity_bus['sequences'])
df[('grid', 'electricity'), 'flow'] = np.absolute(df[('grid', 'electricity'), 'flow'])

df.index.name = 'Time'
production_el = df.drop((('electricity', 'demand'), 'flow'), axis=1)
demand_el = df[('electricity', 'demand'), 'flow']

# set a color dictionary
cdict = {(('electricity', 'excess_bel'), 'flow'): '#eeac7e',
         (('electricity', 'storage'), 'flow'): '#B576AD',
         (('pv', 'electricity'), 'flow'): '#ffde32',
         (('grid', 'electricity'), 'flow'): '#4ca7c3',
         ('electricity', 'demand_el'): '#000000',
         (('electricity', 'feed_in'), 'flow'): '#E04644',
         (('storage', 'electricity'), 'flow'): '#B7D968',
         (('gasboiler', 'heat'), 'flow'): '#545454',
         (('electricity', 'heating_rod'), 'flow'): '#DC143C',  # '#800080',
         (('heating_rod', 'heat'), 'flow'): '#DC143C',
         (('biomass_boiler', 'heat'), 'flow'): '#42c77a',
         (('heat_storage', 'heat'), 'flow'): 'orange',
         (('heat', 'heat_storage'), 'flow'): 'gold',
         (('CHP', 'electricity'), 'flow'): '#636f6b',
         (('CHP', 'heat'), 'flow'): '#636f6b',
         (('district_heating', 'heat'), 'flow'): '#5b5bae'}

# plot the time series (sequences) of a specific component/bus
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 5))
color = [cdict[column] for column in production_el.columns]
production_el.plot.area(ax=ax, color=color)
demand_el.plot(ax=ax, kind="line", linewidth=1, color='k')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Power in kW', fontsize=16)
plt.legend(loc="upper center", prop={"size": 10}, bbox_to_anchor=(0.5, 1.25), ncol=3)
fig.subplots_adjust(top=0.8)
plt.show()
