
from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pandas as pd
import math
import matplotlib.pyplot as plt
import random as rd
##############
# Excel Data #
##############
def excel_data():
    """Reads excel file"""
    tmp=pd.read_csv('Demand_T0.csv',skiprows = 1,header=None)
    Num_vehicles=int(tmp.loc[0,11])
    Capacity=(tmp.loc[0:,9])
    Location_X=(tmp.loc[0:,2])
    Location_Y=(tmp.loc[0:,3])
    Vehicle_X=(tmp.loc[0:,7])
    Vehicle_Y=(tmp.loc[0:,8])
    Demand=(tmp.loc[0:,1])
    Time_current=(tmp.loc[0,10])
    Location_coordinates=[]
    Vehicle_coordinates=[]
    Vehicle_capacity=[]
    Client_demand=[]
    Num_clients=int(tmp.loc[0,4])
    start_nodes = [0, 1, 2, 3]
    end_nodes = []
    Distance_travelled=[0,0,0,0]
    i=0
    j=0
    print("\n***Program start***\n")
    print('\nAt Time = {0}'.format(Time_current))
    while(j<Num_vehicles):
        Client_demand.append(0)
        j=j+1
    j=0
    while(j<Num_vehicles):
        coordinates=[]
        coordinates.append(int(Vehicle_X[j]))
        coordinates.append(int(Vehicle_Y[j]))
        Vehicle_capacity.append(int(Capacity[i]))
        j=j+1
        Vehicle_coordinates.append(coordinates)
    i=0
    while(i<Num_clients):
        coordinates=[]
        if(Demand[i]!=0):
            coordinates.append(Location_X[i])
            coordinates.append(Location_Y[i])
            Client_demand.append(Demand[i])
            Location_coordinates.append(coordinates)
        i=i+1           
    i=0
    Location_coordinates=Vehicle_coordinates+Location_coordinates
    for i in range(0,Num_vehicles):
        end_nodes.append(len(Location_coordinates)+i)
        Client_demand.append(0)
    Location_coordinates=Location_coordinates+Vehicle_coordinates
    
    [Time, Distance, Load, Routes]=Routing_logic(Location_coordinates,Time_current,Client_demand,Vehicle_capacity,Num_vehicles,start_nodes,end_nodes,Distance_travelled)
    
    return Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity
    print( Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity)
#################
#New Excel Data #
#################
def New_data(Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity, Name):
    """Reads new excel file"""
    tmp=pd.read_csv(Name,skiprows = 1,header=None,skip_blank_lines=True)
    Num_vehicles=int(tmp.loc[0,11])
    #Capacity=(tmp.loc[0:,9])
    Location_X=(tmp.loc[0:,2])
    Location_Y=(tmp.loc[0:,3])
    Depot_X=(tmp.loc[0:,7])
    Depot_Y=(tmp.loc[0:,8])
    New_demand=(tmp.loc[0:,1])
    Time_current=(tmp.loc[0,10])
    Depot_location=[]
    Vehicle_coordinates=[]
    Vehicle_capacity=Vehicle_capacity#[25, 25, 25, 25]
    New_client_demand=[]
    New_client_location=[]
    Current_load=[]
    Distance_travelled=[]#Time_current*1#vehiclespeed
    Num_clients=int(tmp.loc[0,4])
    start_nodes = [0, 1, 2, 3]
    end_nodes = []
    print('\n=========================================================================================================')
    print('\nAt Time = {0}\n'.format(Time_current))

    print("Distance: \n",Distance)
    print("Routes: \n",Routes)
    print("Load: ",Load)
    print(Distance_travelled)
    
    i=0
    j=0
    '''Find curent Distance travelled by each vehicle'''
    while(i<len(Distance)):
        if(max(Distance[i])<=Time_current*1):
            Distance_travelled.append(max(Distance[i]))
            i+=1
        else:
            Distance_travelled.append(Time_current*1)
            i+=1
    i=0
    '''Write Depot locations to list Depot_location'''
    for j in range(0,Num_vehicles):
        coordinates=[]
        coordinates.append(int(Depot_X[j]))
        coordinates.append(int(Depot_Y[j]))
        Depot_location.append(coordinates)

    '''Write New client location and demand to lists New_client_location,New_client_demand'''
    for i in range(0, Num_clients):
        coordinates=[]
        coordinates.append(Location_X[i])
        coordinates.append(Location_Y[i])
        New_client_demand.append(int(New_demand[i]))
        New_client_location.append(coordinates)
    
    j=0 
    i=0           
    '''Finds the current location of vehicle based on distance travelled'''
    while(j<len(Routes[i])):
        if(Distance_travelled[i]<Distance[i][j]):
            Current_load.append(Load[i][j-1])
            v_c=vehicle_coordinates(Location_coordinates[Routes[i][j-1]],Location_coordinates[Routes[i][j]],Distance_travelled[i]-Distance[i][j-1])
            Vehicle_coordinates.append(v_c)
            i=i+1
            j=0
            if(i==Num_vehicles):
                break
        elif(max(Distance[i])<=Distance_travelled[i]):
            Current_load.append(max(Load[i]))
            m=max(Distance[i])
            Index=Distance[i].index(m)
            v_c=Location_coordinates[Routes[i][Index]]
            Vehicle_coordinates.append(v_c)
            i=i+1
            j=0
            if(i==Num_vehicles):
                break
        else:
            j=j+1    
    i=0           
    j=0

    '''Makes the Client_demand 0 for all the locations covered till Time_current'''
    while(j<len(Distance[i])):

        if(Distance_travelled[i]<Distance[i][j]):
            k=0
            while(k<j):
                Client_demand[Routes[i][k]]=0
                k+=1
            i=i+1
            j=0
            if(i==Num_vehicles):
                break
        elif(max(Distance[i])<=Distance_travelled[i]):
            m=max(Distance[i])
            Index=Distance[i].index(m)
            k=0
            while(k<=Index):
                Client_demand[Routes[i][k]]=0
                k+=1
            i=i+1
            j=0
            if(i==Num_vehicles):
                break
        else:
            j=j+1
    i=0
    j=0

    '''Delets the last 4 location coordinates and client demand to update new data'''
    j=len(Location_coordinates)          
    del Location_coordinates [j-4:]
    del Client_demand [j-4:]

    j=0
    
    '''Checks for update in client demand and updates lists Client_demand,Location_coordinates'''
    for j in range(0,Num_clients):
        for i in range (0, len(Location_coordinates)):
            if(New_client_location[j]==Location_coordinates[i]):
                Client_demand[i]=New_client_demand[j]
                break

        Location_coordinates.append(New_client_location[j])
        Client_demand.append(New_client_demand[j])
           
    j=0

    '''Updates the new vehicle positions and their current load'''
    while(j<Num_vehicles):
        Client_demand[j]=Current_load[j]
        Location_coordinates[j]=Vehicle_coordinates[j]
        j=j+1
        
    j=0
    '''Delets the locations and demands with 0 demand'''    
    i=Num_vehicles
    while(i<len(Client_demand)):
        if(Client_demand[i]==0):
            del Location_coordinates[i]
            del Client_demand[i]
        else:
            i+=1
    j=0
    '''Adds final depot destinations to Location_coordinates'''
    Location_coordinates=Location_coordinates+Depot_location
    
    '''Adds demand for final depot destinations as 0'''
    for j in range (0,Num_vehicles):
        Client_demand.append(0)

    '''Sets the final destination nodes'''    
    for i in range(0,Num_vehicles):
        end_nodes.append(len(Location_coordinates)-Num_vehicles+i)
    [Time, Distance, Load, Routes]=Routing_logic(Location_coordinates,Time_current,Client_demand,Vehicle_capacity,Num_vehicles,start_nodes,end_nodes,Distance_travelled)
    
    return Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity
    
########################################
# calculate curent vehicle coordinates #
########################################
def vehicle_coordinates(c1,c2,s):
    c3=[]
    if(c1==c2):
        c3=c1
    else:
        x=c2[0]-c1[0]
        y=c2[1]-c1[1]
        xy=math.sqrt((x*x)+(y*y))
        x1=s*(x/xy)
        c3.append(round((x1+c1[0]),2))
        y1=s*(y/xy)
        c3.append(round((y1+c1[1]),2))    
    return c3

###########################
# Problem Data Definition #
###########################
def create_data_model(Location_coordinates,Client_demand,Vehicle_capacity,Num_vehicles):
  """Creates the data for the example."""
  data = {}
  locations = Location_coordinates
  num_locations = len(locations)
  # Array of distances between locations.
  dist_matrix = {}
  for from_node in range(num_locations):
    dist_matrix[from_node] = {}

    for to_node in range(num_locations):
      dist_matrix[from_node][to_node] = (
        manhattan_distance(
          locations[from_node],
          locations[to_node]))
  _distances = dist_matrix

  demands = Client_demand#[0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
  capacities = Vehicle_capacity#[15, 15, 15, 15]
  data["distances"] = _distances
  data["num_locations"] = len(_distances)
  data["num_vehicles"] = Num_vehicles
  data["depot"] = 0
  data["demands"] = demands
  data["vehicle_capacities"] = capacities
  return data

#######################
# Problem Constraints #
#######################
def manhattan_distance(position_1, position_2):
  """Computes the Euclidean distance between two points"""
  return round((math.sqrt((position_2[0] - position_1[0])**2 + (position_2[1] - position_1[1])**2)),1)

def create_distance_callback(data):
  """Creates callback to return distance between points."""
  distances = data["distances"]

  def distance_callback(from_node, to_node):
    """Returns the manhattan distance between the two nodes"""
    return distances[from_node][to_node]
  return distance_callback

def create_demand_callback(data):
    """Creates callback to get demands at each location."""
    def demand_callback(from_node, to_node):
        return data["demands"][from_node]
    return demand_callback

def add_capacity_constraints(routing, data, demand_callback):
    """Adds capacity constraint"""
    capacity = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0, # null capacity slack
        data["vehicle_capacities"], # vehicle maximum capacities
        True, # start cumul to zero
        capacity)

def travel_time(from_node, to_node,data):
    """Gets the travel times between two locations."""
    travel_time = data["distances"][from_node][to_node] / 1#data["vehicle_speed"]
    return travel_time

####################
# Get Routes Array #
####################
def get_routes_array(assignment, num_vehicles, routing,end_nodes):
  # Get the routes for an assignent and return as a list of lists.
  routes = []
  for route_nbr in range(num_vehicles):
    node = routing.Start(route_nbr)
    route = []

    while not routing.IsEnd(node):
      index = routing.NodeToIndex(node)
      route.append(index)
      node = assignment.Value(routing.NextVar(node))
    routes.append(route)
  #Routes=routes  
  for i in range(0,len(routes)):
      routes[i].append(end_nodes[i])
      
  return routes

###########
# Printer #
###########
def print_solution(data, routing, assignment,Location_coordinates,Time_current,Num_vehicles,Distance_travelled):
    """Print routes on console."""
    total_dist = 0#Num_vehicles*Time_current*1#0
    Time = []
    Distance= []
    Load= []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id+1)
        route_dist = Distance_travelled[vehicle_id]
        route_load = 0
        route_time = Time_current
        Time_matrix=[Time_current]
        Distance_between=[Distance_travelled[vehicle_id]]
        Load_total=[]
        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            node_cod=Location_coordinates[node_index]
            route_dist += manhattan_distance(Location_coordinates[node_index],Location_coordinates[next_node_index])
            route_time += travel_time(node_index, next_node_index,data)           
            route_load += data["demands"][node_index]

            plan_output += ' {0} Load({1}) -> '.format(node_cod, route_load)
            index = assignment.Value(routing.NextVar(index))
            Time_matrix.append(route_time)
            Distance_between.append(route_dist)
            Load_total.append(route_load)
                   
        Load.append(Load_total)
        Distance.append(Distance_between)
        Time.append(Time_matrix)
        node_index = routing.IndexToNode(index)
        node_cod=Location_coordinates[node_index]
        total_dist += route_dist 
        plan_output += ' {0} Load({1})\n'.format(node_cod, route_load)
        plan_output += 'Distance of the route: {0}m\n'.format(round(route_dist,2))
        plan_output += 'Load of the route: {0}\n'.format(route_load)
        print("\n",plan_output)
    print('Total Distance of all routes: {0}m'.format(round(total_dist,2)))
    for i in range(0,len(Load)):
        Load[i].append(Load[i][len(Load[i])-1])
        #print(Load)
    return Time, Distance, Load

#################
# Routing_logic #
#################
def Routing_logic(Location_coordinates,Time_current,Client_demand,Vehicle_capacity,Num_vehicles,start_nodes,end_nodes,Distance_travelled):
  """Entry point of the program"""
  # Instantiate the data problem.
  data = create_data_model(Location_coordinates,Client_demand,Vehicle_capacity,Num_vehicles)
  # Create data.
  num_locations = len(Location_coordinates)
  num_vehicles = Num_vehicles
  # Create Routing Model
  start_locations = start_nodes
  end_locations = end_nodes
  routing = pywrapcp.RoutingModel(num_locations, num_vehicles, start_locations, end_locations)
  # Define weight of each edge
  distance_callback = create_distance_callback(data)
  routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
  # Add Capacity constraint
  demand_callback = create_demand_callback(data)
  add_capacity_constraints(routing, data, demand_callback)
  # Setting first solution heuristic.
  '''
  PATH_CHEAPEST_ARC - Starting from a route "start" node, connect it to the node which produces 
  the cheapest route segment, then extend the route by iterating on the last node added to the route.
  SAVINGS - Savings algorithm (Clarke & Wright).
  GLOBAL_CHEAPEST_ARC - Iteratively connect two nodes which produce the cheapest route segment.
  CHRISTOFIDES - Works by extending a route until no nodes can be inserted on it.
  '''
  search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
  search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
  # Setting Local search options
  '''
  GUIDED_LOCAL_SEARCH - Uses guided local search to escape local minima
  SIMULATED_ANNEALING - Uses simulated annealing to escape local minima
  TABU_SEARCH - Uses tabu search to escape local minima
  GREEDY_DESCENT - Search neighbors until a local minimum is reached
  '''
  search_parameters.time_limit_ms = 50000
  search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)

  assignment = routing.SolveWithParameters(search_parameters)
  Routes=get_routes_array(assignment, num_vehicles, routing,end_nodes)
  if assignment:
    [Time, Distance, Load]=print_solution(data, routing, assignment,Location_coordinates,Time_current,Num_vehicles,Distance_travelled)
  return Time, Distance, Load, Routes
#################
# Ploting 1 #
#################
def plot(Location_coordinates,Routes,Client_demand,a):
    i=0
    plt.scatter([i[0] for i in Location_coordinates],[i[1] for i in Location_coordinates],c='r',lw=4.1)
    plt.scatter([i[0] for i in Location_coordinates if i == [30,40] or i == [20,25]],[i[1] for i in Location_coordinates if i == [30,40] or i == [20,25]],c='b',marker='s',lw=4.1)
    for i in Routes:
        for j in range(len(i)-1):
            p0 = Location_coordinates[i[j]]
            p1 = Location_coordinates[i[j+1]] 
            #print(p0[0],p0[1])
            plt.plot([p0[0],p1[0]],[p0[1],p1[1]],c='r',alpha=1)
            plt.plot([p0[0],p1[0]],[p0[1],p1[1]],c='b',alpha=1)
            plt.plot([p0[0],p1[0]],[p0[1],p1[1]],c='g',alpha=1)
            plt.plot([p0[0],p1[0]],[p0[1],p1[1]],c='y',alpha=1)
    for i in range(4,len(Location_coordinates)-4):
        plt.annotate(str(Location_coordinates[i])+"="+ str(Client_demand[i]),Location_coordinates[i])
    plt.grid(True)
    plt.ylabel("Y",fontsize=14)

    plt.title('Vehicle Routing Problem at t='+ str(a),fontsize=14)
    plt.show()

#########
# Main  #
#########
def main():
    """Entry point of the program"""
    [Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity]=excel_data()
    plt.xlabel("X",fontsize=14)
    plot(Location_coordinates,Routes,Client_demand,a=0)
    
    Name='Demand_T13.csv'
    [Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity]=New_data(Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity, Name)
    plt.text(26.5, 8, r'O ', fontsize=30)
    plt.xlabel("X\n At time=13, a new demand at point (28,10) appears.",fontsize=14)
    plot(Location_coordinates,Routes,Client_demand,a=13)

    Name='Demand_T31.csv'
    [Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity]=New_data(Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity, Name)
    plt.text(4, 37, r'O ', fontsize=30)
    plt.xlabel("X\n At time=31, a new demand at point (5,38) appears.",fontsize=14)
    plot(Location_coordinates,Routes,Client_demand,a=31)
    
    Name='Demand_T45.csv'
    plt.text(18, 44, r'O ', fontsize=30)
    [Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity]=New_data(Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity, Name)
    plt.xlabel("X\n At time=45, a new demand at point (19,45) appears.",fontsize=14)
    plot(Location_coordinates,Routes,Client_demand,a=45)

    Name='Demand_T66.csv'
    [Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity]=New_data(Time, Distance, Load, Routes, Location_coordinates, Client_demand, Vehicle_capacity, Name)
    plt.xlabel("X\n At time=66, a new demand at point (35,35) appears.",fontsize=14)
    plt.text(34, 34, r'O ', fontsize=30)
    plot(Location_coordinates,Routes,Client_demand,a=66)
    print("\n*** END ***\n")
    
if __name__ == '__main__':
  main()

