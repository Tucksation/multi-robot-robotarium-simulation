import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.controllers import *
import numpy as np
import matplotlib.pyplot as plt

#Number of robot (4)
N = 4

#starting postion
initial_conditions = np.array([
	[-1.0, -1.0, -1.0, -1.0],
	[-0.6, -0.2,  0.2,  0.6],
	[0, 0, 0, 0]
])

#Initialize Robotarium
r = robotarium.Robotarium(
	number_of_robots=N,
	initial_conditions=initial_conditions,
	sim_in_real_time=False,
	show_figure=True
)

#Goal position
goal_points = np.array([
	[1.2, 1.2, 1.2, 1.2],
	[-0.6, -0.2, 0.2, 0.6]
])

waypoints = [
	np.array([[-0.2, 0.5, 1.2], [-0.6, -0.6, -0.6]]), 
	np.array([[-0.2, 0.5, 1.2], [-0.2, -0.2, -0.2]]), 
	np.array([[-0.2, 0.5, 1.2], [0.2, 0.2, 0.2]]),
	np.array([[-0.2, 0.5, 1.2], [0.6, 0.6, 0.6]])
]

#start & goal markers
start_points = initial_conditions[:2]

plt.scatter(start_points[0], start_points[1],
            c='green', marker='o', s=100, label='Start')

plt.scatter(goal_points[0], goal_points[1],
            c='red', marker='*', s=150, label='Goal')

plt.legend()

#Random obstacles
num_obstacles = 6
obstacles = []

while len(obstacles) < num_obstacles:

	ox = np.random.uniform(-1.2, 1.2)
	oy = np.random.uniform(-0.8, 0.8)

	obstacle = [ox, oy]

	if not any(np.linalg.norm(np.array(obstacle) - start_points[:,i]) < 0.2 for i in range(N)) \
		and not any(np.linalg.norm(np.array(obstacle) - goal_points[:,i]) < 0.2 for i in range(N)):

        	obstacles.append(obstacle)

obstacles = np.array(obstacles)

plt.scatter(obstacles[:,0], obstacles[:,1],
            c='blue', marker='s', s=120, label='Obstacle')

plt.legend()

current_waypoint = [0, 0, 0, 0]
tolerance = 0.05

#Position controller
si_position_controller = create_si_position_controller()

#Run the saimulation
for _ in range(1000):

	#Get robot current poses
	x = r.get_poses()

	#Compute velocities
	dxu = np.zeros((2, N))

	for i in range(N):
		wp = waypoints[i][:, current_waypoint[i]].reshape(2,1)
		dxu[:, i:i+1] = si_position_controller(x[:2, i:i+1], wp)
		distance = np.linalg.norm(x[:2, i] - wp.flatten())

		if distance < tolerance and current_waypoint[i] < waypoints[i].shape[1] - 1:
			current_waypoint[i] += 1

	#Set the computed velocities for the robot
	r.set_velocities(np.arange(N), dxu)

	#Step simulation
	r.step()

#Close simulation
r.call_at_scripts_end()