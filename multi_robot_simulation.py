import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.controllers import *
from rps.utilities.barrier_certificates import (
    create_si_barrier_certificate_with_boundary,
)
import numpy as np
import matplotlib.pyplot as plt
import heapq

#==================================================
#	1. Creating a Grid
#==================================================
GRID_WIDTH = 16
GRID_HEIGHT = 10

grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

print("Occupancy Grid:")
print(grid)

#==================================================
#	2. World boundaries
#==================================================
X_MIN = -1.2
X_MAX = 1.2
Y_MIN = -0.8
Y_MAX = 0.8

CELL_WIDTH = (X_MAX - X_MIN)/GRID_WIDTH
CELL_HEIGHT = (Y_MAX - Y_MIN)/GRID_HEIGHT

def world_to_grid(x, y):
    column = int((x - X_MIN) / CELL_WIDTH)
    row = int((y - Y_MIN) / CELL_HEIGHT)

    # Keep coordinates inside the grid
    column = min(max(column, 0), GRID_WIDTH - 1)
    row = min(max(row, 0), GRID_HEIGHT - 1)

    return row, column

#==================================================
#  3. A* Path Planning
#==================================================

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):

    neighbors = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0)
    ]

    open_set = []
    heapq.heappush(open_set, (0, start))

    open_set_lookup = {start}

    came_from = {}

    g_score = {start: 0}

    f_score = {
        start: heuristic(start, goal)
    }

    while open_set:

        _, current = heapq.heappop(open_set)
        open_set_lookup.discard(current)

        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        for dx, dy in neighbors:

            neighbor = (
                current[0] + dx,
                current[1] + dy
            )

            if (
                not (
                    0 <= neighbor[0] < GRID_HEIGHT
                    and 0 <= neighbor[1] < GRID_WIDTH
                )
                or grid[neighbor[0], neighbor[1]] == 1
            ):
                continue

            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score.get(
                neighbor,
                float('inf')
            ):

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score

                f_score[neighbor] = (
                    tentative_g_score
                    + heuristic(neighbor, goal)
                )

                if neighbor not in open_set_lookup:
                    heapq.heappush(
                        open_set,
                        (f_score[neighbor], neighbor)
                    )

                    open_set_lookup.add(neighbor)

    return None



    
# Number of robots
NUM_ROBOTS = 4

#starting postion
initial_poses = np.array(
    [
        [-1.0, -1.0, -1.0, -1.0],
        [-0.6, -0.2, 0.2, 0.6],
        [0, 0, 0, 0],
    ]
)

for i in range(NUM_ROBOTS):
    x = initial_poses[0, i]
    y = initial_poses[1, i]
    
    cell = world_to_grid(x, y)

    print(f"Robot {i + 1}: ({x}, {y}) -> {cell}")
    

#Initialize Robotarium
r = robotarium.Robotarium(
	number_of_robots= NUM_ROBOTS,
	initial_conditions = initial_poses,
	sim_in_real_time=False,
	show_figure=True
)

#Goal position
goal_points = np.array([
	[1.2, 1.2, 1.2, 1.2],
	[-0.6, -0.2, 0.2, 0.6]
])

for i in range(NUM_ROBOTS):
    x = goal_points[0, i]
    y = goal_points[1, i]

    cell = world_to_grid(x, y)

    print(f"Robot {i + 1}: ({x}, {y}) -> {cell}")
    
"""
waypoints = [
	np.array([[-0.2, 0.5, 1.2], [-0.6, -0.6, -0.6]]), 
	np.array([[-0.2, 0.5, 1.2], [-0.2, -0.2, -0.2]]), 
	np.array([[-0.2, 0.5, 1.2], [0.2, 0.2, 0.2]]),
	np.array([[-0.2, 0.5, 1.2], [0.6, 0.6, 0.6]])
]
"""


#start & goal markers
start_points = initial_poses[:2]

plt.scatter(start_points[0], start_points[1],
            c='green', marker='o', s=100, label='Start')

plt.scatter(goal_points[0], goal_points[1],
            c='red', marker='*', s=150, label='Goal')

plt.legend()

protected_cells = set()
 
for i in range(NUM_ROBOTS):
    start_cell = world_to_grid(
		initial_poses[0, i],
		initial_poses[1,i]

	)
    
    goal_cell = world_to_grid(
		goal_points[0, i],
		goal_points[1, i]
	)
    protected_cells.add(start_cell)
    protected_cells.add(goal_cell)

print("Protected cells:")
print(protected_cells)

# Fastest a robot is commanded to move (m/s)
MAX_SPEED = 0.40

# Obstacle geometry (metres)
OBSTACLE_RADIUS = 0.1
ROBOT_RADIUS = 0.07
SAFETY_MARGIN = 0.03
# Closest a robot's centre may get to an obstacle's centre
SAFE_DISTANCE = OBSTACLE_RADIUS + ROBOT_RADIUS + SAFETY_MARGIN
# Robots start reacting to an obstacle once they are this close to its centre
INFLUENCE_DISTANCE = SAFE_DISTANCE + 0.2
# How fast a robot may approach an obstacle, per metre of distance left to SAFE_DISTANCE
APPROACH_SPEED_GAIN = 2.0

# ---------------------------------------------------------
# Physical obstacles
# ---------------------------------------------------------
NUM_OBSTACLES = 8

# Obstacles stay this far from every start, waypoint and goal so each one is reachable
MIN_CLEARANCE = SAFE_DISTANCE + 0.1
# Obstacles stay this far apart so a robot can always fit between two of them
MIN_OBSTACLE_SPACING = 2 * SAFE_DISTANCE + 0.1

# Every point the robots must be able to reach
points_to_keep_clear = np.hstack([
    initial_poses[:2],
    goal_points
]).T

obstacle_positions = []
placement_attempts = 0
MAX_PLACEMENT_ATTEMPTS = 10000

while len(obstacle_positions) < NUM_OBSTACLES:
    placement_attempts += 1

    if placement_attempts > MAX_PLACEMENT_ATTEMPTS:
        raise RuntimeError(
            "Could not place all obstacles. "
            "Reduce NUM_OBSTACLES or the clearances."
        )

    candidate_obstacle = np.array([
        np.random.uniform(
            X_MIN + SAFE_DISTANCE,
            X_MAX - SAFE_DISTANCE
        ),
        np.random.uniform(
            Y_MIN + SAFE_DISTANCE,
            Y_MAX - SAFE_DISTANCE
        )
    ])

    too_close_to_path_points = np.any(
        np.linalg.norm(
            points_to_keep_clear - candidate_obstacle, axis=1
        ) < MIN_CLEARANCE
    )
    too_close_to_other_obstacles = any(
        np.linalg.norm(candidate_obstacle - obstacle) < MIN_OBSTACLE_SPACING
        for obstacle in obstacle_positions
    )

    if (
        not too_close_to_path_points and not too_close_to_other_obstacles
        ):
        obstacle_positions.append(candidate_obstacle)

obstacle_positions = np.array(obstacle_positions)
    
    
# ---------------------------------------------------------
# Convert physical obstacles into A* grid obstacles
# ---------------------------------------------------------

grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

obstacles = []

for row in range(GRID_HEIGHT):
    for column in range(GRID_WIDTH):
        cell_x_min = X_MIN + column * CELL_WIDTH
        cell_x_max = cell_x_min + CELL_WIDTH
        
        cell_y_min = Y_MIN + row * CELL_HEIGHT
        cell_y_max = cell_y_min + CELL_HEIGHT
        
        for obstacle in obstacle_positions:
            
            obstacle_x, obstacle_y = obstacle
            
            closest_x = np.clip(
                obstacle_x, 
                cell_x_min, 
                cell_x_max
                )
            closest_y = np.clip(
                obstacle_y, 
                cell_y_min, 
                cell_y_max
                )
            
            distance_to_cell = np.linalg.norm(
                np.array([
                    closest_x - obstacle_x, 
                    closest_y - obstacle_y
                ])
            )
            
            if distance_to_cell < OBSTACLE_RADIUS:
                grid[row, column] = 1
                obstacles.append((row, column))
                break
            
print("Physical obstacle positions:")
print(obstacle_positions)

print("\nA* blocked cells:")
print(obstacles)

print("\nGrid with obstacles:")
print(grid)
"""
obstacle_positions = []
placement_attempts = 0

obstacles = []

while len(obstacles) < NUM_OBSTACLES:
    placement_attempts += 1
    row = np.random.randint(0, GRID_HEIGHT)
    column = np.random.randint(0, GRID_WIDTH)

    cell = (row, column)

    if grid[row, column] == 0 and cell not in protected_cells:
        grid[row, column] = 1
        obstacles.append(cell)

    if placement_attempts > MAX_PLACEMENT_ATTEMPTS:
        raise RuntimeError(
            "Could not place all obstacles; reduce NUM_OBSTACLES or the clearances."
        )

    candidate_obstacle = np.array(
        [np.random.uniform(-1.2, 1.2), np.random.uniform(-0.8, 0.8)]
    )

    too_close_to_path_points = np.any(
        np.linalg.norm(points_to_keep_clear - candidate_obstacle, axis=1)
        < MIN_CLEARANCE
    )
    too_close_to_other_obstacles = any(
        np.linalg.norm(candidate_obstacle - obstacle) < MIN_OBSTACLE_SPACING
        for obstacle in obstacle_positions
    )

    if not too_close_to_path_points and not too_close_to_other_obstacles:
        obstacle_positions.append(candidate_obstacle)
  
obstacle_positions = np.array(obstacle_positions)

print("Obstacle cells:")
print(obstacles)


print("\nGrid with obstacles:")
print(grid)
"""
print("\nA* Paths:")

robot_waypoints = []
for i in range(NUM_ROBOTS):

    start = world_to_grid(
        initial_poses[0, i],
        initial_poses[1, i]
    )

    goal = world_to_grid(
        goal_points[0, i],
        goal_points[1, i]
    )

    path = astar(grid, start, goal)
    if path is None:
        raise RuntimeError(f"No valid path found for robot {i + 1}")

    world_path = np.array([
        [X_MIN + (col + 0.5) * CELL_WIDTH, Y_MIN + (row + 0.5) * CELL_HEIGHT]
        for row, col in path
    ]).T
    robot_waypoints.append(world_path)
    
    
    
    print("Number of waypoints:", world_path.shape[1])

    print(f"\nRobot {i + 1}")
    print("Start:", start)
    print("Goal:", goal)
    print("Path:", path)
    
#==================================================
# Visualize Grid, Obstacles and A* Paths
#==================================================

fig, ax = plt.subplots()

# Draw grid
for x in np.arange(X_MIN, X_MAX + CELL_WIDTH, CELL_WIDTH):
    ax.axvline(x, linewidth=0.5)

for y in np.arange(Y_MIN, Y_MAX + CELL_HEIGHT, CELL_HEIGHT):
    ax.axhline(y, linewidth=0.5)

# Draw A* paths
for robot in range(NUM_ROBOTS):

    path = robot_waypoints[robot]

    ax.plot(
        path[0],
        path[1],
        marker='o',
        linewidth=2,
        label=f'Robot {robot + 1} A* path'
    )

# Draw grid obstacles
for row, column in obstacles:

    x = X_MIN + column * CELL_WIDTH
    y = Y_MIN + row * CELL_HEIGHT

    rectangle = plt.Rectangle(
        (x, y),
        CELL_WIDTH,
        CELL_HEIGHT,
        alpha=0.3
    )

    ax.add_patch(rectangle)

# Draw physical obstacles
for obstacle in obstacle_positions:

    circle = plt.Circle(
        obstacle,
        OBSTACLE_RADIUS,
        alpha=0.7
    )

    ax.add_patch(circle)

# Draw starts
ax.scatter(
    start_points[0],
    start_points[1],
    marker='o',
    s=100,
    label='Start'
)

# Draw goals
ax.scatter(
    goal_points[0],
    goal_points[1],
    marker='*',
    s=150,
    label='Goal'
)

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)

ax.set_aspect('equal')

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_title("A* Grid, Obstacles and Planned Paths")

ax.legend()

plt.show()

##
# Draw obstacles at their true size so it is visible that robots keep clear
#for obstacle_number, obstacle in enumerate(obstacle_positions):
    #plt.gca().add_patch(
        #plt.Circle(
           # obstacle,
            #OBSTACLE_RADIUS,
            #color="blue",
           # alpha=0.6,
           # label="Obstacle" if obstacle_number == 0 else None,
        #)
    #)

#plt.legend()


##obstacles = np.array(obstacles)

##plt.scatter(obstacles[:,0], obstacles[:,1],
       ##     c='blue', marker='s', s=120, label='Obstacle')

#plt.legend()
def avoid_obstacles(robot_position, desired_velocity):
    """
    Adjust a robot's desired (x, y) velocity so it never gets closer than
    SAFE_DISTANCE to any obstacle.

    Near an obstacle, the part of the velocity heading into it is limited, and
    the speed removed is redirected sideways so the robot slides around the
    obstacle instead of stopping in front of it.
    """
    safe_velocity = desired_velocity.copy()

    for obstacle in obstacle_positions:
        offset_from_obstacle = robot_position - obstacle
        distance_to_obstacle = np.linalg.norm(offset_from_obstacle)

        if distance_to_obstacle > INFLUENCE_DISTANCE:
            continue

        away_direction = offset_from_obstacle / distance_to_obstacle
        approach_speed = -np.dot(safe_velocity, away_direction)
        allowed_approach_speed = APPROACH_SPEED_GAIN * (
            distance_to_obstacle - SAFE_DISTANCE
        )
        excess_approach_speed = approach_speed - allowed_approach_speed

        if excess_approach_speed <= 0:
            continue

        # Slide around the obstacle on whichever side the robot is already heading
        sideways_direction = np.array([-away_direction[1], away_direction[0]])
        if np.dot(safe_velocity, sideways_direction) < 0:
            sideways_direction = -sideways_direction

        safe_velocity += excess_approach_speed * (away_direction + sideways_direction)

    speed = np.linalg.norm(safe_velocity)
    if speed > MAX_SPEED:
        safe_velocity *= MAX_SPEED / speed

    return safe_velocity

"""
current_waypoint = [0, 0, 0, 0]
tolerance = 0.05

#Position controller
si_position_controller = create_si_position_controller()

#Run the saimulation
for _ in range(1000):

	#Get robot current poses
	x = r.get_poses()

	#Compute velocities
	dxu = np.zeros((2, NUM_ROBOTS))

	for i in range(NUM_ROBOTS):
		wp = waypoints[i][:, current_waypoint[i]].reshape(2,1)
		dxu[:, i:i+1] = si_position_controller(x[:2, i:i+1], wp)
		distance = np.linalg.norm(x[:2, i] - wp.flatten())

		if distance < tolerance and current_waypoint[i] < waypoints[i].shape[1] - 1:
			current_waypoint[i] += 1

	#Set the computed velocities for the robot
	r.set_velocities(np.arange(NUM_ROBOTS), dxu)

	#Step simulation
	r.step()

#Close simulation
r.call_at_scripts_end()
"""

# Index of the waypoint each robot is currently heading towards
active_waypoint_index = [0] * NUM_ROBOTS
WAYPOINT_REACHED_TOLERANCE = 0.05

# Position controller: drives each robot straight towards its waypoint
si_position_controller = create_si_position_controller(
    velocity_magnitude_limit=MAX_SPEED
)

# Keeps robots from colliding with each other and with the arena walls
robot_collision_avoidance = create_si_barrier_certificate_with_boundary(
    magnitude_limit=MAX_SPEED
)

# Converts (x, y) velocities into the (forward speed, turn rate) commands the robots use
si_to_unicycle = create_si_to_uni_dynamics()

# Run the simulation
NUM_ITERATIONS = 1000

for _ in range(NUM_ITERATIONS):

    # Get robots' current poses
    current_poses = r.get_poses()

    # Desired (x, y) velocity of each robot
    velocities = np.zeros((2, NUM_ROBOTS))

    for robot in range(NUM_ROBOTS):
        target_waypoint = robot_waypoints[robot][
            :, active_waypoint_index[robot]
        ].reshape(2, 1)
        robot_position = current_poses[:2, robot : robot + 1]

        velocities[:, robot : robot + 1] = si_position_controller(
            robot_position, target_waypoint
        )
        distance_to_waypoint = np.linalg.norm(
            current_poses[:2, robot] - target_waypoint.flatten()
        )

        is_last_waypoint = (
            active_waypoint_index[robot] >= robot_waypoints[robot].shape[1] - 1
        )
        if distance_to_waypoint < WAYPOINT_REACHED_TOLERANCE and not is_last_waypoint:
            active_waypoint_index[robot] += 1

    velocities = robot_collision_avoidance(velocities, current_poses)

    for robot in range(NUM_ROBOTS):
        velocities[:, robot] = avoid_obstacles(
            current_poses[:2, robot], velocities[:, robot]
        )

    # Set the computed velocities for the robots
    r.set_velocities(
        np.arange(NUM_ROBOTS), si_to_unicycle(velocities, current_poses)
    )

    # Step simulation
    r.step()

# Print the simulation summary (run time, collisions, boundary violations)
r.debug()

 
