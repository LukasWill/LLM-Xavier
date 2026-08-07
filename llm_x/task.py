class Task:
    task_name: str = None

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        self.task_description = task_description
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.transition_dynamics = transition_dynamics
        self.init_state = init_state
        self.termination = termination

    @property
    def data(self):
        return {
            "task_name": self.__class__.task_name,
            "task_description": self.task_description,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "reward_space": self.reward_space,
            "transition_dynamics": self.transition_dynamics,
            "init_state": self.init_state,
            "termination": self.termination,
        }


class MountainCarTask(Task):

    task_name = "MountainCar-v0"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )
        self.task_description = """
        The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically at the bottom of a sinusoidal valley, with the only possible actions being the accelerations that can be applied to the car in either direction. The goal of the MDP is to strategically accelerate the car to reach the goal state on top of the right hill.
        """

        self.observation_space = """
        The observation is a ndarray with shape (2,) where the elements correspond to the following:
        position of the car along the x-axis (range from -1.2 to 0.6), velocity of the car (range from -0.07 to 0.07)
        """

        self.action_space = """
        There are 3 discrete deterministic actions,
        0: Accelerate to the left
        1: Do not accelerate
        2: Accelerate to the right
        """

        self.reward_space = """
        The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is penalised with a reward of -1 for each timestep.
        """

        self.transition_dynamics = """
        Given an action, the mountain car follows the following transition dynamics,
        velocity_t+1 = velocity_t + (action - 1) * force - cos(3 * position_t) * gravity
        position_t+1 = position_t + velocity_t+1
        where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
        """

        self.init_state = """
        The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.
        """

        self.termination = """
        The episode ends if the position of the car is greater than or equal to 0.5 (the goal position on top of the right hill).
        """

# class LunarLanderTask(Task):

#     task_name = "LunarLander-v2"

#     def __init__(
#         self,
#         task_description=None,
#         observation_space=None,
#         action_space=None,
#         reward_space=None,
#         transition_dynamics=None,
#         init_state=None,
#         termination=None,
#     ) -> None:
#         super().__init__(
#             task_description,
#             observation_space,
#             action_space,
#             reward_space,
#             transition_dynamics,
#             init_state,
#             termination,
#         )

#         self.task_description = """
#         This environment is a classic rocket trajectory optimization problem. According to Pontryagin's maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
#         """

#         self.observation_space = """
#         The state s is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground (of value 1) or not (of value 0).
#         Attributes:
#             s[0] is the horizontal coordinate
#             s[1] is the vertical coordinate
#             s[2] is the horizontal speed
#             s[3] is the vertical speed
#             s[4] is the angle
#             s[5] is the angular speed
#             s[6] 1 if first leg has contact, else 0
#             s[7] 1 if second leg has contact, else 0
#         """

#         self.action_space = """
#         There are four discrete actions available,
#         0: do nothing
#         1: fire left orientation engine (i.e., rotating the lander rightward)
#         2: fire main engine (i.e., slowing the downward movement)
#         3: fire right orientation engine (i.e, rotating the lander leftward)
#         """

#         self.reward_space = """
#         After every step, a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

#         For each step, the reward:

#             a. is increased/decreased the closer/further the lander is to the landing pad (measured by euclidean distance)
#             b. is increased/decreased the slower/faster the lander is moving
#             c. is decreased the more the lander is tilted (angle not horizontal)
#             d. is increased by 10 points for each leg that is in contact with the ground
#             e. is decreased by 0.03 points each frame a side engine is firing
#             f. is decreased by 0.3 points each frame the main engine is firing

#         The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively.

#         ```python
#         # A snippet of source code for computing the reward for each step
#         reward = 0
#         shaping = (
#             -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
#             - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
#             - 100 * abs(state[4])
#             + 10 * state[6]
#             + 10 * state[7]
#         ) # And ten points for legs contact, the idea is if you lose contact again after landing, you get negative reward
#         if self.prev_shaping is not None:
#             reward = shaping - self.prev_shaping
#         self.prev_shaping = shaping

#         reward -= (m_power * 0.30) # less fuel spent is better, about -30 for heuristic landing
#         reward -= s_power * 0.03

#         terminated = False
#         if self.game_over or abs(state[0]) >= 1.0:
#             terminated = True
#             reward = -100
#         if not self.lander.awake:
#             terminated = True
#             reward = +100
#         ```
#         """

#         self.transition_dynamics = """
#         ```python
#         # A snippet of source code for computing the next state after taking an action, e.g., action = 0, 1, 2 or 3
#         FPS = 50
#         SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

#         MAIN_ENGINE_POWER = 13.0
#         SIDE_ENGINE_POWER = 0.6

#         INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

#         LEG_DOWN = 18

#         SIDE_ENGINE_HEIGHT = 14
#         SIDE_ENGINE_AWAY = 12
#         MAIN_ENGINE_Y_LOCATION = (
#             4  # The Y location of the main engine on the body of the Lander.
#         )

#         VIEWPORT_W = 600
#         VIEWPORT_H = 400

#         H = VIEWPORT_H / SCALE

#         self.helipad_y = H / 4

#         # Tip is a the (X and Y) components of the rotation of the lander.
#         tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

#         # Side is the (-Y and X) components of the rotation of the lander.
#         side = (-tip[1], tip[0])

#         # Generate two random numbers between -1/SCALE and 1/SCALE.
#         dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

#         m_power = 0.0
#         if action == 2:
#             # Main engine
#             m_power = 1.0

#             # 4 is move a bit downwards, +-2 for randomness
#             # The components of the impulse to be applied by the main engine.
#             ox = (
#                 tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
#                 + side[0] * dispersion[1]
#             )
#             oy = (
#                 -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
#                 - side[1] * dispersion[1]
#             )

#             impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
#             self.lander.ApplyLinearImpulse(
#                 (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
#                 impulse_pos,
#                 True,
#             )

#         s_power = 0.0
#         if action in [1, 3]:
#             # Orientation/Side engines
#             # action = 1 is left, action = 3 is right
#             direction = action - 2
#             s_power = 1.0

#             # The components of the impulse to be applied by the side engines.
#             ox = tip[0] * dispersion[0] + side[0] * (
#                 3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
#             )
#             oy = -tip[1] * dispersion[0] - side[1] * (
#                 3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
#             )

#             # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
#             # However, SIDE_ENGINE_HEIGHT is defined as 14
#             # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
#             # This in turn results in an orientation depentant torque being applied to the lander.
#             impulse_pos = (
#                 self.lander.position[0] + ox - tip[0] * 17 / SCALE,
#                 self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
#             )
#             self.lander.ApplyLinearImpulse(
#                 (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
#                 impulse_pos,
#                 True,
#             )

#         self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

#         pos = self.lander.position
#         vel = self.lander.linearVelocity

#         state = [
#             (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
#             (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
#             vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
#             vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
#             self.lander.angle,
#             20.0 * self.lander.angularVelocity / FPS,
#             1.0 if self.legs[0].ground_contact else 0.0,
#             1.0 if self.legs[1].ground_contact else 0.0,
#         ]
#         assert len(state) == 8
#         ```
#         """

#         self.init_state = """
#         The lander starts at the top centre of the viewport with a random initial force applied to its centre of mass.
#         """

#         self.termination = """
#         The episode finishes if

#         a. the lander crashes (the lander's body gets in contact with the moon);

#         b. the lander gets outside of the viewport (x coordinate is greater than 1);

#         c. the lander is not awake.

#         From the Box2D docs, a body which is not awake is a body which doesn't move and doesn't collide with any other body:
#         Note, when Box2D determines that a body (or group of bodies) has come to rest, the body enters a sleep state which has very little CPU overhead. If a body is awake and collides with a sleeping body, then the sleeping body wakes up. Bodies will also wake up if a joint or contact attached to them is destroyed.

#         ```python
#         # A snippet of source code for checking termination
#         terminated = False
#         if self.game_over or abs(state[0]) >= 1.0:
#             terminated = True
#             reward = -100
#         if not self.lander.awake:
#             terminated = True
#             reward = +100
#         ```

#         """

class LunarLanderTask(Task):

    task_name = "LunarLander-v2"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        This environment is a classic rocket trajectory optimization problem. According to Pontryagin's maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
        """

        self.observation_space = """
        The state s is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground (of value 1) or not (of value 0).
        State Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0
        """

        self.action_space = """
        There are four discrete actions available,
        0: do nothing
        1: fire left orientation engine (i.e., rotating the lander rightward)
        2: fire main engine (i.e., slowing the downward movement)
        3: fire right orientation engine (i.e, rotating the lander leftward)
        """

        self.reward_space = """
        After every step, a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

        For each step, the reward:
            a. is increased/decreased the closer/further the lander is to the landing pad (measured by euclidean distance)
            b. is increased/decreased the slower/faster the lander is moving
            c. is decreased the more the lander is tilted (angle not horizontal)
            d. is increased by 10 points for each leg that is in contact with the ground
            e. is decreased by 0.03 points each frame a side engine is firing
            f. is decreased by 0.3 points each frame the main engine is firing

        The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively.
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        The lander starts at the top centre of the viewport with a random initial force applied to its centre of mass.
        """

        self.termination = """
        The episode finishes if
        a. the lander crashes (the lander's body gets in contact with the moon);
        b. the lander gets outside of the viewport (x coordinate is greater than 1);
        c. the lander is not awake.

        From the Box2D docs, a body which is not awake is a body which doesn't move and doesn't collide with any other body:
        Note, when Box2D determines that a body (or group of bodies) has come to rest, the body enters a sleep state which has very little CPU overhead. If a body is awake and collides with a sleeping body, then the sleeping body wakes up. Bodies will also wake up if a joint or contact attached to them is destroyed.
        """


class CartPoleTask(Task):

    task_name = "CartPole-v1"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in “Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem”. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.
        """

        self.observation_space = """
        The observation is a ndarray with shape (4,) with the values corresponding to the following positions and velocities:
        Cart Position (range from -4.8 to 4.8), Cart Velocity (range from -Inf ro Inf), Pole Angle (range from ~ -0.418 rad (-24°) to ~ 0.418 rad (24°)), Pole Angular Velocity (range from -Inf to Inf)

        Note: While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
        - The cart x-position (index 0) can be take values between (-4.8, 4.8), but the episode terminates if the cart leaves the (-2.4, 2.4) range.
        - The pole angle can be observed between (-.418, .418) radians (or ±24°), but the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)
        """

        self.action_space = """
        The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.
        0: Push cart to the left
        1: Push cart to the right

        Note: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
        """

        self.reward_space = """
        Since the goal is to keep the pole upright for as long as possible, a reward of +1 for every step taken, including the termination step, is allotted.
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        All observations are assigned a uniformly random value in (-0.05, 0.05)
        """

        self.termination = """
        The episode ends if any one of the following occurs:
        a. Pole Angle is greater than ±12°
        b. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
        """

class PendulumTask(Task):

    task_name = "Pendulum-v1"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The inverted pendulum swingup problem is based on the classic problem in control theory. The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point.
        """

        self.observation_space = """
        The observation is a ndarray with shape (3,) representing the x coordinate, y coordinate of the pendulum's free end, and its angular velocity:
        x = cos(theta), ranging from -1.0 to 1.0; y = sin(theta), ranging from -1.0 to 1.0; Angular velocity, ranging from -8.0 to 8.0
        """

        self.action_space = """
        The dimensionality of the action space is one and each action is a ndarray with shape (1,) representing the torque applied to free end of the pendulum ranging from -2.0 to 2.0, defined as positive counterclockwise
        """

        self.reward_space = """
        The reward function is defined as (in latex):
        r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)
        where theta is the pendulum's angle normalized between [-pi, pi] (with 0 being in the upright position). Based on the above equation, the minimum reward that can be obtained is -(pi2 + 0.1 * 82 + 0.001 * 22) = -16.27, while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
        """

        self.transition_dynamics = """
        A specification of the pendulum's dynamic equations, i.e., how the torque τ affects the change in angle θ over time:
            a. A red pendulum swings in a plane with an angle θ (in radians) from the vertical x-axis, which points upward. The y-axis extends horizontally to the left, perpendicular to the x-axis. The pivot point of the pendulum is at the origin where these axes intersect;
            b. A torque τ acts in the counterclockwise direction, which would cause the pendulum to swing to the left. The angle θ is marked between the vertical x-axis and the pendulum rod, indicating the displacement angle from equilibrium.
        """

        self.init_state = """
        The starting state is a random angle in [-pi, pi] and a random angular velocity in [-1,1].
        """

        self.termination = """
        """

class AcrobotTask(Task):

    task_name = "Acrobot-v1"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The Acrobot environment is based on Sutton's work in “Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding” and Sutton and Barto's book. The system consists of two links connected linearly to form a chain, with one end of the chain fixed. The joint between the two links is actuated. The goal is to apply torques on the actuated joint to swing the free end of the outer-link above a given height while starting from the initial state of hanging downwards.
        """

        self.observation_space = """
        The observation is a ndarray with shape (6,) that provides information about the two rotational joint angles as well as their angular velocities:
        s[0]: Cosine of theta1, ranging from -1 to 1
        s[1]: Sine of theta1, ranging from -1 to 1
        s[2]: Cosine of theta2, ranging from -1 to 1
        s[3]: Sine of theta2, ranging from -1 to 1
        s[4]: Angular velocity of theta1, ranging from -12.567  to 12.567
        s[5]: Angular velocity of theta2, ranging from -28.274 to 28.274
        Note, theta1 is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly downwards. theta2 is relative to the angle of the first link. An angle of 0 corresponds to having the same angle between the two links.
        """

        self.action_space = """
        The action is discrete, deterministic, and represents the torque applied on the actuated joint between the two links.
        0: apply -1 torque (N m) to the actuated joint
        1: apply 0 torque (N m) to the actuated joint
        2: apply 1 torque (N m) to the actuated joint
        """

        self.reward_space = """
        The goal is to have the free end reach a designated target height in as few steps as possible, and as such all steps that do not reach the goal incur a reward of -1. Achieving the target height results in termination with a reward of 0. The reward threshold is -100.
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        Each parameter in the underlying state (theta1, theta2, and the two angular velocities) is initialized uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.
        """

        self.termination = """
        The episode ends if the free end reaches the target height, which is constructed as: -cos(theta1) - cos(theta2 + theta1) > 1.0
        """

class CliffWalkingTask(Task):

    task_name = "CliffWalking-v0"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The game starts with the player at location [3, 0] of the 4x12 grid world with the goal located at [3, 11]. If the player reaches the goal the episode ends. A cliff runs along [3, 1..10]. If the player moves to a cliff location it returns to the start location. The player makes moves until they reach the goal.
        """

        self.observation_space = """
        There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at the goal as the latter results in the end of the episode. What remains are all the positions of the first 3 rows plus the bottom-left cell.
        The observation is an integer value representing the player's current position as current_row * nrows + current_col (where both the row and col are indexed from 0).
        For example, the starting position [3, 0] can be calculated as follows: 3 * 12 + 0 = 36.
        """

        self.action_space = """
        The action shape is (1,) in the range {0, 3} indicating which direction to move the player.
        0: Move up
        1: Move right
        2: Move down
        3: Move left
        """

        self.reward_space = """
        Each time step incurs -1 reward, unless the player stepped into the cliff, which incurs -100 reward.
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        The episode starts with the player in state [36] (location [3, 0]).
        """

        self.termination = """
        The episode terminates when the player enters state [47] (location [3, 11]).
        """

class InvertedPendulumTask(Task):  # remove

    task_name = "InvertedPendulum-v4"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        This environment is the cartpole environment based on the work done by Barto, Sutton, and Anderson in “Neuronlike adaptive elements that can solve difficult learning control problems”, just like in the classic environments but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity). This environment involves a cart that can moved linearly, with a pole fixed on it at one end and having another end free. The cart can be pushed left or right, and the goal is to balance the pole on the top of the cart by applying forces on the cart.
        """

        self.observation_space = """
        The state space consists of positional values of different body parts of the pendulum system, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

        The observation is a ndarray with shape (4,) where the elements correspond to the following:
        s[0]: position of the cart along the linear surface
        s[1]: vertical angle (in rad) of the pole on the cart
        s[2]: linear velocity of the cart
        s[3]: angular velocity of the pole on the cart
        """

        self.action_space = """
        The agent take a 1-element vector for actions. The action space is a continuous (action) in [-3, 3], where action represents the numerical force applied to the cart (with magnitude representing the amount of force and sign representing the direction).
        """

        self.reward_space = """
        The goal is to make the inverted pendulum stand upright (within a certain angle limit) as long as possible - as such a reward of +1 is awarded for each timestep that the pole is upright.
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        All observations start in state (0.0, 0.0, 0.0, 0.0) with a uniform noise in the range of [-0.01, 0.01] added to the values for stochasticity.
        """

        self.termination = """
        The episode ends when the absolute value of the vertical angle between the pole and the cart is greater than 0.2 radian.
        """

class InvertedDoublePendulumTask(Task):

    task_name = "InvertedDoublePendulum-v4"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        This environment originates from control theory and builds on the cartpole environment based on the work done by Barto, Sutton, and Anderson in “Neuronlike adaptive elements that can solve difficult learning control problems”, powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity or constraints). This environment involves a cart that can moved linearly, with a pole fixed on it and a second pole fixed on the other end of the first one (leaving the second pole as the only one with one free end). The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces on the cart.
        """

        self.observation_space = """
        The state space consists of positional values of different body parts of the pendulum system, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

        The observation is a ndarray with shape (11,) where the elements correspond to the following:
        s[0]: position of the cart along the linear surface
        s[1]: sine of the angle between the cart and the first pole
        s[2]: sine of the angle between the two poles
        s[3]: cosine of the angle between the cart and the first pole
        s[4]: cosine of the angle between the two poles
        s[5]: velocity of the cart
        s[6]: angular velocity of the angle between the cart and the first pole
        s[7]: angular velocity of the angle between the two poles
        s[8]: the first constraint force
        s[9]: the second constraint force
        s[10]: the third constraint force
        Note, each constraint force is applied to contacts for each degree of freedom (3).
        """

        self.action_space = """
        The agent take a 1-element vector for actions. The action space is a continuous (action) in [-3, 3], where action represents the numerical force applied to the cart (with magnitude representing the amount of force and sign representing the direction).
        """

        self.reward_space = """
        The reward consists of two parts:

        alive_bonus: The goal is to make the second inverted pendulum stand upright (within a certain angle limit) as long as possible - as such a reward of +10 is awarded for each timestep that the second pole is upright.

        distance_penalty: This reward is a measure of how far the tip of the second pendulum (the only free end) moves, and it is calculated as 0.01 * x^2 + (y - 2)^2, where x is the x-coordinate of the tip and y is the y-coordinate of the tip of the second pole.

        velocity_penalty: A negative reward for penalising the agent if it moves too fast 0.001 * (v_1)^2 + 0.005 * (v_2)^2

        The total reward returned is reward = alive_bonus - distance_penalty - velocity_penalty
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range of [-0.1, 0.1] added to the positional values (cart position and pole angles) and standard normal force with a standard deviation of 0.1 added to the velocity values for stochasticity.
        """

        self.termination = """
        The episode ends when the y_coordinate of the tip of the second pole is less than or equal to 1. The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other).
        """


class MiniGridUnlockTask(Task):

    task_name = "MiniGrid-Unlock-v0"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The world is a 6x11 grid of tiles (cells) and the agent has to open a locked door.
        """

        self.observation_space = """
        The observation is a partially observable view of the environment (represented as a 7x7 grid) using a compact and efficient encoding, with 3 input values (OBJECT_TO_IDX, COLOR_TO_IDX, STATE) per visible grid cell, 7x7x3 values total. These input values are not pixels.

        - 'OBJECT_TO_IDX' is the mapping of object types to integers:
            {"unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4, "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10,}

        - 'COLOR_TO_IDX' is the mapping of colors to integers:
            {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

        - 'STATE' refers to the door state:
            {"open": 0, "closed": 1, "locked": 2}

        Additionally, the agent's direction is indexed by:
        0: Pointing right (positive X); 1: Pointing down (positive Y); 2: Pointing left (negative X); 3: Pointing up (negative Y)
        """

        self.action_space = """
        There are 7 actions, but only four are used in this scenario:
        0: Turn left
        1: Turn right
        2: Move forward
        5: Toggle (e.g., open doors, interact with objects)

        Actions 3 (Pick up an object), 4 (Drop the object being carried), and 6 (Task completed) are not utilized in this task.
        """

        self.reward_space = """
        A reward of 1 - 0.9 * (step_count / max_steps) is given for success, and 0 for failure.
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        """

        self.termination = """
        The episode ends if the agent opens the door.
        """


class FetchPushTask(Task):

    task_name = "FetchPush-v2"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The task in the environment is for a manipulator to move a block to a target position on top of a table by pushing with its gripper. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper (i.e., end effector). The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper is locked in a closed configuration in order to perform the push task. The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time.
        """

        self.observation_space = """
        The observation consists of a dictionary with information about the robot's end effector state and goal. The dictionary consists of the following 3 keys:
        - 'observation'. Its value is an ndarray of shape (25,), consisting of kinematic information of the block object and gripper. The elements of the array correspond to the following:
            observation[0]: End effector x position in global coordinates; observation[1]: End effector y position in global coordinates;
            observation[2]: End effector z position in global coordinates; observation[3]: Block x position in global coordinates;
            observation[4]: Block y position in global coordinates; observation[5]: Block z position in global coordinates;
            observation[6]: Relative block x position with respect to gripper x position in global coordinates. Equals to 'x_block - x_gripper';
            observation[7]: Relative block y position with respect to gripper y position in global coordinates. Equals to 'y_block - y_gripper';
            observation[8]: Relative block z position with respect to gripper z position in global coordinates. Equals to 'z_block - z_gripper';
            observation[9]: Joint displacement of the right gripper finger; observation[10]: Joint displacement of the left gripper finger;
            observation[11]: Global x rotation of the block in a XYZ Euler frame rotation; observation[12]: Global y rotation of the block in a XYZ Euler frame rotation;
            observation[13]: Global z rotation of the block in a XYZ Euler frame rotation; observation[14]: Relative block linear velocity in x direction with respect to the gripper;
            observation[15]: Relative block linear velocity in y direction with respect to the gripper; observation[16]: Relative block linear velocity in z direction with respect to the gripper;
            observation[17]: Block angular velocity along the x axis; observation[18]: Block angular velocity along the y axis;
            observation[19]: Block angular velocity along the z axis; observation[20]: End effector linear velocity x direction;
            observation[21]: End effector linear velocity y direction; observation[22]: End effector linear velocity z direction;
            observation[23]: Right gripper finger linear velocity; observation[24]: Right gripper finger linear velocity

        - 'desired_goal'. This key represents the final goal to be achieved. In this environment it is a 3-dimensional ndarray that consists of the three cartesian coordinates of the desired final block position [x,y,z]. In order for the robot to perform a push trajectory, the goal position can only be placed on top of the table. The elements of the array are the following:
            desired_goal[0]: Final goal block position in the x coordinate;
            desired_goal[1]: Final goal block position in the y coordinate;
            desired_goal[2]: Final goal block position in the z coordinate

        - 'achieved_goal'. This key represents the current state of the block, as if it would have achieved a goal. The value is an ndarray with shape (3,). The elements of the array are the following:
            achieved_goal[0]: Current block position in the x coordinate;
            achieved_goal[1]: Current block position in the y coordinate;
            achieved_goal[2]: Current block position in the z coordinate
        """

        self.action_space = """
        The action space is defined as a Box(-1.0, 1.0, (4,), float32), encompassing four dimensions. An action represents the Cartesian displacement dx, dy, and dz of the end effector. Additionally, the fourth action controls the closing and opening of the gripper. Specifically, the elements correspond to the following:
            action[0]: Displacement of the end effector in the x direction dx; action[1]: Displacement of the end effector in the y direction dy; action[2]: Displacement of the end effector in the z direction dz; action[3]: Positional displacement per timestep of each finger of the gripper
        Each component of the action can take any value between -1.0 and 1.0.
        """

        self.reward_space = """
        The returned reward can have two values: -1 if the block hasn't reached its final target position, and 0 if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        When the environment is reset the gripper is placed in the following global cartesian coordinates (x,y,z) = [1.3419 0.7491 0.555] m, and its orientation in quaternions is (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at (x,y,z) = [0.405, 0.48, 0] in global coordinates.

        The block's position has a fixed height of (z) = [0.42] m (on top of the table). The initial (x,y) position of the block is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of [-0.15, 0.15] m. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than 0.1 m. The initial orientation of the block is the same as for the gripper, (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]

        Finally the target position where the robot has to move the block is generated. The random target is also generated by adding an offset to the initial grippers position (x,y) sampled from a uniform distribution with a range of [-0.15, 0.15] m. The height of the target is initialized and fixed at (z) = [0.42] m on the table.
        """

        self.termination = """
        The episode will be truncated when the duration reaches 50 timesteps.
        """

class FetchSlideTask(Task):

    task_name = "FetchSlide-v2"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The task in the environment is for a manipulator to hit a puck in order to reach a target position on top of a long and slippery table. The table has a low friction coefficient in order to make it slippery for the puck to slide and be able to reach the target position which is outside of the robot's workspace. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper (i.e., end effector). The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper is locked in a closed configuration since the puck doesn't need to be graspped. The task is also continuing which means that the robot has to maintain the puck in the target position for an indefinite period of time.
        """

        self.observation_space = """
        The observation consists of a dictionary with information about the robot's end effector state and goal. The dictionary consists of the following 3 keys:
        - 'observation'. Its value is an ndarray of shape (25,), consisting of kinematic information of the puck object and gripper. The elements of the array correspond to the following:
            observation[0]: End effector x position in global coordinates; observation[1]: End effector y position in global coordinates;
            observation[2]: End effector z position in global coordinates; observation[3]: Puck x position in global coordinates;
            observation[4]: Puck y position in global coordinates; observation[5]: Puck z position in global coordinates;
            observation[6]: Relative puck x position with respect to gripper x position in global coordinates. Equals to 'x_puck - x_gripper';
            observation[7]: Relative puck y position with respect to gripper y position in global coordinates. Equals to 'y_puck - y_gripper';
            observation[8]: Relative puck z position with respect to gripper z position in global coordinates. Equals to 'z_puck - z_gripper';
            observation[9]: Joint displacement of the right gripper finger; observation[10]: Joint displacement of the left gripper finger;
            observation[11]: Global x rotation of the puck in a XYZ Euler frame rotation; observation[12]: Global y rotation of the puck in a XYZ Euler frame rotation;
            observation[13]: Global z rotation of the puck in a XYZ Euler frame rotation; observation[14]: Relative puck linear velocity in x direction with respect to the gripper;
            observation[15]: Relative puck linear velocity in y direction with respect to the gripper; observation[16]: Relative puck linear velocity in z direction with respect to the gripper;
            observation[17]: Puck angular velocity along the x axis; observation[18]: Puck angular velocity along the y axis;
            observation[19]: Puck angular velocity along the z axis; observation[20]: End effector linear velocity x direction;
            observation[21]: End effector linear velocity y direction; observation[22]: End effector linear velocity z direction;
            observation[23]: Right gripper finger linear velocity; observation[24]: Right gripper finger linear velocity

        - 'desired_goal'. This key represents the final goal to be achieved. In this environment it is a 3-dimensional ndarray that consists of the three cartesian coordinates of the desired final puck position [x,y,z]. In order for the robot to perform a push trajectory, the goal position can only be placed on top of the table. The elements of the array are the following:
            desired_goal[0]: Final goal puck position in the x coordinate;
            desired_goal[1]: Final goal puck position in the y coordinate;
            desired_goal[2]: Final goal puck position in the z coordinate

        - 'achieved_goal'. This key represents the current state of the puck, as if it would have achieved a goal. The value is an ndarray with shape (3,). The elements of the array are the following:
            achieved_goal[0]: Current puck position in the x coordinate;
            achieved_goal[1]: Current puck position in the y coordinate;
            achieved_goal[2]: Current puck position in the z coordinate
        """

        self.action_space = """
        The action space is defined as a Box(-1.0, 1.0, (4,), float32), encompassing four dimensions. An action represents the Cartesian displacement dx, dy, and dz of the end effector. Additionally, the fourth action controls the closing and opening of the gripper.
        Each component of the action can take any value between -1.0 and 1.0. The elements correspond to the following:
        action[0]: Displacement of the end effector in the x direction dx; action[1]: Displacement of the end effector in the y direction dy; action[2]: Displacement of the end effector in the z direction dz; action[3]: Positional displacement per timestep of each finger of the gripper
        """

        self.reward_space = """
        The returned reward can have two values: -1 if the puck hasn't reached its final target position, and 0 if the puck is in the final target position (the puck is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        When the environment is reset the gripper is placed in the following global cartesian coordinates (x,y,z) = [1 0.75 0.41] m, and its orientation in quaternions is (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at (x,y,z) = [0.405, 0.48, 0] in global coordinates.

        The puck's position has a fixed height of (z) = [0.42] m (on top of the table). The initial (x,y) position of the puck is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of [-0.1, 0.1] m. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the puck is greater than 0.1 m. The initial orientation of the puck is the same as for the gripper, (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]

        Finally the target position where the robot has to move the puck is generated. The random target is also generated by adding an offset to the initial grippers position (x,y) sampled from a uniform distribution with a range of [-0.3, 0.3] m. The height of the target is initialized and fixed at (z) = [0.42] m on the table.
        """

        self.termination = """
        The episode will be truncated when the duration reaches 50 timesteps.
        """

class FetchPickAndPlaceTask(Task):

    task_name = "FetchPickAndPlace-v2"

    def __init__(
        self,
        task_description=None,
        observation_space=None,
        action_space=None,
        reward_space=None,
        transition_dynamics=None,
        init_state=None,
        termination=None,
    ) -> None:
        super().__init__(
            task_description,
            observation_space,
            action_space,
            reward_space,
            transition_dynamics,
            init_state,
            termination,
        )

        self.task_description = """
        The task in the environment is for a manipulator to move a block to a target position on top of a table or in mid-air. The robot is a 7-DoF Fetch Mobile Manipulator with a two-fingered parallel gripper (i.e., end effector). The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper can be opened or closed in order to perform the graspping operation of pick and place. The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time.
        """

        self.observation_space = """
        The observation consists of a dictionary with information about the robot's end effector state and goal. The dictionary consists of the following 3 keys:
        - 'observation'. Its value is an ndarray of shape (25,), consisting of kinematic information of the block object and gripper. The elements of the array correspond to the following:
            observation[0]: End effector x position in global coordinates; observation[1]: End effector y position in global coordinates;
            observation[2]: End effector z position in global coordinates; observation[3]: Block x position in global coordinates;
            observation[4]: Block y position in global coordinates; observation[5]: Block z position in global coordinates;
            observation[6]: Relative block x position with respect to gripper x position in global coordinates. Equals to 'x_block - x_gripper';
            observation[7]: Relative block y position with respect to gripper y position in global coordinates. Equals to 'y_block - y_gripper';
            observation[8]: Relative block z position with respect to gripper z position in global coordinates. Equals to 'z_block - z_gripper';
            observation[9]: Joint displacement of the right gripper finger; observation[10]: Joint displacement of the left gripper finger;
            observation[11]: Global x rotation of the block in a XYZ Euler frame rotation; observation[12]: Global y rotation of the block in a XYZ Euler frame rotation;
            observation[13]: Global z rotation of the block in a XYZ Euler frame rotation; observation[14]: Relative block linear velocity in x direction with respect to the gripper;
            observation[15]: Relative block linear velocity in y direction with respect to the gripper; observation[16]: Relative block linear velocity in z direction with respect to the gripper;
            observation[17]: Block angular velocity along the x axis; observation[18]: Block angular velocity along the y axis;
            observation[19]: Block angular velocity along the z axis; observation[20]: End effector linear velocity x direction;
            observation[21]: End effector linear velocity y direction; observation[22]: End effector linear velocity z direction;
            observation[23]: Right gripper finger linear velocity; observation[24]: Right gripper finger linear velocity

        - 'desired_goal'. This key represents the final goal to be achieved. In this environment it is a 3-dimensional ndarray that consists of the three cartesian coordinates of the desired final block position [x,y,z]. In order for the robot to perform a push trajectory, the goal position can only be placed on top of the table. The elements of the array are the following:
            desired_goal[0]: Final goal block position in the x coordinate;
            desired_goal[1]: Final goal block position in the y coordinate;
            desired_goal[2]: Final goal block position in the z coordinate

        - 'achieved_goal'. This key represents the current state of the block, as if it would have achieved a goal. The value is an ndarray with shape (3,). The elements of the array are the following:
            achieved_goal[0]: Current block position in the x coordinate;
            achieved_goal[1]: Current block position in the y coordinate;
            achieved_goal[2]: Current block position in the z coordinate
        """

        self.action_space = """
        The action space is defined as a Box(-1.0, 1.0, (4,), float32), encompassing four dimensions. An action represents the Cartesian displacement dx, dy, and dz of the end effector. Additionally, the fourth action controls the closing and opening of the gripper. Specifically, the elements correspond to the following:
            action[0]: Displacement of the end effector in the x direction dx; action[1]: Displacement of the end effector in the y direction dy;
            action[2]: Displacement of the end effector in the z direction dz; action[3]: Positional displacement per timestep of each finger of the gripper
        Each component of the action can take any value between -1.0 and 1.0.
        """

        self.reward_space = """
        The returned reward can have two values: -1 if the block hasn't reached its final target position, and 0 if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """

        self.transition_dynamics = """
        """

        self.init_state = """
        When the environment is reset the gripper is placed in the following global cartesian coordinates (x,y,z) = [1.3419 0.7491 0.555] m, and its orientation in quaternions is (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at (x,y,z) = [0.405, 0.48, 0] in global coordinates.

        The block's position has a fixed height of (z) = [0.42] m (on top of the table). The initial (x,y) position of the block is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of [-0.15, 0.15] m. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than 0.1 m. The initial orientation of the block is the same as for the gripper, (w,x,y,z) = [1.0, 0.0, 1.0, 0.0]

        Finally the target position where the robot has to move the block is generated. The random target is also generated by adding an offset to the initial grippers position (x,y) sampled from a uniform distribution with a range of [-0.15, 0.15] m. The height of the target is initialized at (z) = [0.42] m and an offset is added to it sampled from another uniform distribution with a range of [0, 0.45] m.
        """

        self.termination = """
        The episode will be truncated when the duration reaches 50 timesteps.
        """


# class PendulumTask(Task):

#     task_name = "Pendulum-v1"

#     def __init__(
#         self,
#         task_description=None,
#         observation_space=None,
#         action_space=None,
#         reward_space=None,
#         transition_dynamics=None,
#         init_state=None,
#         termination=None,
#     ) -> None:
#         super().__init__(
#             task_description,
#             observation_space,
#             action_space,
#             reward_space,
#             transition_dynamics,
#             init_state,
#             termination,
#         )

#         self.task_description = """
#         """

#         self.observation_space = """
#         """

#         self.action_space = """
#         """

#         self.reward_space = """
#         """

#         self.transition_dynamics = """
#         """

#         self.init_state = """
#         """

#         self.termination = """
#         """

ALL_CLS = [
    MountainCarTask,
    LunarLanderTask,
    CartPoleTask,
    PendulumTask,
    AcrobotTask,
    CliffWalkingTask,
    InvertedPendulumTask,
    InvertedDoublePendulumTask,
    FetchPushTask,
    FetchSlideTask,
    FetchPickAndPlaceTask,
    MiniGridUnlockTask
    ]


def get_task_cls(task_name):
    for cls in ALL_CLS:
        if task_name == cls.task_name:
            return cls
    return None