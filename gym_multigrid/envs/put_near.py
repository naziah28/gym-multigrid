from gym_multigrid.multigrid import *

class PutNearEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        goal_pst = [],
        goal_index = [],
        num_balls=[],
        agents_index = [],
        balls_index=[],
        zero_sum=False,
        path=[]
    ):

        self.num_balls = num_balls
        self.goal_pst = goal_pst
        self.goal_index = goal_index

        # colour of the ball
        self.balls_index = balls_index
        self.zero_sum = zero_sum
        self.path=path

        agents = []
        for i in agents_index:
            agent = Agent(i)
            agents.append(agent)
            agent.pos = (1,1)
            print(agent.pos)


        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=True,
            agents=agents,
            partial_obs=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        # add in maze walls
        if len(self.path) < 1:
            for wall in self.walls:
                self.grid.set(*wall, Wall())
        else:
            for i in range(1, self.width - 1):
                for j in range(1, self.height - 1):
                    if (j, i) not in self.path:
                        self.grid.set(j, i, Wall())

        for i in range(len(self.goal_pst)):
            self.place_obj(ObjectGoal(self.goal_index[i], 'ball'), top=self.goal_pst[i], size=[1,1])

        for number, index in zip(self.num_balls, self.balls_index):
            for i in range(number):
                self.place_obj(Ball(index))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a, a.pos)

    def _reward(self, i, rewards,reward=1):
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info


class PutNearEnv12x12N2(PutNearEnv):
    def __init__(self):
        super().__init__(size=None,
        height=12,
        width=12,
        goal_pst = [[6,5]],
        goal_index = [0],
        num_balls=[2, 4],
        agents_index = [4,4,4,4],
        balls_index=[5,5,5,5,5],
        zero_sum=True,
        path=[
            (1, 1), (2, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9,1), (10,1),
            (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6,2), (10, 2),
            (1, 3), (5, 3), (10, 3),
            (1, 4), (5, 4), (10, 4),
            (1, 5), (5, 5), (10, 5),
            (1, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6),
            (1, 7), (7, 7), (10, 7),
            (1, 8), (7, 8), (10, 8),
            (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (10, 9),
            (1, 10), (7, 10), (8, 10), (9, 10), (10, 10),
            (6, 2), (9, 7), (9, 2), (6, 10), (6,5), (6,4), (7,5), (7,4)],
        )