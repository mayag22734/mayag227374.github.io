## Search Algorithms in Snake

**Project description:** This project uses a premade pygame version of snake to create an AI player utilizing various search algorithms. The algorithm used can be toggled in the SearchBasedPlayer class. Available algorithms are BFS, DFS, Heuristic, and A*. Dark blue squares represent tiles explored by the algorithm, yellow the goal, and red obstacles which will reduce the player's score.

### Implementation
```python
#!/usr/bin/env python3
from typing import List, Set
from dataclasses import dataclass
import pygame
from enum import Enum, unique
import sys
import random
from queue import Queue, LifoQueue


FPS = 10

INIT_LENGTH = 4

WIDTH = 480
HEIGHT = 480
GRID_SIDE = 24
GRID_WIDTH = WIDTH // GRID_SIDE
GRID_HEIGHT = HEIGHT // GRID_SIDE

BRIGHT_BG = (103, 223, 235)
DARK_BG = (78, 165, 173)

SNAKE_COL = (6, 38, 7)
FOOD_COL = (224, 160, 38)
OBSTACLE_COL = (209, 59, 59)
VISITED_COL = (24, 42, 142)


@unique
class Direction(tuple, Enum):
	UP = (0, -1)
	DOWN = (0, 1)
	LEFT = (-1, 0)
	RIGHT = (1, 0)

	def reverse(self):
		x, y = self.value
		return Direction((x * -1, y * -1))


@dataclass
class Position:
	x: int
	y: int

	def check_bounds(self, width: int, height: int):
		return (self.x >= width) or (self.x < 0) or (self.y >= height) or (self.y < 0)

	def draw_node(self, surface: pygame.Surface, color: tuple, background: tuple):
		r = pygame.Rect(
			(int(self.x * GRID_SIDE), int(self.y * GRID_SIDE)), (GRID_SIDE, GRID_SIDE)
		)
		pygame.draw.rect(surface, color, r)
		pygame.draw.rect(surface, background, r, 1)

	def __eq__(self, o: object) -> bool:
		if isinstance(o, Position):
			return (self.x == o.x) and (self.y == o.y)
		else:
			return False

	def __str__(self):
		return f"X{self.x};Y{self.y};"

	def __hash__(self):
		return hash(str(self))


class GameNode:
	nodes: Set[Position] = set()

	def __init__(self):
		self.position = Position(0, 0)
		self.color = (0, 0, 0)

	def randomize_position(self):
		try:
			GameNode.nodes.remove(self.position)
		except KeyError:
			pass

		condidate_position = Position(
			random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1),
		)

		if condidate_position not in GameNode.nodes:
			self.position = condidate_position
			GameNode.nodes.add(self.position)
		else:
			self.randomize_position()

	def draw(self, surface: pygame.Surface):
		self.position.draw_node(surface, self.color, BRIGHT_BG)


class Food(GameNode):
	def __init__(self):
		super(Food, self).__init__()
		self.color = FOOD_COL
		self.randomize_position()


class Obstacle(GameNode):
	def __init__(self):
		super(Obstacle, self).__init__()
		self.color = OBSTACLE_COL
		self.randomize_position()


class Snake:
	def __init__(self, screen_width, screen_height, init_length):
		self.color = SNAKE_COL
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.init_length = init_length
		self.reset()

	def reset(self):
		self.length = self.init_length
		self.positions = [Position((GRID_SIDE / 2), (GRID_SIDE / 2))]
		self.direction = random.choice([e for e in Direction])
		self.score = 0
		self.hasReset = True

	def get_head_position(self) -> Position:
		return self.positions[0]

	def turn(self, direction: Direction):
		if self.length > 1 and direction.reverse() == self.direction:
			return
		else:
			self.direction = direction

	def move(self):
		cur = self.get_head_position()
		x, y = self.direction.value
		new = Position(cur.x + x, cur.y + y,)
		self.hasReset = False
		if self.collide(new):
			self.reset()
		else:
			self.positions.insert(0, new)
			while len(self.positions) > self.length:
				self.positions.pop()

	def collide(self, new: Position):
		return (new in self.positions) or (new.check_bounds(GRID_WIDTH, GRID_HEIGHT))

	def eat(self, food: Food):
		if self.get_head_position() == food.position:
			self.length += 1
			self.score += 1
			while food.position in self.positions:
				food.randomize_position()

	def hit_obstacle(self, obstacle: Obstacle):
		if self.get_head_position() == obstacle.position:
			self.length -= 1
			self.score -= 1
			if self.length == 0:
				self.reset()

	def draw(self, surface: pygame.Surface):
		for p in self.positions:
			p.draw_node(surface, self.color, BRIGHT_BG)


class Player:
	def __init__(self) -> None:
		self.visited_color = VISITED_COL
		self.visited: Set[Position] = set()
		self.chosen_path: List[Direction] = []

	def move(self, snake: Snake) -> bool:
		try:
			next_step = self.chosen_path.pop(0)
			snake.turn(next_step)
			return False
		except IndexError:
			return True

	def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
		"""
		Do nothing, control is defined in derived classes
		"""
		pass

	def turn(self, direction: Direction):
		"""
		Do nothing, control is defined in derived classes
		"""
		pass

	def draw_visited(self, surface: pygame.Surface):
		for p in self.visited:
			p.draw_node(surface, self.visited_color, BRIGHT_BG)


class SnakeGame:
	def __init__(self, snake: Snake, player: Player) -> None:
		pygame.init()

		self.snake = snake
		self.food = Food()
		self.obstacles: Set[Obstacle] = set()
		for _ in range(40):
			ob = Obstacle()
			while any([ob.position == o.position for o in self.obstacles]):
				ob.randomize_position()
			self.obstacles.add(ob)

		self.player = player

		self.fps_clock = pygame.time.Clock()

		self.screen = pygame.display.set_mode(
			(snake.screen_height, snake.screen_width), 0, 32
		)
		self.surface = pygame.Surface(self.screen.get_size()).convert()
		self.myfont = pygame.font.SysFont("monospace", 16)

	def drawGrid(self):
		for y in range(0, int(GRID_HEIGHT)):
			for x in range(0, int(GRID_WIDTH)):
				p = Position(x, y)
				if (x + y) % 2 == 0:
					p.draw_node(self.surface, BRIGHT_BG, BRIGHT_BG)
				else:
					p.draw_node(self.surface, DARK_BG, DARK_BG)

	def run(self):
		while not self.handle_events():
			self.fps_clock.tick(FPS)
			self.drawGrid()
			if self.player.move(self.snake) or self.snake.hasReset:
				self.player.search_path(self.snake, self.food, self.obstacles)
				self.player.move(self.snake)
			self.snake.move()
			self.snake.eat(self.food)
			for ob in self.obstacles:
				self.snake.hit_obstacle(ob)
			for ob in self.obstacles:
				ob.draw(self.surface)
			self.player.draw_visited(self.surface)
			self.snake.draw(self.surface)
			self.food.draw(self.surface)
			self.screen.blit(self.surface, (0, 0))
			text = self.myfont.render(
				"Score {0}".format(self.snake.score), 1, (0, 0, 0)
			)
			self.screen.blit(text, (5, 10))
			pygame.display.update()

	def handle_events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					pygame.quit()
					sys.exit()
				if event.key == pygame.K_UP:
					self.player.turn(Direction.UP)
				elif event.key == pygame.K_DOWN:
					self.player.turn(Direction.DOWN)
				elif event.key == pygame.K_LEFT:
					self.player.turn(Direction.LEFT)
				elif event.key == pygame.K_RIGHT:
					self.player.turn(Direction.RIGHT)
		return False


class HumanPlayer(Player):
	def __init__(self):
		super(HumanPlayer, self).__init__()

	def turn(self, direction: Direction):
		self.chosen_path.append(direction)

OBS_COST = 4

class Node(Position):
	def __init__(self, x, y, previous, previousDir):
		self.x = x
		self.y = y
		self.previous = previous
		self.previousDir = previousDir

	def checkBounds(self):
		return (self.x >= GRID_WIDTH) or (self.x < 0) or (self.y >= GRID_HEIGHT) or (self.y < 0)

	def addCost(self, cost):
		self.cost = cost
		return self.cost
	
	def addDist(self, pos: Position):
		self.dist = abs(self.x-pos.x) + abs(self.y-pos.y)
		return self.dist

class SearchBasedPlayer(Player):
	def __init__(self):
		super(SearchBasedPlayer, self).__init__()
		#self.searchMethod = 'bfs'
		#self.searchMethod = 'dfs'
		#self.searchMethod = 'heuristic'
		self.searchMethod = 'a*'

	def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
		self.visited.clear()
		self.chosen_path.clear()
		root = Node(snake.get_head_position().x,snake.get_head_position().y,None,snake.direction)
		if self.searchMethod == 'bfs':
			self.BFS(snake, food, root)
		elif self.searchMethod == 'dfs':
			self.DFS(snake, food, root)
		elif self.searchMethod == 'heuristic':
			self.heuristicSearch(snake, food, root)
		elif self.searchMethod == 'a*':
			self.aStarSearch(snake, food, *obstacles, root)

	def aStarSearch(self, snake: Snake, food: Food, obstacles: Set[Obstacle], root: Node):
		unexpanded = []
		unexpanded.append(root)
		root.addCost(0)
		root.addDist(food.position)
		for pos in snake.positions:
			pos = Node(pos.x, pos.y, None, None)
			self.visited.add(pos)
		while len(unexpanded) > 0:
			minCost = 100
			indexMin = 0
			for i in range(0,len(unexpanded)):
				if (unexpanded[i].cost + unexpanded[i].dist) < minCost:
					minCost = unexpanded[i].cost + unexpanded[i].dist
					indexMin = i
			node = unexpanded[indexMin]
			unexpanded.remove(node)
			if (node.x == food.position.x) and (node.y == food.position.y):
				while node.previous:
					self.chosen_path.insert(0,node.previousDir)
					node = node.previous
				return
			for dir in Direction:
				newNode = Node(node.x + dir.value[0], node.y + dir.value[1], node, dir)
				if not newNode.checkBounds() and not newNode in self.visited and not dir == node.previousDir.reverse():
					unexpanded.append(newNode)
					self.visited.add(newNode)
					newNode.addDist(food.position)
					newNode.addCost(node.cost)
					for obs in obstacles:
						if (obs.position.x == newNode.x) and (obs.position.y == newNode.y):
							newNode.addCost(node.cost+OBS_COST)
							break
		if len(unexpanded) == 0:
			print("ERROR: No legal path found.")

	def heuristicSearch(self, snake: Snake, food: Food, root: Node):
		unexpanded = []
		unexpanded.append(root)
		root.addDist(food.position)
		for pos in snake.positions:
			pos = Node(pos.x, pos.y, None, None)
			self.visited.add(pos)
		while len(unexpanded) > 0:
			minDist = GRID_SIDE*2
			indexMin = 0
			for i in range(0,len(unexpanded)):
				if unexpanded[i].dist < minDist:
					minDist = unexpanded[i].dist
					indexMin = i
			node = unexpanded[indexMin]
			unexpanded.remove(node)
			if (node.x == food.position.x) and (node.y == food.position.y):
				while node.previous:
					self.chosen_path.insert(0,node.previousDir)
					node = node.previous
				return
			for dir in Direction:
				newNode = Node(node.x + dir.value[0], node.y + dir.value[1], node, dir)
				if not newNode.checkBounds() and not newNode in self.visited and not dir == node.previousDir.reverse():
					unexpanded.append(newNode)
					self.visited.add(newNode)
					newNode.addDist(food.position)
		if len(unexpanded) == 0:
			print("ERROR: No legal path found.")

	def DFS(self, snake: Snake, food: Food, root: Node):
		unexpanded = LifoQueue()
		unexpanded.put(root)
		for pos in snake.positions:
			pos = Node(pos.x, pos.y, None, None)
			self.visited.add(pos)
		self.visited.remove(root)
		while not unexpanded.empty():
			node = unexpanded.get()
			if not node in self.visited:
				self.visited.add(node)
				if (node.x == food.position.x) and (node.y == food.position.y):
					while node.previous:
						self.chosen_path.insert(0,node.previousDir)
						node = node.previous
					return
				for dir in Direction:
					newNode = Node(node.x + dir.value[0], node.y + dir.value[1], node, dir)
					if not newNode.checkBounds() and not dir == node.previousDir.reverse():
						unexpanded.put(newNode)
		if unexpanded.empty():
			print("ERROR: No legal path found.")

	def BFS(self, snake: Snake, food: Food, root: Node):
		unexpanded = Queue()
		unexpanded.put(root)
		for pos in snake.positions:
			pos = Node(pos.x, pos.y, None, None)
			self.visited.add(pos)
		while not unexpanded.empty():
			node = unexpanded.get()
			if (node.x == food.position.x) and (node.y == food.position.y):
				while node.previous:
					self.chosen_path.insert(0,node.previousDir)
					node = node.previous
				return
			for dir in Direction:
				newNode = Node(node.x + dir.value[0], node.y + dir.value[1], node, dir)
				if not newNode.checkBounds() and not newNode in self.visited and not dir == node.previousDir.reverse():
					unexpanded.put(newNode)
					self.visited.add(newNode)
		if unexpanded.empty():
			print("ERROR: No legal path found.")

if __name__ == "__main__":
	snake = Snake(WIDTH, WIDTH, INIT_LENGTH)
	#player = HumanPlayer()
	player = SearchBasedPlayer()
	game = SnakeGame(snake, player)
	game.run()

```
