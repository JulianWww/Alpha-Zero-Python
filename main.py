from game import Game
from model import AlphaZeroModel
from agent import Agent
from memory import ShortTermMemory
from play_game import play_game
from pickle import dumps

game = Game()
model = AlphaZeroModel(42, (2,6,7), 3, 0.1, 0.9)
agent = Agent(model)
memory = ShortTermMemory()

play_game(agent, agent, memory, 0, game)

print(dumps(memory.game[0]))