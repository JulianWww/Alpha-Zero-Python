from numpy import zeros, max as npmax, argmax, ndarray

from game import GameState, Game
from config import MCTS_SIMS
from mcts import Node, randomMoveSelector
from model import AlphaZeroModel

def deterministic_selector(policy: ndarray):
    return int(argmax(policy)), policy

class Agent:
  def __init__(self, model):
    self.model = model
    self.state = None
  
  def setState(self, state: GameState):
    self.state = state
    self.root = Node(state, 0)
  
  def getSelector(self):
    return deterministic_selector
  
  def takeAction(self):
    for i in range(MCTS_SIMS):
      leaf = self.root.moveToLeaf(randomMoveSelector)
      leaf.expand(self.model)

    policy = zeros((Game.action_space))
    for action, child in self.root.children.items():
      policy[action] = child.evalutation
    
    policy /= npmax(policy)
    
    return self.getSelector()(policy)
  
  def moveRoot(self, action, state):
    if (self.root.isLeaf()):
      self.setState(state)

    self.root = self.root.children[action]  

#game = Game()
#model = AlphaZeroModel(42, (2,6,7), 3, 0.1, 0.9)
#agent = Agent(model)
#
#agent.setState(game.state)
#print(agent.takeAction())