from game import GameState
from config import MCTS_SIMS
from mcts import Node, randomMoveSelector

class Agent:
  def __init__(self, model):
    self.model = model
    self.state = None
  
  def setState(self, state: GameState):
    self.state = state
    self.root = Node(state, 0)
  
  def takeAction(self):
    for i in range(MCTS_SIMS):
      leaf = self.root.moveToLeaf(randomMoveSelector)

  def evaluateLeaf(self, leaf):
    
