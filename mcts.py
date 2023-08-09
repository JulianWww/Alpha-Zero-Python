from game import GameState, Game
import config
from math import sqrt
from numpy.random import choice
from numpy import zeros

def randomMoveSelector(options, probability):
  return choice(options, p=probability)


class Node:
  def __init__(self, state: GameState, policy, parent=None):
    self.policy = policy
    self.evalutation = 0
    self.rewardAcc = 0
    self.quotient = 0

    self.children = {}
    self.parent = parent
    self.state = state
  
  def __str__(self):
    return str(self.state)
  
  def isLeaf(self):
    return len(self.children) == 0

  def isRoot(self):
    return self.parent is None
  
  def makeParent(self):
    self.parent = None
  
  def computeQU(self, root):
    return self.quotient + config.CPUT * self.policy * root / (self.evalutation + 1)

  def moveToLeaf(self, selector):
    if self.isLeaf():
      return self

    #root_sum = 0
    #for child in self.children.values():
    #  root_sum += child.evalutation
    self.evalutation += 1
    root = sqrt(self.evalutation)

    actions = list(self.children.keys())
    actionProbs = [child.computeQU(root) for child in self.children.values()]
    actionProbs /= sum(actionProbs)

    next = self.children[selector(actions, actionProbs)]
    return next.moveToLeaf(selector)

  def expand(self, policy):
    for action in self.state.legalActions:
      self.children[action] = Node(self.state.takeAction(action), policy[action], self)

#game = Game()
#node = Node(game.state, 5)
#node.moveToLeaf(randomMoveSelector)
#node.expand(zeros(42)+1)
#print(node.moveToLeaf(randomMoveSelector))
