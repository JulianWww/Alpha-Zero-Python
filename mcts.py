from game import GameState
import config
from math import sqrt
from numpy.random import choice
import numpy as np

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
    print([float(x.policy) for x in self.children.values()])
    return str(self.state) + \
      "\n" + \
      ", ".join([str(x.policy) for x in self.children.values()])

  
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

  def expand(self, model):
    state = self.state.toTensor()
    value, policy = model.predict(state.reshape(*(1, *(state.shape))) + 1, verbose = 0)
    policy = self.policy_softmax(policy[0], self.state.legalActions)
    
    for action, poly in zip(self.state.legalActions, policy):
      self.children[action] = Node(self.state.takeAction(action), poly, self)

    return value
  
  @staticmethod
  def policy_softmax(policy, allowed_actions):
    policy = policy[allowed_actions]
    policy = np.e**policy
    policy = policy / np.sum(policy)
    return policy


