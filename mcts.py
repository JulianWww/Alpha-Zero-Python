class Node:
  def __init__(self, policy):
    self.policy = policy
    self.evaluations = 0
    self.rewardAccumulation = 0
    self.Quotient = 0
    self.children = {}
  