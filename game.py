import numpy as np;
from bitlist import bitlist
from copy import copy

pieces = ["-", "X", "O"]

class Game:
  def __init__(self):
    name = "connect4"
    self.state = GameState(bitlist(0, length=84), 1, list(range(35, 42)))
  
  def __str__(self) -> str:
    return str(self.state)

  __repr__ = __str__

  def takeAction(self, action):
    self.state = self.state.takeAction(action)


class GameState:
  def __init__(self, board, player, legalActions, done=False):
    self.board = board
    self.player = player
    self.legalActions = legalActions
    self.done = done
  
  def getValues(self):
    return [-1, -1, 1] # [current player reward, current player points, other player points]
  
  def get_value(self, idx):
    if self.board[idx] == 1:
      return 1
    if self.board[idx + 42] == 1:
      return -1
    return 0

  def getRow(self, rowIdx: int) -> str:
    return " ".join(["+" if i in self.legalActions else pieces[self.get_value(i)] for i in range(7*rowIdx, 7*(rowIdx+1))])
  
  def __str__(self) -> str:
    return "\n".join([self.getRow(i) for i in range(6)]) \
      + f"\nPlayer {pieces[self.player]}" + f" {self.done}"

  __repr__ = __str__

  @staticmethod
  def computeDistance(center, step, board, offset):
    distance = 0
    for i in range(1, 4):
      point = center + step * i
      if point < 0 or point > 41:
        break

      if board[point + offset] == 1:
        distance+=1
      else:
        return distance

      mod = point % 7
      if mod == 0 or mod == 6:
        break
    return distance

  @staticmethod
  def checkDone(center: int, board: bitlist, player):
    steps = [(1, -1), (7, -7), (8, -8), (6, -6)]

    if player == 1:
      offset = 0
    else:
      offset = 42

    for step1, step2 in steps:
      if GameState.computeDistance(center, step1, board, offset)\
       + GameState.computeDistance(center, step2, board, offset) >= 3:
        return True

    return False


  def takeAction(self, action: int):
    board = copy(self.board)

    newAction = action - 7
    legalActions = copy(self.legalActions)
    if (newAction >= 0):
      legalActions.append(newAction)
    legalActions.remove(action)

    actionIdx = action
    if self.player == -1:
      actionIdx += 42

    board[actionIdx] = 1


    return GameState(board, -self.player, legalActions, self.checkDone(action, board, self.player))
  
  def toTensor(self):
    arr = np.array(self.board).reshape((2,6,7))
    if self.player == -1:
      arr = np.flip(arr, axis=0)
    
    return arr

game = Game()
game.takeAction(40)
game.state.toTensor()