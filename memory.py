from typing import List
from random import choice

from game import GameState, Game

class MemoryNode:
    def __init__(self, state: GameState, policy):
        self.state = state
        self.policy = policy
    
    def __str__(self):
        return str(self.state) + "\n" + str(self.policy.reshape(Game.input_space[1:])) + "\n"
    
    __repr__ = __str__

    def setValue(self, value, player):
        self.value = value * player * self.state.player

class ShortTermMemoryDummy:
    def __init__(self):
        pass
    
    def update_values(self, value: float, player: int):
        pass

    def add_state(self, state: GameState, Policy):
        pass

    def get_states() -> List[MemoryNode]:
        return []

class ShortTermMemory(ShortTermMemoryDummy):
    def __init__(self):
        super().__init__()
        self.game: List[MemoryNode] = []
    
    def update_values(self, value: float, player: int):
        for node in self.game:
            node.setValue(value, player)

    def add_state(self, state: GameState, policy):
        self.game.append(MemoryNode(state, policy))
    
    def get_states(self):
        return self.game

class LongTermMemmory:
    def __init__(self):
        self.memory = []
    
    def addNodes(self, stmemory: ShortTermMemoryDummy):
        self.memory.extend(stmemory.get_states())
    
    def __len__(self):
        return self.memory.__len__()
    
    def get_batch(self, size):
        return choice(self.memory, size)