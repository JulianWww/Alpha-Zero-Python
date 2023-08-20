from typing import Dict
from random import randint

from agent import Agent
from game import Game
from memory import ShortTermMemoryDummy, ShortTermMemory
from model import AlphaZeroModel

def play_game(playerA: Agent, playerB: Agent, memory: ShortTermMemory, starting_player: int, game: Game):
    if starting_player == 0:
        starting_player = randint(0, 1) * 2 - 1
    
    players = {
         starting_player: playerA,
        -starting_player: playerB
    }

    playerA.setState(game.state)
    playerB.setState(game.state)

    def updatePlayerPoses(state, action):
        playerA.moveRoot(action, state)
        if (not playerA is playerB):
            playerB.moveRoot(action, state)

    while (not game.is_done):
        agent = players[game.player]
        action, policy = agent.takeAction()

        memory.add_state(game.state, policy)
        game.takeAction(action)
        updatePlayerPoses(game.state, action)

    value, _, _ = game.getValues()
    memory.update_values(value, game.player)

    return memory

def generate_data():
    pass



game = Game()
model = AlphaZeroModel(42, (2,6,7), 3, 0.1, 0.9)
agent = Agent(model)
memory = ShortTermMemory()

play_game(agent, agent, memory, 0, game)