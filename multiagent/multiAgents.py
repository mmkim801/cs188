# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import pdb

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # pdb.set_trace()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
          return float("inf")
        elif successorGameState.isLose():
          return float("-inf")

        closest_ghost = 1000
        ghosts = [manhattanDistance(newPos, ghost.getPosition()) 
          for ghost in newGhostStates if ghost.scaredTimer == 0]
        if len(ghosts) != 0:
          closest_ghost = min(ghosts)
        if closest_ghost <= 2:
          # return float("-inf")
          return closest_ghost - 15
        closest_food = min([manhattanDistance(newPos, pellet) for pellet in newFood.asList()])
        return successorGameState.getScore() + (1.0/closest_food)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # pdb.set_trace()
        return self.maxValue(gameState, self.depth, "Stop")[1]
    
    def minValue(self, gameState, ghostIndex, depth, currentAction):
        # pdb.set_trace()
        if gameState.isWin() or gameState.isLose():
          return (self.evaluationFunction(gameState), currentAction)

        minVal = (float("inf"), None)
        for action in gameState.getLegalActions(ghostIndex):
          # if action == "Stop":
          #   continue
          if ghostIndex == gameState.getNumAgents() - 1:
            val = self.maxValue(gameState.generateSuccessor(ghostIndex, action), depth - 1, action)
          else:
            val = self.minValue(gameState.generateSuccessor(ghostIndex, action), ghostIndex + 1,depth, action)
          if minVal[0] > val[0]:
            minVal = (val[0], action)
        return minVal

    def maxValue(self, gameState, depth, currentAction):
        # pdb.set_trace()
        if depth == 0 or not gameState.getLegalActions(0):
          return (self.evaluationFunction(gameState), currentAction)

        maxVal = (float("-inf"), None)
        for action in gameState.getLegalActions(0):
          # if action == "Stop":
          #   continue
          val = self.minValue(gameState.generateSuccessor(0, action), 1, depth, action)
          if maxVal[0] < val[0]:
            maxVal = (val[0], action)
        return maxVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.pruningMax(gameState, self.depth, "Stop", float("-inf"), float("inf"))[1]

    def pruningMin(self, gameState, ghostIndex, depth, currentAction, alpha, beta):
        # pdb.set_trace()
        if gameState.isWin() or gameState.isLose():
          return (self.evaluationFunction(gameState), currentAction)

        minVal = (float("inf"), None)
        for action in gameState.getLegalActions(ghostIndex):
          if ghostIndex == gameState.getNumAgents() - 1:
            val = self.pruningMax(gameState.generateSuccessor(ghostIndex, action), depth - 1, action, alpha, beta)
          else:
            val = self.pruningMin(gameState.generateSuccessor(ghostIndex, action), ghostIndex + 1,depth, action, alpha, beta)
          if minVal[0] > val[0]:
            minVal = (val[0], action)
          if minVal[0] < alpha:
            return minVal
          beta = min(beta, minVal[0])
        return minVal

    def pruningMax(self, gameState, depth, currentAction, alpha, beta):
        # pdb.set_trace()
        if depth == 0 or not gameState.getLegalActions(0):
            return (self.evaluationFunction(gameState), currentAction)
        maxVal = (float("-inf"), None)
        for action in gameState.getLegalActions(0):
          val = self.pruningMin(gameState.generateSuccessor(0, action), 1, depth, action, alpha, beta)
          if maxVal[0] < val[0]:
            maxVal = (val[0], action)
          if maxVal[0] > beta:
            return maxVal
          alpha = max(alpha, maxVal[0])
        return maxVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, self.depth, "Stop")[1]

    def expValue(self, gameState, ghostIndex, depth):
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        sumVal = 0
        expVal = (float("inf"), None)
        for action in gameState.getLegalActions(ghostIndex):
          if ghostIndex == gameState.getNumAgents() - 1:
            sumVal += self.maxValue(gameState.generateSuccessor(ghostIndex, action), depth - 1, action)[0]
          else:
            sumVal += self.expValue(gameState.generateSuccessor(ghostIndex, action), ghostIndex + 1,depth)
        expVal = sumVal / len(gameState.getLegalActions(ghostIndex))

        return expVal



    def maxValue(self, gameState, depth, currentAction):
        if depth == 0 or not gameState.getLegalActions(0):
          return (self.evaluationFunction(gameState), currentAction)

        maxVal = (float("-inf"), None)
        for action in gameState.getLegalActions(0):
          val = self.expValue(gameState.generateSuccessor(0, action), 1, depth)
          if maxVal[0] < val:
            maxVal = (val, action)
        return maxVal

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I have modified the evaluationFunction() that we wrote in Question1.
      Instead of calculating successorGameState, I calculate the currentGameState.
      For calculating the score, we add the reciprocal of the distance of the closest food and
      that of the closest scared Ghost since those give us points. However, since eating all the
      food pellet determines the outcome of the game, I have given more weight on the food.
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
      return float("inf")
    elif currentGameState.isLose():
      return float("-inf")

    # if currentGameState.getScore() > 180:
    #   pdb.set_trace()
    closest_ghost = 1000
    ghosts = [manhattanDistance(pacPos, ghost.getPosition()) 
      for ghost in ghostStates if ghost.scaredTimer == 0]
    if len(ghosts) != 0:
      closest_ghost = min(ghosts)
    if closest_ghost <= 2:
      return closest_ghost - 100
    closest_food = min([manhattanDistance(pacPos, pellet) for pellet in foodPos.asList()])

    closest_scaredGhost = float("inf")
    scaredGhost = [manhattanDistance(pacPos, ghost.getPosition())
    for ghost in ghostStates if ghost.scaredTimer != 0]
    if len(scaredGhost) != 0:
      closest_scaredGhost = min(scaredGhost)
    return currentGameState.getScore() + (3.0/closest_food) + (1.0/closest_scaredGhost)

# Abbreviation
better = betterEvaluationFunction

