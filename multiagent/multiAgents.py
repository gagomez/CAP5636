# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

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

        ghostsCost = 0
        for ghost in newGhostStates:
            ghostCost = manhattanDistance(ghost.getPosition(), newPos)
            if ghost.scaredTimer > ghostCost:
                ghostCost = 1000000/(ghostCost + 1)
            elif ghostCost < 4:
                ghostCost = -10000/(ghostCost + 1)
            else:
                ghostCost = 0

            ghostsCost += ghostCost

        minToFood = 999999
        foodList = newFood.asList()
        for food in foodList:
            minToFood = min(minToFood, manhattanDistance(food, newPos))

        if len(foodList) == 0:
            minToFood = 0

        score = 50000/(len(foodList) + 1) + 10000 - minToFood + ghostsCost

        #print "food count =", len(foodList)
        #print newPos, "=", score

        return score

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
        """

        cost, action = self.value(gameState, 0)

        return action

    def value(self, gameState, layer):
        numAgents = gameState.getNumAgents()

        if layer >= (self.depth * numAgents):
            return self.evaluationFunction(gameState), Directions.STOP

        agentId = layer % numAgents
        if agentId == self.index:
            return self.max_value(gameState, layer)
        else:
            return self.min_value(gameState, layer)

    def max_value(self, gameState, layer):
        maxCost = -999999
        agentId = layer % gameState.getNumAgents()
        actions = gameState.getLegalActions(agentId)

        if len(actions) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        for action in actions:
            cost, _ = self.value(gameState.generateSuccessor(agentId, action), layer + 1)

            if cost > maxCost:
                maxCost = cost
                maxAction = action

        return maxCost, maxAction

    def min_value(self, gameState, layer):
        minCost = 999999
        agentId = layer % gameState.getNumAgents()
        actions = gameState.getLegalActions(agentId)

        if len(actions) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        for action in actions:
            cost, _ = self.value(gameState.generateSuccessor(agentId, action), layer + 1)

            if cost < minCost:
                minCost = cost
                minAction = action

        return minCost, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        cost, action = self.value(gameState, 0, -999999, 999999)

        return action

    def value(self, gameState, layer, alpha, beta):
        numAgents = gameState.getNumAgents()

        if layer >= (self.depth * numAgents):
            return self.evaluationFunction(gameState), Directions.STOP

        agentId = layer % numAgents
        if agentId == self.index:
            return self.max_value(gameState, layer, alpha, beta)
        else:
            return self.min_value(gameState, layer, alpha, beta)

    def max_value(self, gameState, layer, alpha, beta):
        maxCost = -999999
        agentId = layer % gameState.getNumAgents()
        actions = gameState.getLegalActions(agentId)

        if len(actions) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        for action in actions:
            cost, _ = self.value(gameState.generateSuccessor(agentId, action), layer + 1, alpha, beta)

            if cost > beta:
                return cost, Directions.STOP

            alpha = max(alpha, cost)

            if cost > maxCost:
                maxCost = cost
                maxAction = action

        return maxCost, maxAction

    def min_value(self, gameState, layer, alpha, beta):
        minCost = 999999
        agentId = layer % gameState.getNumAgents()
        actions = gameState.getLegalActions(agentId)

        if len(actions) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        for action in actions:
            cost, _ = self.value(gameState.generateSuccessor(agentId, action), layer + 1, alpha, beta)

            if cost < alpha:
                return cost, Directions.STOP

            beta = min(beta, cost)

            if cost < minCost:
                minCost = cost
                minAction = action

        return minCost, minAction

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
        cost, action = self.value(gameState, 0)

        return action

    def value(self, gameState, layer):
        numAgents = gameState.getNumAgents()

        if layer >= (self.depth * numAgents):
            return self.evaluationFunction(gameState), Directions.STOP

        agentId = layer % numAgents
        if agentId == self.index:
            return self.max_value(gameState, layer)
        else:
            return self.chance_value(gameState, layer)

    def max_value(self, gameState, layer):
        maxCost = -999999
        agentId = layer % gameState.getNumAgents()
        actions = gameState.getLegalActions(agentId)

        if len(actions) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        for action in actions:
            cost, _ = self.value(gameState.generateSuccessor(agentId, action), layer + 1)

            if cost > maxCost:
                maxCost = cost
                maxAction = action

        return maxCost, maxAction

    def chance_value(self, gameState, layer):
        agentId = layer % gameState.getNumAgents()
        actions = gameState.getLegalActions(agentId)

        if len(actions) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        prob = 1.0/len(actions)
        chanceCost = 0
        for action in actions:
            cost, _ = self.value(gameState.generateSuccessor(agentId, action), layer +1)

            chanceCost += prob * cost

        return chanceCost, Directions.STOP

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      COST = GHOST_SCORE + FOOD_SCORE + MIN_DISTANCE_TO_FOOD

      GHOST_SCORE:
      Reciprocal cost for distance to ghosts only when they are close.
      Take scared ghosts into account as long as they are reachable in time.

      FOOD_SCORE:
      Take into account having less food left than the minimum distance to the next food.

      MIN_DISTANCE_TO_FOOD:
      Linear minimum distance to food to avoid rounding/precision issues.
    """
    position = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    ghostsCost = 0
    for ghost in ghostStates:
        ghostCost = manhattanDistance(ghost.getPosition(), position)
        if ghost.scaredTimer > ghostCost:
            ghostCost = 1000000/(ghostCost + 1)
        elif ghostCost < 4:
            ghostCost = -10000/(ghostCost + 1)
        else:
            ghostCost = 0

        ghostsCost += ghostCost

    minToFood = 999999
    foodList = foodGrid.asList()
    for food in foodList:
        minToFood = min(minToFood, manhattanDistance(food, position))

    if len(foodList) == 0:
        minToFood = 0

    score = 50000/(len(foodList) + 1) + 10000 - minToFood + ghostsCost

    return score

# Abbreviation
better = betterEvaluationFunction

