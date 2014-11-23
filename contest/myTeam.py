# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import Counter
from operator import itemgetter, attrgetter, methodcaller
import distanceCalculator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
        first = 'OffensiveAgent', second = 'DaveDefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        # Initialize observation distribution
        self.legalPositions = [p for p in gameState.getWalls().asList(False)]
        belief = util.Counter()
        for legalPosition in self.legalPositions:
            belief[legalPosition] = 1
        belief.normalize()

        self.beliefs = {}
        for enemyIndex in self.getOpponents(gameState):
            self.beliefs[enemyIndex] = belief.copy()

        # Set Start Position
        self.startPosition = gameState.getAgentPosition(self.index)

        # Load and Initialize Weights
        previousState = self.readState()

        self.weights = util.Counter()

        if False: #previousState != False:
            for key in previousState.keys():
                self.weights[key] = previousState[key]
        else:
            self.weights = self.getInitialWeights()

        #TODO randomize weights based on an alpha

    def final(self, gameState):

        #TODO recalculate weights based on score
        self.writeState(self.weights)

    def chooseAction(self, gameState):

        self.updateBeliefs(gameState)

        dist = []
        for i in range(gameState.getNumAgents()):
            if self.beliefs.has_key(i):
                dist.append(self.beliefs[i])
            else:
                dist.append(util.Counter())

        #-self.displayDistributionsOverPositions(dist)

        actions = [a for a in gameState.getLegalActions(self.index) if a is not Directions.STOP]

        actionWeights = util.Counter()
        for action in actions:
            actionWeights[action] = self.getValue(gameState, action)

        return actionWeights.argMax()

    def getValue(self, gameState, action):

        state = gameState.generateSuccessor(self.index, action)

        features = self.getFeatures(state)

        return features * self.weights

    def getFeatures(self, gameState):
        features = util.Counter()

        features['closestFood'] = self.getClosestFood(gameState)
        features['closestGhost'] = self.getClosestGhost(gameState)
        features['foodLeft'] = self.getFoodLeft(gameState)
        features['scoreFood'] = self.getScoreFood(gameState)
        features['score'] = gameState.getScore() #TODO: invert when on blue team
        features['caughtInCave'] = self.calculateIfWillBeCaughtInCave(gameState)

        return features

    def getInitialWeights(self):
        weights = util.Counter()

        weights['closestFood'] = -10
        weights['closestGhost'] = -200
        weights['foodLeft'] = -100
        weights['scoreFood'] = 120
        weights['score'] = 10000
        weights['caughtInCave'] = -100000000

        return weights

    def getClosestFood(self, gameState):
        agentPosition = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()

        foodDistances = [self.getMazeDistance(agentPosition, food) for food in foodList]

        return min(foodDistances)

    def getClosestGhost(self, gameState):

        #TODO find another path when stuck (maybe use minimax)

        enemyDistances = self.getEnemyDistanceBeliefs(gameState)

        if len(enemyDistances) == 0:
            return 0

        return 1.0/(min(enemyDistances))

    def getFoodLeft(self, gameState):
        return len(self.getFood(gameState).asList())

    def getScoreFood(self, gameState):

        numberOfFoodCarried = gameState.getAgentState(self.index).numCarrying
        distanceToStart = self.getMazeDistance(self.startPosition, gameState.getAgentPosition(self.index))

        return 1.0 * numberOfFoodCarried * numberOfFoodCarried / (distanceToStart + 1)

    def updateBeliefs(self, gameState):

        position = gameState.getAgentPosition(self.index)

        for enemyIndex in self.getOpponents(gameState):
            enemyPosition = gameState.getAgentPosition(enemyIndex)

            if util.manhattanDistance(position, enemyPosition) <= 5:
                belief = util.Counter()
                belief[enemyPosition] = 1
                self.beliefs[enemyIndex] = belief
                continue

            newBeliefs = util.Counter()
            for oldPosition in self.beliefs[enemyIndex]:

                x,y = oldPosition
                adjacentPositions = util.Counter()
                for diff in [-1,1]:
                    if self.legalPositions.__contains__((x + diff, y)):
                        adjacentPositions[(x + diff, y)] = 1000/(self.getMazeDistance(position, (x + diff, y)) + 1)
                    if self.legalPositions.__contains__((x, y + diff)):
                        adjacentPositions[(x, y + diff)] = 1000/(self.getMazeDistance(position, (x, y + diff)) + 1)

                adjacentPositions = adjacentPositions.sortedKeys()
                values = [90, 2, 2, 2]
                positionWeights = util.Counter()
                for i in range(len(adjacentPositions)):
                    positionWeights[adjacentPositions[i]] = values[i]

                positionWeights.normalize()

                for adjacentPosition in positionWeights:
                    newBeliefs[adjacentPosition] += positionWeights[adjacentPosition]/len(adjacentPositions) * self.beliefs[enemyIndex][oldPosition]

            newBeliefs.normalize()

            if newBeliefs.totalCount() < 1e-6:
                newBeliefs.incrementAll(self.legalPositions, 1)
                newBeliefs.normalize()

            self.beliefs[enemyIndex] = util.Counter()
            noisyDistance = self.getMazeDistance(position, enemyPosition)
            for oldPosition in self.legalPositions:
                if util.manhattanDistance(position, oldPosition) <= 5:
                    self.beliefs[enemyIndex][oldPosition] = 0
                else:
                    trueDistance = self.getMazeDistance(position, oldPosition)
                    prob = gameState.getDistanceProb(trueDistance, noisyDistance)
                    self.beliefs[enemyIndex][oldPosition] = prob * newBeliefs[oldPosition]

            self.beliefs[enemyIndex].normalize()

    def getEnemyDistanceBeliefs(self, gameState):

        position = gameState.getAgentPosition(self.index)
        enemyDistances = []

        for enemyIndex in self.getOpponents(gameState):

            enemyState = gameState.getAgentState(enemyIndex)
            if enemyState.isPacman or enemyState.scaredTimer is not 0:
                continue

            belief = self.beliefs[enemyIndex]

            distance = 0
            for loc in [l for l in belief if self.legalPositions.__contains__(l)]:
                distance += self.getMazeDistance(position, loc) * belief[loc]

            if distance > 0:
                enemyDistances.append(distance)

        return enemyDistances

    def calculateIfWillBeCaughtInCave(self, gameState):
        caveDepth = self.getIsTrappedInCave(gameState)
        ghostDistances = self.getEnemyDistanceBeliefs(gameState)

        if len(ghostDistances) == 0:
            return 0

        closestGhost = min(ghostDistances)

        if caveDepth <= 0:
            return 0
        if 2*(caveDepth + 1) > closestGhost:
            return 1

        return 0

    def getIsTrappedInCave(self, gameState):

        position = gameState.getAgentPosition(self.index)
        actions = [a for a in gameState.getLegalActions(self.index) if a is not Directions.STOP]

        if len(actions) > 2:
            return 0
        if len(actions) == 1:
            return 1

        for action in actions:
            movedTo = gameState.generateSuccessor(self.index, action)
            caveDepth = self.getCaveDepth(movedTo, {position})
            if caveDepth >= 0:
                return caveDepth + 1

        return 0

    def getCaveDepth(self, gameState, previousPositions):
        if len(previousPositions) > 10:
            return -20

        position = gameState.getAgentPosition(self.index)
        actions = [a for a in gameState.getLegalActions(self.index) if a is not Directions.STOP]
        previousPositions.add(position)

        if len(actions) > 2:
            return -20

        for action in actions:
            movedTo = gameState.generateSuccessor(self.index, action)
            if movedTo.getAgentPosition(self.index) not in previousPositions:
                caveDepth = self.getCaveDepth(movedTo, previousPositions)
                if caveDepth < 0:
                    return -20

                return caveDepth + 1

        return 1

######################### CUSTOM DEFENSIVE AGENT #########################

class DaveDefensiveAgent(CaptureAgent):

  MAX_DISTRIBUTION = 1

  def isOffense(self):
      return False;

  def setValidPositions(self, gameState):
    """
    Sets the field: validPositions to be a list of all valid position
    tuples on the map
    """
    self.validPositions = []
    wallsInMaze = gameState.getWalls()

    for x in range(wallsInMaze.width):
      for y in range(wallsInMaze.height):
        if not wallsInMaze[x][y]:
          self.validPositions.append((x,y))

  def getNeighbors(self, gameState, (x,y)):
    """
    Returns a list of valid neigboring tuple positions to the given position
    (x,y). The position (x,y) itself is returned in the list
    """
    walls = gameState.getWalls()
    positions = [(x,y)]

    #North
    if y+1 < walls.height and not walls[x][y+1]:
      positions.append((x,y+1))

    #East
    if x+1 < walls.width and not walls[x+1][y]:
      positions.append((x+1,y))

    #West
    if x-1 >= 0 and not walls[x-1][y]:
      positions.append((x-1,y))

    #South
    if y-1 >= 0 and not walls[x][y-1]:
      positions.append((x,y-1))

    return positions

  ######################## INFERENCE FUNCTIONS ########################

  def initializeBeliefDistributions(self, gameState):

    self.beliefDistributions = dict()
    for agent in self.getOpponents(gameState):
      distribution = Counter()
      self.beliefDistributions[agent] = Counter()
      wallsInMaze = gameState.getWalls()

      validPositions = self.validPositions
      midline = wallsInMaze.width / 2

      for (x,y) in validPositions:
        if gameState.isOnRedTeam(agent) and x <= midline or \
            not gameState.isOnRedTeam(agent) and x >= midline:

          self.beliefDistributions[agent][(x,y)] = self.MAX_DISTRIBUTION

      self.beliefDistributions[agent].normalize()

  def observe(self, observedState):

    agentPosition = observedState.getAgentPosition(self.index)
    noisyDistances = observedState.getAgentDistances()
    validPositions = self.validPositions

    newDistributions = dict()
    for agent in self.getOpponents(observedState):

      if self.beliefDistributions[agent].totalCount() == 0:

        for agent in self.getOpponents(observedState):

          if self.beliefDistributions[agent].totalCount() == 0:
            self.beliefDistributions[agent] = Counter()
            wallsInMaze = observedState.getWalls()

            validPositions = self.validPositions
            midline = wallsInMaze.width / 2

            for (x,y) in validPositions:

              if observedState.isOnRedTeam(agent) and x <= midline or \
                not observedState.isOnRedTeam(agent) and x >= midline:

                self.beliefDistributions[agent][(x,y)] = self.MAX_DISTRIBUTION

            self.beliefDistributions[agent].normalize()

      distribution = Counter()
      if observedState.data.agentStates[agent].configuration != None:
        distribution[observedState.data.agentStates[agent].configuration.getPosition()] = 1

      else:
        for position in validPositions:
          distance = util.manhattanDistance(agentPosition, position)
          distribution[position] = self.beliefDistributions[agent][position] * \
            observedState.getDistanceProb(distance, noisyDistances[agent])

        distribution.normalize()

      newDistributions[agent] = distribution

    self.beliefDistributions = newDistributions

  def getMostDangerousOpponents(self, observedState):

    opponentRanking = []
    isRed = self.red
    for opponent in self.getOpponents(observedState):
      position = observedState.getAgentPosition(opponent)

      if position is None:
          position = self.getMostLikelyPosition(opponent)
      (x,y) = position

      opponentRanking.append((x,opponent))

    if(isRed):
      opponentRanking.sort(key=itemgetter(0), reverse=False )
    else:
      opponentRanking.sort(key=itemgetter(0), reverse=True)

    result = []

    for (x,opponent) in opponentRanking:
      result.append(opponent)
    return result

  def elapseTime(self, observedState):

    newDistributions = dict()
    for agent in self.getOpponents(observedState):
      distribution = Counter()

      for position in self.validPositions:
        newPosDist = Counter()
        neighbors = self.getNeighbors(observedState, position)

        for neighborLocation in neighbors:
          newPosDist[neighborLocation] = 1
        newPosDist.normalize()

        items = newPosDist.items()

        for nextLocation, prob in items:
          distribution[nextLocation] += self.beliefDistributions[agent][position] * prob

      distribution.normalize()
      newDistributions[agent] = distribution

    self.beliefDistributions = newDistributions

  ###################### CONVENIENCE FUNCTIONS ######################

  def getMostLikelyPosition(self, agent):

    return self.beliefDistributions[agent].argMax()

  def getClosestAttacker(self, observedState):

    myPos = observedState.getAgentPosition(self.index)
    closestAttacker = None
    isPacman = False
    minDistance = float('inf')
    opponents = self.getOpponents(observedState)

    for agent in opponents:
      attackerPos = observedState.getAgentPosition(agent)

      if attackerPos is None:
          attackerPos = self.getMostLikelyPosition(agent)
      attackerDist = self.getMazeDistance(myPos, attackerPos)

      if (not isPacman and (attackerDist < minDistance or \
        observedState.getAgentState(agent).isPacman)) or \
        (observedState.getAgentState(agent).isPacman and \
        attackerDist < minDistance):

        if observedState.getAgentState(agent).isPacman:
            isPacman = True

        minDistance = attackerDist
        closestAttacker = agent

    return closestAttacker

  def getNumberOfOpponentsAttacking(self, gameState):
    opponents = self.getOpponents(gameState)

    numOpponents = 0

    for agent in opponents:
        attackerPos = gameState.getAgentPosition(agent)

        if attackerPos is None:
            attackerPos = self.getMostLikelyPosition(agent)
        (x,y) = attackerPos

        midline = gameState.getWalls().width / 2

        if gameState.isOnRedTeam(agent) and  x >= midline or \
              not gameState.isOnRedTeam(agent) and x <= midline:
            numOpponents = numOpponents + 1

    return numOpponents

  def registerInitialState(self, gameState):

    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.distancer.getMazeDistances()

    self.setValidPositions(gameState)
    self.initializeBeliefDistributions(gameState)

    """import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display"""

  def chooseAction(self, gameState):

    observedState = self.getCurrentObservation()
    self.observe(observedState)
    self.elapseTime(observedState)

    legalActions = observedState.getLegalActions(self.index)
    currentPosition = observedState.getAgentPosition(self.index)
    bestAction = Directions.STOP

    numDefense = 1
    numOffense = 1

    #Run away from enemies who have consumed pill
    if gameState.getAgentState(self.index).scaredTimer > 0:
      closestTarget = self.getClosestAttacker(observedState)
      targetPosition = observedState.getAgentPosition(closestTarget)

      #Get approximate enemy position if it isn't in LOS
      if targetPosition is None:
        targetPosition = self.getMostLikelyPosition(closestTarget)

      maxDistance = self.getMazeDistance(currentPosition, targetPosition)

      #Loop through actions to find best way to get enemy
      for action in legalActions:
        successor = observedState.generateSuccessor(self.index, action)
        agentPos = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(agentPos, targetPosition)

        #Update action and distance if a better one is found
        if dist > maxDistance and not successor.getAgentState(self.index).isPacman:
          maxDistance = dist
          bestAction = action

    #Chase down enemies
    else:
      targets = self.getMostDangerousOpponents(observedState)
      closestTarget = targets[0]
      targetPosition = observedState.getAgentPosition(closestTarget)

      #Get approximate enemy position if it isn't in LOS
      if targetPosition is None:
        targetPosition = self.getMostLikelyPosition(closestTarget)

      #If only Defender
      if self.getNumberOfOpponentsAttacking(gameState) == 0:
          (x,y) = (6,11)

          targetPosition = (x,y)
          print "TARGET POS " + str(targetPosition)

      minDistance = self.getMazeDistance(currentPosition, targetPosition)

      #Loop through actions to find best way to get enemy
      for action in legalActions:
        successor = observedState.generateSuccessor(self.index, action)
        agentPos = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(agentPos, targetPosition)

        #Update action and distance if a better one is found
        if dist < minDistance and not successor.getAgentState(self.index).isPacman:
          minDistance = dist
          bestAction = action

    return bestAction

class DaveDefensiveAgent2(DaveDefensiveAgent):

  def chooseAction(self, gameState):

    observedState = self.getCurrentObservation()
    self.observe(observedState)
    self.elapseTime(observedState)

    legalActions = observedState.getLegalActions(self.index)
    currentPosition = observedState.getAgentPosition(self.index)
    bestAction = Directions.STOP

    numDefense = 2
    numOffense = 0

    #Run away from enemies who have consumed pill
    if gameState.getAgentState(self.index).scaredTimer > 0:
      closestTarget = self.getClosestAttacker(observedState)
      targetPosition = observedState.getAgentPosition(closestTarget)

      #Get approximate enemy position if it isn't in LOS
      if targetPosition is None:
        targetPosition = self.getMostLikelyPosition(closestTarget)

      maxDistance = self.getMazeDistance(currentPosition, targetPosition)

      #Loop through actions to find best way to get enemy
      for action in legalActions:
        successor = observedState.generateSuccessor(self.index, action)
        agentPos = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(agentPos, targetPosition)

        #Update action and distance if a better one is found
        if dist > maxDistance and not successor.getAgentState(self.index).isPacman:
          maxDistance = dist
          bestAction = action

    #Chase down enemies
    else:
      targets = self.getMostDangerousOpponents(observedState)
      if self.getNumberOfOpponentsAttacking(observedState) == 1:
          closestTarget = targets[0]
      else:
          closestTarget = targets[self.index/2]

      targetPosition = observedState.getAgentPosition(closestTarget)

      #Get approximate enemy position if it isn't in LOS
      if targetPosition is None:
        targetPosition = self.getMostLikelyPosition(closestTarget)

      #Split Zones
      if self.getNumberOfOpponentsAttacking(gameState) == 0:
          if(self.index / 2 == 0):
                (x,y) = (4,6)
          else:
            (x,y) = (11, 6)
          targetPosition = (x,y)
          print "TARGET POS " + str(targetPosition)

      minDistance = self.getMazeDistance(currentPosition, targetPosition)

      #Loop through actions to find best way to get enemy
      for action in legalActions:
        successor = observedState.generateSuccessor(self.index, action)
        agentPos = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(agentPos, targetPosition)

        #Update action and distance if a better one is found
        if dist < minDistance and not successor.getAgentState(self.index).isPacman:
          minDistance = dist
          bestAction = action

    return bestAction