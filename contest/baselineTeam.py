# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from keyboardAgents import KeyboardAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from util import Counter
from operator import itemgetter, attrgetter, methodcaller

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CustomDefensiveAgent', second = 'CustomDefensiveAgent'):
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
  if isRed:
    return [eval("DaveDefensiveAgent2")(firstIndex), eval("DaveDefensiveAgent2")(secondIndex)]   
      
  else:
    return [eval("OffensiveReflexAgent")(firstIndex), eval("KeyboardAgent")(secondIndex)]
  
  

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def isOffense(self):
      return True;
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def isOffense(self):
      return False;

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
    

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
 
    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display

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
    

