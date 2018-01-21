# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import pdb
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # fringe = util.Stack()
    # trace = {}
    # visited = [problem.getStartState()]
    # for child in problem.getSuccessors(visited[0]):
    #     fringe.push(child)
    #     trace[child] = visited[0]
    # while not fringe.isEmpty():
    #     current_node = fringe.pop()
    #     if problem.isGoalState(current_node[0]):
    #         return trackBack(problem.getStartState(), current_node, trace)
    #     if current_node[0] not in visited:
    #         visited.append(current_node[0])
    #         successors = problem.getSuccessors(current_node[0])
    #         for child in successors:
    #             # pdb.set_trace()
    #             fringe.push(child)
    #             trace[child] = current_node
    # return []

    fringe = util.Stack()
    fringe.push(((problem.getStartState(), None, 0), []))
    visited = []
    # pdb.set_trace()
    while not fringe.isEmpty():
        current_node, path = fringe.pop()
        if problem.isGoalState(current_node[0]):
            return path
        if current_node[0] not in visited:
            visited.append(current_node[0])
            for child in problem.getSuccessors(current_node[0]):
                new_path = path + [child[1]]
                fringe.push((child, new_path))
    return []

# def trackBack(start, goal, trace):
#     path = []
#     node = goal
#     # pdb.set_trace()
#     while node != start:
#         path.append(node[1])
#         node = trace[node]
#     path.reverse()
#     return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # pdb.set_trace()
    """
    fringe = util.Queue()
    trace = {}
    visited = [problem.getStartState()]
    for child in problem.getSuccessors(visited[0]):
        fringe.push(child)
        trace[child] = visited[0]
    while not fringe.isEmpty():
        current_node = fringe.pop()
        if problem.isGoalState(current_node[0]):
            return trackBack(problem.getStartState(), current_node, trace)
        if current_node[0] not in visited:
            visited.append(current_node[0])
            successors = problem.getSuccessors(current_node[0])
            for child in successors:
                fringe.push(child)
                trace[child] = current_node
    return []
    """

    fringe = util.Queue()
    fringe.push(((problem.getStartState(), None, 0), []))
    visited = []
    # pdb.set_trace()
    while not fringe.isEmpty():
        current_node, path = fringe.pop()
        if problem.isGoalState(current_node[0]):
            return path
        if current_node[0] not in visited:
            visited.append(current_node[0])
            for child in problem.getSuccessors(current_node[0]):
                new_path = path + [child[1]]
                fringe.push((child, new_path))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # fringe = util.PriorityQueue()
    # visited = [problem.getStartState()]
    # trace = {}
    # # pdb.set_trace()
    # for child in problem.getSuccessors(visited[0]):
    #     fringe.push(child, child[2])
    #     trace[child] = visited[0]
    # while not fringe.isEmpty():
    #     current_node = fringe.pop()
    #     if problem.isGoalState(current_node[0]):
    #         return trackBack(problem.getStartState(), current_node, trace)
    #     if current_node[0] not in visited:
    #         visited.append((current_node[0]))
    #         successors = problem.getSuccessors(current_node[0])
    #         for child in successors:
    #             cost = child[2] + current_node[2]
    #             updated_child = (child[0], child[1], cost)
    #             fringe.update(updated_child, cost)
    #             trace[updated_child] = current_node
    # # pdb.set_trace()
    # return []
    fringe = util.PriorityQueue()
    visited = []
    fringe.push((problem.getStartState(), [], 0), 0)

    while not fringe.isEmpty():
        current_node, path, cost = fringe.pop()
        if problem.isGoalState(current_node):
            return path
        if current_node not in visited:
            visited.append(current_node)
            for child in problem.getSuccessors(current_node):
                new_path = path + [child[1]]
                new_cost = problem.getCostOfActions(new_path)
                fringe.update((child[0], new_path, new_cost), new_cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # fringe = util.PriorityQueue()
    # visited = [problem.getStartState()]
    # trace = {}
    # for child in problem.getSuccessors(visited[0]):
    #     fringe.push(child, heuristic(child[0], problem) + child[2])
    #     trace[child] = visited[0]

    # while not fringe.isEmpty():
    #     current_node = fringe.pop()
    #     if problem.isGoalState(current_node[0]):
    #         return trackBack(problem.getStartState(), current_node, trace)
    #     if current_node[0] not in visited:
    #         visited.append(current_node[0])
    #         successors = problem.getSuccessors(current_node[0])
    #         for child in successors:
    #             cost = child[2] + current_node[2]
    #             updated_child = (child[0], child[1], cost)
    #             fringe.push(updated_child, heuristic(updated_child[0], problem) + cost)
    #             trace[updated_child] = current_node
    # return[]

    fringe = util.PriorityQueue()
    visited = []
    fringe.push((problem.getStartState(), [], 0), 0)

    while not fringe.isEmpty():
        current_node, path, cost = fringe.pop()
        if problem.isGoalState(current_node):
            return path
        if current_node not in visited:
            visited.append(current_node)
            for child in problem.getSuccessors(current_node):
                new_path = path + [child[1]]
                new_cost = problem.getCostOfActions(new_path)
                fringe.update((child[0], new_path, new_cost), new_cost + heuristic(child[0], problem))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
