import random
import math

BOT_NAME = "Noisy Boy" # INSERT NAME FOR YOUR BOT HERE OR IT WILL THROW AN EXCEPTION

class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
  
    rseed = None  # change this to a value if you want consistent random choices

    def __init__(self):
        if self.rseed is None:
            self.rstate = None
        else:
            random.seed(self.rseed)
            self.rstate = random.getstate()

    def get_move(self, state):
        if self.rstate is not None:
            random.setstate(self.rstate)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move.  Very slow and not always smart."""

    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Gets called by get_move() to determine the value of each successor state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        #
        # Fill this in!
        #
        nextp = state.next_player()
        return self.minimax_helper(state, nextp)  

    def minimax_helper(self,state,p1):
        if (state.is_full()):
            return state.utility()
        if p1 == 1:
            max = -math.inf
            for move, i in state.successors():
                utility = self.minimax_helper(i,-1*p1)
                if utility > max:
                    max = utility
            return max
        if p1 == -1:
            min = math.inf
            for move, i in state.successors():
                utility = self.minimax_helper(i,-1*p1)
                if utility < min:
                    min = utility
            return min           

class MinimaxLookaheadAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        nextp = state.next_player()
        if self.depth_limit == 0:
            return self.evaluation(state)
        if self.depth_limit == None:
            return self.minimax_helper(state,nextp)
        else:
            return self.minimax_depth(state,nextp,self.depth_limit) 
        
    def minimax_depth(self, state, p1, depth):
        """This is just a helper method for minimax(). Feel free to use it or not. """
        if (state.is_full()):
            return state.utility()
        if (depth == 0):
            return self.evaluation(state)
        if p1 == 1:
            max = -math.inf
            for move, i in state.successors():
                utility = self.minimax_depth(i,-1*p1,depth-1)
                if utility > max:
                    max = utility
            return max
        if p1 == -1:
            min = math.inf
            for move, i in state.successors():
                utility = self.minimax_depth(i,-1*p1,depth-1)
                if utility < min:
                    min = utility
            return min

    def evaluation(self, state):
        cols = state.get_cols()
        rows = state.get_rows()
        diags = state.get_diags()
        currUtil = state.utility()
        p1eval = 0
        p2eval = 0
        p1fours = 0
        p2fours = 0
        p1threes = 0
        p2threes = 0
        rowCount = 0
        bottomFilled = False

        for diag in diags:
            length = len(diag)
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [diag[i],diag[i+1],diag[i+2],diag[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [0,0,1,1]:
                        p1fours+=0.5
                    if sequenceFour == [0,1,1,1]:
                        p1fours += 1
                    else:
                        sequenceThree = [diag[i], diag[i+1], diag[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [0,1,1]:
                            p1threes += 1
            if length >= 4:
                sequenceThree = [diag[length-3],diag[length-2],diag[length-1]]
                sequenceThree.sort()
                if sequenceThree == [0,1,1]:
                    p1threes += 1
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [diag[i],diag[i+1],diag[i+2],diag[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [-1,-1,0,0]:
                        p2fours+=0.5
                    if sequenceFour == [-1,-1,-1,0]:
                        p2fours += 1
                    else:
                        sequenceThree = [diag[i], diag[i+1], diag[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [-1,-1,0]:
                            p2threes += 1
            
            if length >= 4:
                sequenceThree = [diag[length-3],diag[length-2],diag[length-1]]
                sequenceThree.sort()
                if sequenceThree == [-1,-1,0]:
                    p2threes += 1        

        for row in rows:
            length = len(row)
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [row[i],row[i+1],row[i+2],row[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [0,0,1,1]:
                        p1fours+=0.5
                    if sequenceFour == [0,1,1,1]:
                        for k in range(i,i+4):
                            if row[k] == 0:
                                if rowCount - 1 >= 0:
                                    if state.board[rowCount-1][k] != 0:
                                        bottomFilled = True
                        if bottomFilled:
                            if state.next_player() == 1: p1fours += 2 
                            else: p1fours -= 0.5
                        else: p1fours += 1
                        bottomFilled = False
                    else:
                        sequenceThree = [row[i], row[i+1], row[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [0,1,1]:
                            for k in range(i,i+3):
                                if row[k] == 0:
                                    if rowCount - 1 >= 0:
                                        if state.board[rowCount-1][k] != 0:
                                            bottomFilled = True
                        if bottomFilled:
                            if state.next_player() == 1: p1threes += 2
                            else: p1threes -= 0.5 
                        else: p1threes += 1
                        bottomFilled = False
            sequenceThree = [row[length-3],row[length-2],row[length-1]]
            sequenceThree.sort()
            if sequenceThree == [0,1,1]:
                p1threes += 1

            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [row[i],row[i+1],row[i+2],row[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [-1,-1,0,0]:
                        p2fours+=0.5
                    if sequenceFour == [-1,-1,-1,0]:
                        for k in range(i,i+4):
                            if row[k] == 0:
                                if rowCount - 1 >= 0:
                                    if state.board[rowCount-1][k] != 0:
                                        bottomFilled = True
                        if bottomFilled:
                            if state.next_player() == -1: p2fours += 2 
                            else: p2fours -= 0.5
                        else: p2fours += 1
                        bottomFilled = False
                    else:
                        sequenceThree = [row[i], row[i+1], row[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [-1,-1,0]:
                            for k in range(i,i+3):
                                if row[k] == 0:
                                    if rowCount - 1 >= 0:
                                        if state.board[rowCount-1][k] != 0:
                                            bottomFilled = True
                        if bottomFilled:
                            if state.next_player() == -1: p2threes += 2
                            else: p2threes -= 0.5 
                        else: p2threes += 1
                        bottomFilled = False
            sequenceThree = [row[length-3],row[length-2],row[length-1]]
            sequenceThree.sort()
            if sequenceThree == [-1,-1,0]:
                p2threes += 1
            rowCount += 1

        for col in cols:
            length = len(col)
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [col[i],col[i+1],col[i+2],col[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [0,0,1,1]:
                        p1fours+=0.5
                    if sequenceFour == [0,1,1,1]:
                        p1fours += 1
                    else:
                        sequenceThree = [col[i], col[i+1], col[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [0,1,1]:
                            p1threes += 1
            sequenceThree = [col[length-3],col[length-2],col[length-1]]
            sequenceThree.sort()
            if sequenceThree == [0,1,1]:
                p1threes += 1

            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [col[i],col[i+1],col[i+2],col[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [-1,-1,0,0]:
                        p2fours+=0.5
                    if sequenceFour == [-1,-1,-1,0]:
                        p2fours += 1
                    else:
                        sequenceThree = [col[i], col[i+1], col[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [-1,-1,0]:
                            p2threes += 1
            sequenceThree = [col[length-3],col[length-2],col[length-1]]
            sequenceThree.sort()
            if sequenceThree == [-1,-1,0]:
                p2threes += 1      

        p1eval = 4*p1fours + 3*p1threes
        p2eval = 4*p2fours + 3*p2threes
        return currUtil + (p1eval-p2eval)# Change this line, unless you have something better to do.

class AltMinimaxLookaheadAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move.
 
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want? 
    """

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        Gets called by get_move() to determine the value of successor states.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the (possibly estimated) minimax utility value of the state
        """
        #
        # Fill this in!
        #
        nextp = state.next_player()
        if self.depth_limit == 0:
            return self.evaluation(state)
        if self.depth_limit == None:
            return self.minimax_helper(state,nextp)
        else:
            return self.minimax_depth(state,nextp,self.depth_limit) 
        
    def minimax_depth(self, state, p1, depth):
        """This is just a helper method for minimax(). Feel free to use it or not. """
        if (state.is_full()):
            return state.utility()
        if (depth == 0):
            return self.evaluation(state)
        if p1 == 1:
            max = -math.inf
            for move, i in state.successors():
                utility = self.minimax_depth(i,-1*p1,depth-1)
                if utility > max:
                    max = utility
            return max
        if p1 == -1:
            min = math.inf
            for move, i in state.successors():
                utility = self.minimax_depth(i,-1*p1,depth-1)
                if utility < min:
                    min = utility
            return min

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        Gets called by minimax() once the depth limit has been reached.  
        N.B.: This method must run in "constant" time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        #
        # Fill this in!
        #

        # Note: This cannot be "return state.utility() + c", where c is a constant. 
        cols = state.get_cols()
        rows = state.get_rows()
        diags = state.get_diags()
        currUtil = state.utility()
        p1eval = 0
        p2eval = 0
        p1fours = 0
        p2fours = 0
        p1threes = 0
        p2threes = 0

        for diag in diags:
            length = len(diag)
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [diag[i],diag[i+1],diag[i+2],diag[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [0,1,1,1]:
                        p1fours += 1
                    else:
                        sequenceThree = [diag[i], diag[i+1], diag[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [0,1,1]:
                            p1threes += 1
            if length >= 4:
                sequenceThree = [diag[length-3],diag[length-2],diag[length-1]]
                sequenceThree.sort()
                if sequenceThree == [0,1,1]:
                    p1threes += 1
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [diag[i],diag[i+1],diag[i+2],diag[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [-1,-1,-1,0]:
                        p2fours += 1
                    else:
                        sequenceThree = [diag[i], diag[i+1], diag[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [-1,-1,0]:
                            p2threes += 1
            
            if length >= 4:
                sequenceThree = [diag[length-3],diag[length-2],diag[length-1]]
                sequenceThree.sort()
                if sequenceThree == [-1,-1,0]:
                    p2threes += 1        

        for row in rows:
            length = len(row)
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [row[i],row[i+1],row[i+2],row[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [0,1,1,1]:
                        p1fours += 1
                    else:
                        sequenceThree = [row[i], row[i+1], row[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [0,1,1]:
                            p1threes += 1
            sequenceThree = [row[length-3],row[length-2],row[length-1]]
            sequenceThree.sort()
            if sequenceThree == [0,1,1]:
                p1threes += 1
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [row[i],row[i+1],row[i+2],row[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [-1,-1,-1,0]:
                        p2fours += 1
                    else:
                        sequenceThree = [row[i], row[i+1], row[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [-1,-1,0]:
                            p2threes += 1
            sequenceThree = [row[length-3],row[length-2],row[length-1]]
            sequenceThree.sort()
            if sequenceThree == [-1,-1,0]:
                p2threes += 1

        for col in cols:
            length = len(col)
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [col[i],col[i+1],col[i+2],col[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [0,1,1,1]:
                        p1fours += 1
                    else:
                        sequenceThree = [col[i], col[i+1], col[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [0,1,1]:
                            p1threes += 1
            sequenceThree = [col[length-3],col[length-2],col[length-1]]
            sequenceThree.sort()
            if sequenceThree == [0,1,1]:
                p1threes += 1
            for i in range(length-3):
                if i+1 and i+2 and i+3 < length: 
                    sequenceFour = [col[i],col[i+1],col[i+2],col[i+3]]
                    sequenceFour.sort()
                    if sequenceFour == [-1,-1,-1,0]:
                        p2fours += 1
                    else:
                        sequenceThree = [col[i], col[i+1], col[i+2]]
                        sequenceThree.sort()
                        if sequenceThree == [-1,-1,0]:
                            p2threes += 1
            sequenceThree = [col[length-3],col[length-2],col[length-1]]
            sequenceThree.sort()
            if sequenceThree == [-1,-1,0]:
                p2threes += 1         

        p1eval = 4*p1fours + 3*p1threes
        p2eval = 4*p2fours + 3*p2threes
        return currUtil + (p1eval-p2eval)


class MinimaxPruneAgent(MinimaxAgent):
    """Computer agent that uses minimax with alpha-beta pruning to select the best move.
    
    Hint: Consider what you did for MinimaxAgent.  What do you need to change to prune a
    branch of the state space? 
    """
    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not have a depth limit.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to column 1 (we're trading optimality for gradeability here).

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        #
        # Fill this in!
        #
        nextp = state.next_player()  # Change this line!
        return self.alphabeta(state, nextp, -math.inf, math.inf)

    def alphabeta(self, state,p1,alpha, beta):
        """This is just a helper method for minimax(). Feel free to use it or not."""
        if (state.is_full()):
            return state.utility()
        if p1 == 1:
            bestVal = -math.inf
            for move, i in state.successors():
                value = self.alphabeta(i,-1*p1,alpha, beta)
                bestVal = max(bestVal,value)
                alpha = max(alpha,bestVal)
                if beta <= alpha:
                    break
            return bestVal
        if p1 == -1:
            bestVal = math.inf
            for move, i in state.successors():
                value = self.alphabeta(i,-1*p1,alpha, beta)
                bestVal = min(bestVal,value)
                beta = min(beta,bestVal)
                if beta <= alpha:
                    break
            return bestVal


def get_agent(tag):
    if tag == 'random':
        return RandomAgent()
    elif tag == 'human':
        return HumanAgent()
    elif tag == 'mini':
        return MinimaxAgent()
    elif tag == 'prune':
        return MinimaxPruneAgent()
    elif tag.startswith('look'):
        depth = int(tag[4:])
        return MinimaxLookaheadAgent(depth)
    elif tag.startswith('alt'):
        depth = int(tag[3:])
        return AltMinimaxLookaheadAgent(depth)
    else:
        raise ValueError("bad agent tag: '{}'".format(tag))