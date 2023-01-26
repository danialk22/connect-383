import sys
import argparse

import agents
import boards


class GameState:
    """Class representing a single state of a Connect 4-esque game.

    For details on the game, see: https://en.wikipedia.org/wiki/Connect_Four

    Once created, a game state object should usually not be modified; instead, use the successors()
    function to generate reachable states.

    The board is stored as a 2D list, containing 1's representing Player 1's pieces and -1's
    for Player 2 (unused spaces are 0, and "obstacles" are -2).
    """
    
    state_count = 0  # bookkeeping to help track how efficient agents' search methods are running
    
    def __init__(self, board):
        """Constructor for Connect383 state.

        Args:
            Either a board (sequence of sequences, filled with 1, -1, 0, or -2, describing game state)
            or two numbers specifying the rows, columns for a blank board
        """
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.board = tuple([ tuple(board[r]) for r in range(self.num_rows) ])

        # 1 for Player 1, -1 for Player 2
        self._next_p = 1 if (sum(sum(row) for row in self.board) % 2) == 0 else -1  
        self._moves_left = sum(sum([1 if x == 0 else 0 for x in row]) for row in self.board)

    def next_player(self):
        """Determines who's move it is based on the board state.

        Returns: 1 if Player 1 goes next, -1 if it's Player 2's turn
        """
        return self._next_p

    def is_full(self):
        """Checks to see if there are available moves left."""
        return self._moves_left <= 0

    def _create_successor(self, col):
        """Create the successor state that follows from a given move."""

        successor_board = [ list(row) for row in self.board ]
        row = 0
        while (successor_board[row][col] != 0):
            row += 1
        if row >= self.num_rows:
            raise Exception("Illegal successor: {}, {}".format(col, self.board))
        successor_board[row][col] = self._next_p
        successor = GameState(successor_board)
        GameState.state_count += 1
        return successor

    def successors(self):
        """Generates successor state objects for all valid moves from this board.

        Returns: a _sorted_ list of (move, state) tuples
        """
        move_states = []
        for col in range(self.num_cols):
            if self.board[self.num_rows-1][col] == 0:
                move_states.append((col, self._create_successor(col))) 
        return move_states

    # These accessor methods might be useful for calculation an agent's evaluation method!  They
    # are based on:
    # https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
    
    def get_rows(self):
        """Return a list of rows for the board."""
        return [[c for c in r] for r in self.board]

    def get_cols(self):
        """Return a list of columns for the board."""
        return list(zip(*self.board))

    def get_diags(self):
        """Return a list of all the diagonals for the board."""
        b = [None] * (len(self.board) - 1)
        grid_forward = [b[i:] + r + b[:i] for i, r in enumerate(self.get_rows())]
        forwards = [[c for c in r if c is not None] for r in zip(*grid_forward)]
        grid_back = [b[:i] + r + b[i:] for i, r in enumerate(self.get_rows())]
        backs = [[c for c in r if c is not None] for r in zip(*grid_back)]
        return forwards + backs

    def scores(self):
        """Calculate the score for each player.
        
        Players are awarded points for each streak (horizontal, vertical, or diagonal) of length 3
        or greater equal to the square of the length (e.g., 4-in-a-row scores 16 points).        
        """
        p1_score = 0
        p2_score = 0
        for run in self.get_rows() + self.get_cols() + self.get_diags():
            for elt, length in streaks(run):
                if (elt == 1) and (length >= 3):
                    p1_score += length**2
                elif (elt == -1) and (length >= 3):
                    p2_score += length**2
        return p1_score, p2_score

    def utility(self):
        """Return the utility of the state as determined by score().
        
        The utility is from the perspective of Player 1; i.e., when it is positive, Player 1 wins;
        when negative, Player 2 wins.
        """
        s1, s2 = self.scores()
        return s1 - s2

    def __str__(self):
        symbols = { -1: "O", 1: "X", 0: "-", -2: "#" }
        s = ""
        for r in range(self.num_rows-1, -1, -1):
            s += "\n"
            for c in range(self.num_cols):
                s += "  " + symbols[self.board[r][c]]

        s += "\n  " + "." * (self.num_cols * 3 - 2) + "\n"
        for c in range(self.num_cols):
            s += "  " + str(c)
        s += "\n"
        return s


def streaks(lst):  
    """Get the lengths of all the streaks of the same element in a sequence."""
    rets = []  # list of (element, length) tuples
    prev = lst[0]
    curr_len = 1
    for curr in lst[1:]:
        if curr == prev:
            curr_len += 1
        else:
            rets.append((prev, curr_len))
            prev = curr
            curr_len = 1
    rets.append((prev, curr_len))
    return rets


def play_game(player1, player2, state):
    """Run a Connect383 game.

    Player objects can be of any class that defines a get_move(state, depth) method that returns
    a move, state tuple.
    """
    print(state)

    turn = 0
    p1_state_count, p2_state_count = 0, 0

    while not state.is_full():
        player = player1 if state.next_player() == 1 else player2

        state_count_before = GameState.state_count
        move, state_next = player.get_move(state)
        state_count_after = GameState.state_count

        states_created = state_count_after - state_count_before
        if state.next_player() == 1:
            p1_state_count += states_created
        else:
            p2_state_count += states_created        

        print("Turn {}:".format(turn))        
        # print("Player {} generated {} states".format(1 if state.next_player() == 1 else 2, states_created))
        print("Player {} moves to column {}".format(1 if state.next_player() == 1 else 2, move))
        print(state_next)
        print("Current score is:", state_next.scores(), "\n\n")
       
        turn += 1
        state = state_next

    score1, score2 = state.scores()
    if score1 > score2:
        print("Player 1 wins! {} - {}".format(score1, score2))
    elif score1 < score2:
        print("Player 2 wins! {} - {}".format(score1, score2))
    else:
        print("It's a tie. {} - {}".format(score1, score2))
    print("Player 1 generated {} states".format(p1_state_count))
    print("Player 2 generated {} states".format(p2_state_count))
    print("")
    return score1, score2


#############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('play1', help="Player 1 type { random, human, mini, lookN, prune, altN }")
    parser.add_argument('play2', help="Player 2 type { random, human, mini, lookN, prune, altN }")
    parser.add_argument('brd', help="Board (valid tag or specified RxC)")
    args = parser.parse_args()
    # print("args:", args)  

    player1 = agents.get_agent(args.play1)
    player2 = agents.get_agent(args.play2)
    start_state = GameState(boards.get_board(args.brd))
    play_game(player1, player2, start_state)

