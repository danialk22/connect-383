"""Test boards for Connect383

Place test boards in this module to help test your code.  Note that since connect383.GameState
stores board contents as a 0-based list of lists, these boards are reversed to they can be written
right side up here.

"""

boards = {}  # dictionary with label, test board key-value pairs

boards['test_1'] = [  
    [  0, 0, 0, 0 ],                       
    [  0, 0, 0, 0 ],
    [  0, -2, 0, 0 ] 
]

boards['test_2'] = [  
    [ -1,   0,  0, -1, -1 ],  
    [ -1,   1, -1,  1,  1 ],
    [  1,  -1,  1, -1,  1 ],
    [  1,   1, -1,  1, -1 ] 
]

boards['writeup_1'] = [
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0, -1,  0,  0,  0 ],
    [  0,  0,  0,  1,  0,  0,  0 ],
    [  0,  1,  0, -1,  0,  1,  0 ]
]

boards['writeup_2'] = [  
    [ -1,  1, -1, -1 ],                       
    [  1, -1,  1, -1 ],
    [  1, -2, -1,  1 ],
    [  1, -2,  1, -1 ] 
]

boards['tournament'] = [
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0,  0,  0,  0,  0 ],
    [  0,  0,  0, -2,  0,  0,  0 ],
    [  0,  0,  0, -2 , 0,  0,  0 ],
    [  0, -2,  0, -2,  0,  0,  0 ]
]

boards['your_test'] = [
    [  -1,   0,  0,  0,   0,  0,  0 ],
    [   1,  -1,  0,  1,   0,  0,  0 ],
    [   1,   1,  0, -1,   0,  0,  0 ],
    [  -1,  -1,  0, -2,   0,  0,  0 ],
    [  -1,   1,  0, -2,   0,  0,  0 ],
    [   1,  -2,  0, -2,  0,  0,  0 ]
]  # put something here!

boards['your_tester'] = []  # put something here!

boards['your_testest'] = []  # put something here!


def get_board(tag):
    if tag in boards:
        return list(reversed(boards[tag]))  # reversed so the y-coordinates read upward
    else:
        try:
            rowstr, colstr = tag.lower().split('x')
            rows = int(rowstr)
            cols = int(colstr)
            return [ [0]*cols ] * rows

        except ValueError:
            pass
        raise ValueError("bad board tag: '{}'".format(tag))       
