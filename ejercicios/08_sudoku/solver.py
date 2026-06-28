# Backtracking para resolver sudokus.
# solve(board) devuelve la solución o None si no tiene.

import numpy as np


def _candidates(board, r, c):
    used = set(board[r, :]) | set(board[:, c])
    br, bc = 3*(r//3), 3*(c//3)
    used |= set(board[br:br+3, bc:bc+3].flatten())
    return [v for v in range(1, 10) if v not in used]


# límite de pasos para no bloquearse con tableros mal leídos sin solución
MAXSTEPS = 3000


class _GiveUp(Exception):
    pass


def _solve(board, empties, steps):
    steps[0] += 1
    if steps[0] > MAXSTEPS:
        raise _GiveUp
    if not empties:
        return True
    # elegimos la casilla con menos candidatos (mejora mucho el backtracking)
    best_i, best_cands = None, None
    for i, (r, c) in enumerate(empties):
        cands = _candidates(board, r, c)
        if best_cands is None or len(cands) < len(best_cands):
            best_i, best_cands = i, cands
            if len(cands) == 0:
                return False
            if len(cands) == 1:
                break

    r, c = empties[best_i]
    rest = empties[:best_i] + empties[best_i+1:]
    for v in best_cands:
        board[r, c] = v
        if _solve(board, rest, steps):
            return True
        board[r, c] = 0
    return False


def solve(board):
    b = np.array(board, dtype=int).copy()
    empties = [(r, c) for r in range(9) for c in range(9) if b[r, c] == 0]
    try:
        if _solve(b, empties, [0]):
            return b
    except _GiveUp:
        pass
    return None
