## Pygame Sudoku

**Project description:** An early project, this is a fully featured sudoku game made entirely in Pygame. It features difficulty settings, a handmade UI, hint generation, and random puzzle generation.

### Implementation
```python
import sys
import pygame
import random
import copy
import time

white = (255, 255, 255)
gray = (200, 200, 200)
darkgray = (100, 100, 100)
black = (0, 0, 0)
red = (200, 0, 0)
brown = (150, 90, 40)


def button(text, x, y, w, h, color):
    '''Creates a rectangle with text inside it for interactivity, and displays
    it.'''
    button = pygame.draw.rect(screen, color, (x, y, w, h))
    cellfont = pygame.font.Font("freesansbold.ttf", 20)
    msg = cellfont.render(text, True, black)
    msg_box = msg.get_rect()
    msg_box.center = ((x+(w/2)), (y+(h/2)))
    screen.blit(msg, msg_box)
    return button


def make_board(m=3):
    '''Creates a randomly generated solved 9x9 sudoku puzzle using recursion
    and backtracking'''
    n = m**2
    board = [[None for _ in range(n)] for _ in range(n)]

    def search(c=0):
        '''Recursively searches for a solution starting at position c'''
        i, j = divmod(c, n)
        i0, j0 = i - i % m, j - j % m  # Origin of mxm block
        numbers = list(range(1, n + 1))
        random.shuffle(numbers)
        for x in numbers:
            if (x not in board[i]                     # row
                and all(row[j] != x for row in board)  # column
                and all(x not in row[j0:j0+m]         # block
                        for row in board[i0:i])):
                board[i][j] = x
                if c + 1 >= n**2 or search(c + 1):
                    return board
        else:
            # No valid number can be placed, backtrack and try again
            board[i][j] = None
            return None

    return search()


def make_puzzle(diff):
    '''Initiates the game, generating a random solved puzzle using the
    make_board algorithm, then removes random cells to create an unsolved
    puzzle, along with its formatting.'''
    global puzzle
    solution = make_board()
    to_remove = []
    for i in range(diff):
        to_remove.append(1)
    for i in range((9**2)-diff):
        to_remove.append(0)
    random.shuffle(to_remove)
    unsolved_board = copy.deepcopy(solution)
    k = 0
    for i in range(9):
        for j in range(9):
            if to_remove[k] == 1:
                unsolved_board[i][j] = None
            k += 1
    puzzle = {}
    for i in range(9):
        for j in range(9):
            cellposx = 2 + (j * (cellwidth + 4))
            cellposy = 2 + (i * (cellwidth + 4))
            puzzle[i, j] = {}
            puzzle[i, j]["answer"] = "%d" % solution[i][j]
            puzzle[i, j]["row"] = "%d" % i
            puzzle[i, j]["column"] = "%d" % j
            puzzle[i, j]["block"] = "%d, %d" % ((i - i % 3), (j - j % 3))
            if unsolved_board[i][j]:
                puzzle[i, j]["value"] = "%d" % unsolved_board[i][j]
            else:
                puzzle[i, j]["value"] = ""
            puzzle[i, j]["rect"] = pygame.Rect(cellposx, cellposy,
                                               cellwidth, cellheight)
            if puzzle[i, j]["value"]:
                puzzle[i, j]["msg_value"] = cellfont.render(puzzle[i, j]["value"],
                                                            True, black)
                puzzle[i, j]["msg_value_box"] = puzzle[i, j]["msg_value"].get_rect()
                puzzle[i, j]["msg_value_box"].center = puzzle[i, j]["rect"].center
                puzzle[i, j]["notes_allowed"] = False
                puzzle[i, j]["modifiable"] = False
            else:
                puzzle[i, j]["notes_allowed"] = True
                puzzle[i, j]["modifiable"] = True
                for k in range(1, 10):
                    puzzle[i, j]["note%d" % k] = False
                    puzzle[i, j]["msg_note%d" % k] = notesfont.render("",
                                                                      True,
                                                                      brown)
                    puzzle[i, j]["msg_note%d_box" % k] = puzzle[i, j]["msg_note%d" % k].get_rect()
            if (2 < j < 6 and
               (i < 3 or i > 5)) or (2 < i < 6 and
               (j < 3 or j > 5)):
                puzzle[i, j]["color"] = gray
            else:
                puzzle[i, j]["color"] = white


def number_buttons(num_buttons):
    '''Creates the number buttons and their positions in the UI'''
    i = 0
    for num in num_buttons.values():
        i += 1
        if i < 5:
            if num_selected == num["num"]:
                num["button"] = button(num["num"], ui_rect.left + ui_width / 4,
                                       i * (ui_height / 9), ui_width / 5,
                                       ui_height / 14, darkgray)
            else:
                num["button"] = button(num["num"], ui_rect.left + ui_width / 4,
                                       i * (ui_height / 9), ui_width / 5,
                                       ui_height / 14, gray)
        elif i < 9:
            if num_selected == num["num"]:
                num["button"] = button(num["num"], ui_rect.left + ui_width / 2,
                                       (i - 4) * (ui_height / 9), ui_width / 5,
                                       ui_height / 14, darkgray)
            else:
                num["button"] = button(num["num"], ui_rect.left + ui_width / 2,
                                       (i - 4) * (ui_height / 9), ui_width / 5,
                                       ui_height / 14, gray)
        else:
            if num_selected == num["num"]:
                num["button"] = button(num["num"],
                                       ui_rect.left + ui_width / 2.6,
                                       (i - 4) * (ui_height / 9), ui_width / 5,
                                       ui_height / 14, darkgray)
            else:
                num["button"] = button(num["num"],
                                       ui_rect.left + ui_width / 2.6,
                                       (i - 4) * (ui_height / 9), ui_width / 5,
                                       ui_height / 14, gray)
    return num_buttons


def modify_cell(cell, newvalue):
    '''Whenever a cell is modified, this changes its value and if any notes
    are within the same row, column, or block as the cell, those will be
    modified as according to sudoku rules'''
    cell["value"] = newvalue
    cell["msg_value"] = cellfont.render(cell["value"],
                                        True, brown)
    cell["msg_value_box"] = cell["msg_value"].get_rect()
    cell["msg_value_box"].center = cell["rect"].center
    if newvalue:
        for othercell in puzzle.values():
            if (othercell["notes_allowed"]
                and (othercell["row"] == cell["row"]
                     or othercell["column"] == cell["column"]
                     or othercell["block"] == cell["block"])):
                    if othercell["note%s" % newvalue]:
                        take_notes(othercell, newvalue)
        cell["notes_allowed"] = False
    else:
        cell["notes_allowed"] = True
    return cell


def take_notes(cell, num):
    '''Take_notes() adds or removes a note on a cell and is responsible for
    the formatting of notes within the cell.'''
    if num == "1":
        if not cell["note1"]:
            cell["msg_note1"] = notesfont.render("1", True, brown)
            cell["note1"] = True
        else:
            cell["msg_note1"] = notesfont.render("", True, brown)
            cell["note1"] = False
        cell["msg_note1_box"] = cell["msg_note1"].get_rect()
        cell["msg_note1_box"].topleft = cell["rect"].topleft
    elif num == "2":
        if not cell["note2"]:
            cell["msg_note2"] = notesfont.render("2", True, brown)
            cell["note2"] = True
        else:
            cell["msg_note2"] = notesfont.render("", True, brown)
            cell["note2"] = False
        cell["msg_note2_box"] = cell["msg_note2"].get_rect()
        cell["msg_note2_box"].midtop = cell["rect"].midtop
    elif num == "3":
        if not cell["note3"]:
            cell["msg_note3"] = notesfont.render("3", True, brown)
            cell["note3"] = True
        else:
            cell["msg_note3"] = notesfont.render("", True, brown)
            cell["note3"] = False
        cell["msg_note3_box"] = cell["msg_note3"].get_rect()
        cell["msg_note3_box"].topright = cell["rect"].topright
    elif num == "4":
        if not cell["note4"]:
            cell["msg_note4"] = notesfont.render("4", True, brown)
            cell["note4"] = True
        else:
            cell["msg_note4"] = notesfont.render("", True, brown)
            cell["note4"] = False
        cell["msg_note4_box"] = cell["msg_note4"].get_rect()
        cell["msg_note4_box"].midleft = cell["rect"].midleft
    elif num == "5":
        if not cell["note5"]:
            cell["msg_note5"] = notesfont.render("5", True, brown)
            cell["note5"] = True
        else:
            cell["msg_note5"] = notesfont.render("", True, brown)
            cell["note5"] = False
        cell["msg_note5_box"] = cell["msg_note5"].get_rect()
        cell["msg_note5_box"].center = cell["rect"].center
    elif num == "6":
        if not cell["note6"]:
            cell["msg_note6"] = notesfont.render("6", True, brown)
            cell["note6"] = True
        else:
            cell["msg_note6"] = notesfont.render("", True, brown)
            cell["note6"] = False
        cell["msg_note6_box"] = cell["msg_note6"].get_rect()
        cell["msg_note6_box"].midright = cell["rect"].midright
    elif num == "7":
        if not cell["note7"]:
            cell["msg_note7"] = notesfont.render("7", True, brown)
            cell["note7"] = True
        else:
            cell["msg_note7"] = notesfont.render("", True, brown)
            cell["note7"] = False
        cell["msg_note7_box"] = cell["msg_note7"].get_rect()
        cell["msg_note7_box"].bottomleft = cell["rect"].bottomleft
    elif num == "8":
        if not cell["note8"]:
            cell["msg_note8"] = notesfont.render("8", True, brown)
            cell["note8"] = True
        else:
            cell["msg_note8"] = notesfont.render("", True, brown)
            cell["note8"] = False
        cell["msg_note8_box"] = cell["msg_note8"].get_rect()
        cell["msg_note8_box"].midbottom = cell["rect"].midbottom
    elif num == "9":
        if not cell["note9"]:
            cell["msg_note9"] = notesfont.render("9", True, brown)
            cell["note9"] = True
        else:
            cell["msg_note9"] = notesfont.render("", True, brown)
            cell["note9"] = False
        cell["msg_note9_box"] = cell["msg_note9"].get_rect()
        cell["msg_note9_box"].bottomright = cell["rect"].bottomright
    return cell


def give_hint(puzzle):
    '''If a hint is needed, a random cell which is either blank or incorrect
    will be modified to its correct value'''
    global hints_used
    irand, jrand = list(range(0, 9)), list(range(9))
    random.shuffle(irand)
    random.shuffle(jrand)
    for i in irand:
        for j in jrand:
            if puzzle[i, j]["value"] != puzzle[i, j]["answer"]:
                puzzle[i, j]["value"] = puzzle[i, j]["answer"]
                puzzle[i, j]["msg_value"] = cellfont.render(puzzle[i, j]["value"],
                                                            True, black)
                puzzle[i, j]["msg_value_box"] = puzzle[i, j]["msg_value"].get_rect()
                puzzle[i, j]["msg_value_box"].center = puzzle[i, j]["rect"].center
                puzzle[i, j]["notes_allowed"] = False
                puzzle[i, j]["modifiable"] = False
                hints_used += 1
                for othercell in puzzle.values():
                    if (othercell["notes_allowed"]
                        and (othercell["row"] == puzzle[i, j]["row"]
                             or othercell["column"] == puzzle[i, j]["column"]
                             or othercell["block"] == puzzle[i, j]["block"])):
                        if othercell["note%s" % puzzle[i, j]["value"]]:
                            take_notes(othercell, puzzle[i, j]["value"])
                return puzzle
    return puzzle


def check_values(puzzle):
    '''If the check button is clicked, all values entered by the player which
    are incorrect are marked with red'''
    global checks_used
    for cell in puzzle.values():
        if cell["value"] != cell["answer"]:
            cell["msg_value"] = cellfont.render(cell["value"],
                                                True, red)
            cell["msg_value_box"] = cell["msg_value"].get_rect()
            cell["msg_value_box"].center = cell["rect"].center
    checks_used += 1
    return puzzle


def solved():
    '''Solved() iterates through the puzzle to determine if it has been solved
    correctly, and returns a simple boolean'''
    for cell in puzzle.values():
        if cell["value"] != cell["answer"]:
            return False
    return True


def end_game():
    '''When the puzzle is solved correctly, the game ends, and the statistics
    are then calculated for display and the game_end condition is set'''
    global hints_msg, hints_msg_box, checks_msg, checks_msg_box
    global time_spent, time_spent_box
    game_started = False
    game_end = True
    time_end = time.time() - time_started
    time_end_m = time_end // 60
    time_end_s = time_end % 60
    hints_msg = cellfont.render("Hints used: %02d" % hints_used, True, brown)
    hints_msg_box = hints_msg.get_rect()
    hints_msg_box.center = (window_width / 9, window_height / 35)
    checks_msg = cellfont.render("Checks used: %02d" % checks_used, True, brown)
    checks_msg_box = checks_msg.get_rect()
    checks_msg_box.center = (window_width / 8, window_height / 13)
    time_spent = cellfont.render("Time: %02d:%02d" %
                                 (int(time_end_m), int(time_end_s)),
                                 True, brown)
    time_spent_box = time_spent.get_rect()
    time_spent_box.center = (window_width / 10.5, window_height / 7.75)
    return game_started, game_end


def game_handler():
    '''While the game is running, the three input methods (notes, fill,
    and erase) can be set using either the keyboard or the mouse, as well
    as all the numbers. If a cell is clicked on, the game modifies the cell
    as appropriate to the selected input method and number'''
    global notes_selected, fill_selected, erase_selected, num_selected
    global puzzle, intro, game_started
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = event.pos
                for cell in puzzle.values():
                    if cell["rect"].collidepoint(x, y):
                        if fill_selected:
                            if cell["modifiable"]:
                                cell = modify_cell(cell, num_selected)
                        if erase_selected:
                            if cell["modifiable"]:
                                cell = modify_cell(cell, "")
                        if notes_selected:
                            if cell["notes_allowed"]:
                                cell = take_notes(cell, num_selected)
                for num in num_buttons.values():
                    if num["button"].collidepoint(x, y):
                        num_selected = num["num"]
                if hint.collidepoint(x, y):
                    puzzle = give_hint(puzzle)
                if notes.collidepoint(x, y):
                    notes_selected = True
                    fill_selected = False
                    erase_selected = False
                if fill.collidepoint(x, y):
                    notes_selected = False
                    fill_selected = True
                    erase_selected = False
                if erase.collidepoint(x, y):
                    notes_selected = False
                    fill_selected = False
                    erase_selected = True
                if check.collidepoint(x, y):
                    puzzle = check_values(puzzle)
                if quit_game.collidepoint(x, y):
                    game_started = False
                    intro = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                num_selected = "1"
            if event.key == pygame.K_2:
                num_selected = "2"
            if event.key == pygame.K_3:
                num_selected = "3"
            if event.key == pygame.K_4:
                num_selected = "4"
            if event.key == pygame.K_5:
                num_selected = "5"
            if event.key == pygame.K_6:
                num_selected = "6"
            if event.key == pygame.K_7:
                num_selected = "7"
            if event.key == pygame.K_8:
                num_selected = "8"
            if event.key == pygame.K_9:
                num_selected = "9"
            if event.key == pygame.K_n:
                notes_selected = True
                fill_selected = False
                erase_selected = False
            if event.key == pygame.K_f:
                notes_selected = False
                fill_selected = True
                erase_selected = False
            if event.key == pygame.K_e:
                notes_selected = False
                fill_selected = False
                erase_selected = True


def end_handler():
    '''When the game ends, the option is given to play again, in which
    case a new puzzle is generated, or to return to the menu.'''
    global game_end, game_started, intro, hints_used, checks_used, time_started
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = event.pos
                if play_again.collidepoint(x, y):
                    hints_used = 0
                    checks_used = 0
                    make_puzzle(diff)
                    game_end = False
                    game_started = True
                    time_started = time.time()
                if to_menu.collidepoint(x, y):
                    game_end = False
                    intro = True


def intro_handler():
    '''When first starting the game, several difficulty options are presented.
    These difficulty options are the number of cell values to be removed when
    turning the solved puzzle into an unsolved one. Their values can be found
    in the main code'''
    global game_started, intro, diff, time_started
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = event.pos
                if choose_easy.collidepoint(x, y):
                    diff = easy
                    make_puzzle(diff)
                    intro = False
                    game_started = True
                    time_started = time.time()
                if choose_normal.collidepoint(x, y):
                    diff = normal
                    make_puzzle(diff)
                    intro = False
                    game_started = True
                    time_started = time.time()
                if choose_hard.collidepoint(x, y):
                    diff = hard
                    make_puzzle(diff)
                    intro = False
                    game_started = True
                    time_started = time.time()
                if choose_expert.collidepoint(x, y):
                    diff = expert
                    make_puzzle(diff)
                    intro = False
                    game_started = True
                    time_started = time.time()
                if choose_extreme.collidepoint(x, y):
                    diff = extreme
                    make_puzzle(diff)
                    intro = False
                    game_started = True
                    time_started = time.time()


# Screens, fonts, and the UI area are declared here
pygame.init()
puzzle_height = 630
puzzle_width = 630
ui_height = 630
ui_width = 200
window_height = puzzle_height
window_width = puzzle_width + ui_width
screen = pygame.display.set_mode((window_width, window_height))
window = screen.get_rect()
ui_rect = pygame.Rect(puzzle_width, 0, ui_width, ui_height)
pygame.mouse.set_visible(True)
cellwidth = (puzzle_width/9 - 4)
cellheight = (puzzle_height/9 - 4)
cellfont = pygame.font.Font("freesansbold.ttf", 22)
notesfont = pygame.font.Font("freesansbold.ttf", 18)
title_font = pygame.font.Font("freesansbold.ttf", 60)
end_msg = title_font.render("You have won!", True, black)
end_msg_box = end_msg.get_rect()
end_msg_box.center = (window_width / 2.4, window_height / 2.5)
game_title = title_font.render("Sudoku Maya", True, black)
game_title_box = game_title.get_rect()
game_title_box.center = (window_width / 2, window_height / 3.5)

# The number buttons are initialized for each possible number (1-9)

num_buttons = {}
for i in range(1, 10):
    num_buttons[i] = {}
    num_buttons[i]["num"] = "%d" % i

# Here the default mode is set

num_selected = "1"
notes_selected = False
erase_selected = False
fill_selected = True

# Difficulty values are here, equal to the number of cell values to be removed.

easy = 20
normal = 30
hard = 40
expert = 50
extreme = 60

game_started = False
game_end = False
intro = True

fps = pygame.time.Clock()

while True:

    if intro:
        hints_used = 0
        checks_used = 0
        screen.fill(brown)
        screen.blit(game_title, game_title_box)
        choose_easy = button("Easy", window_width / 2.6, window_height / 2.5,
                             window_width / 4, window_height / 14, gray)
        choose_normal = button("Normal", window_width / 2.6, window_height / 2,
                               window_width / 4, window_height / 14, gray)
        choose_hard = button("Hard", window_width / 2.6, window_height / 1.65,
                             window_width / 4, window_height / 14, gray)
        choose_expert = button("Expert", window_width / 2.6,
                               window_height / 1.41, window_width / 4,
                               window_height / 14, gray)
        choose_extreme = button("Extreme", window_width / 2.6,
                                window_height / 1.23, window_width / 4,
                                window_height / 14, gray)
        intro_handler()

    elif game_started or game_end:
        screen.fill(black)
        pygame.draw.rect(screen, brown, ui_rect)
        for cell in puzzle.values():
            if cell["value"] == num_selected:
                pygame.draw.rect(screen, darkgray, cell["rect"])
            else:
                pygame.draw.rect(screen, cell["color"], cell["rect"])
            if cell["value"]:
                screen.blit(cell["msg_value"], cell["msg_value_box"])
            if cell["notes_allowed"]:
                for k in range(1, 10):
                    screen.blit(cell["msg_note%d" % k],
                                cell["msg_note%d_box" % k])

        num_buttons = number_buttons(num_buttons)

        hint = button("Hint", ui_rect.left + ui_width / 4.5, ui_height / 40,
                      ui_width / 2, ui_height / 16, gray)
        if notes_selected:
            notes = button("Notes", ui_rect.left + ui_width / 4.5,
                           ui_height / 1.55, ui_width / 2, ui_height / 16,
                           darkgray)
        else:
            notes = button("Notes", ui_rect.left + ui_width / 4.5,
                           ui_height / 1.55, ui_width / 2, ui_height / 16,
                           gray)
        if fill_selected:
            fill = button("Fill", ui_rect.left + ui_width / 4.5,
                          ui_height / 1.4, ui_width / 2, ui_height / 16,
                          darkgray)
        else:
            fill = button("Fill", ui_rect.left + ui_width / 4.5,
                          ui_height / 1.4, ui_width / 2, ui_height / 16, gray)
        if erase_selected:
            erase = button("Erase", ui_rect.left + ui_width / 4.5,
                           ui_height / 1.275, ui_width / 2, ui_height / 16,
                           darkgray)
        else:
            erase = button("Erase", ui_rect.left + ui_width / 4.5,
                           ui_height / 1.275, ui_width / 2, ui_height / 16,
                           gray)
        check = button("Check", ui_rect.left + ui_width / 4.5,
                       ui_height / 1.165, ui_width / 2, ui_height / 16, gray)
        quit_game = button("Quit", ui_rect.left + ui_width / 4.5,
                           ui_height / 1.075, ui_width / 2, ui_height / 16,
                           gray)

        if game_started:
            if solved():
                game_started, game_end = end_game()
            game_handler()
        else:
            screen.blit(end_msg, end_msg_box)
            screen.blit(hints_msg, hints_msg_box)
            screen.blit(checks_msg, checks_msg_box)
            screen.blit(time_spent, time_spent_box)
            play_again = button("Play again", window_width / 4,
                                window_height / 2, window_width / 3,
                                window_height / 12, brown)
            to_menu = button("Back to menu", window_width / 4,
                             window_height / 1.6, window_width / 3,
                             window_height / 12, brown)
            end_handler()

    pygame.display.flip()
    fps.tick(200)


```
