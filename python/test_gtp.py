# from gtp import Game, letter_to_int, int_to_letter
#
#
# def test_letter_conversion():
#     conversions = [
#         ('A', 1),
#         ('H', 8),
#         ('J', 9),
#         ('Z', 25),
#     ]
#     for l, i in conversions:
#         assert letter_to_int(l) == i
#         assert int_to_letter(i) == l
#
#
# def test_genmove():
#     g = Game()
#     g.command_boardsize(19)
#     g.command_clear_board('')
#     g.command_genmove('B')
