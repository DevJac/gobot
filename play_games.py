if __name__ == '__main__':
    import gobot
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        gobot.play_games(20, 13, verbose=False)
