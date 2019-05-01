if __name__ == '__main__':
    import gobot
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        gobot.play_games(1000, 9, train_frequency=20, verbose=True)
