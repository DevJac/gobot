if __name__ == '__main__':
    import gobot
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        gobot.self_play(200, 9, verbose=False)
