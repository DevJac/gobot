if __name__ == '__main__':
    import gobot
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        gobot.compete(
            p1_model_file='model.1.h5',
            p2_model_file='model.2.h5',
            board_size=9)
