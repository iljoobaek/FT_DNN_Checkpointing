import tag_layer

@tag_layer.tag_fn
def test_client_work(msg):
    print("Starting client work:", msg)
    import time
    time.sleep(2)
    print("Finished client work.")


if __name__ == "__main__":
    # Testing decorator
    test_client_work("testing ...")

