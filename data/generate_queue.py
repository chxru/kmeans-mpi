import random
import numpy as np
import redis
import time

r = redis.Redis(host="localhost", port=6379, db=0)


def push_random_values_to_queue() -> None:
    while True:
        X = np.random.rand(100, 2)

        # loop over each row and push it to queue
        for x in X:
            r.rpush("queue", x.tobytes())

        # r.rpush("queue", X.tobytes())

        print("Pushing random value to queue")

        # sleep random time between 1 and 5 seconds
        time.sleep(random.randint(1, 5))


if __name__ == "__main__":
    push_random_values_to_queue()
