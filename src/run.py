from main.com.pavlof.deeplearning.tensorflow import lesson_p_1 as less1
from main.com.pavlof.deeplearning.tensorflow import lesson_p_2_doggs_and_cats as less2
from main.com.pavlof.deeplearning.tensorflow import lesson_p_3_convolutional_neural_networks as less3
import os

BASE_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(os.path.sep, BASE_DIR, 'res', 'pets')


def run():
    comannd = str(input("Print lessom number (p\"number\"): "))

    if comannd == "p1":
        less1.run()
    elif comannd == "p2":
        less2.run(DATA_DIR)
    elif comannd == "p3":
        less3.run(DATA_DIR)
    elif comannd == "q":
        pass
    else:
        print("Wrong lesson")
        run()


run()
