from main.com.pavlof.deeplearning.tensorflow import lesson_p_2_doggs_and_cats as less2
import os

BASE_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(os.path.sep, BASE_DIR, 'res', 'pets')

less2.run(DATA_DIR)
