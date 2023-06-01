import os

from data import EquityData

if __name__ == '__main__':
    if not os.path.isfile('database.db'):
        EquityData().update_data()