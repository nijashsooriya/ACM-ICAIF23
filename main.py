import pickle
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
import Diffusion
x = np.linspace(0,1000)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open('val_data.pkl', 'rb') as f:
        data = pickle.load(f)
    np_data = np.asarray(data)
    asset1_price_path  = [[]]
    for i in range(len(data)):
        asset1_price_path.append(data[i][:,0])
    print("The asset 1 price is \n")
    print(asset1_price_path)