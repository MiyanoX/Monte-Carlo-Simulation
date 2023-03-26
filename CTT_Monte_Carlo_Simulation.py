import cv2
from Class_def import MonteCarlo, img, data_num
import numpy as np
from CTT_Strength_New import demo
import time


##############
# MonteCarlo #
##############

def simulation(sim_times):
    f = open('data/result.txt', 'a')
    f.write('simulation start\n')
    for i in range(sim_times):
        print('simulation', i+1, '...')
        # simulation
        sim = MonteCarlo()
        sim.axis_intersect('y', data_num)
        # sim.draw(1)
        for j in range(1, data_num+1):
            np.save("data/Ctt_data_{}.npy".format(j), sim.data(j))

        # draw
        # cv2.namedWindow("image")
        # cv2.imshow('image', img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        # strength calculation
        res = demo(data_num)
        f.write(str(res) + '\n')
    f.write('simulation end\n')
    f.close()


t = time.time()
simulation(1)
print("Runnning time:", time.time()-t)



