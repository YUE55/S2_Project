import numpy as np
from epf_narx import epf_narx
from datetime import datetime
# read dataset
data = np.loadtxt('GEFCOM.txt')
# how many days in the dataset
n = len(data)//24
endd = 728  # first day to forecast
Ndays = 354  # forecasted days
res_avg = np.zeros((354*24,701))
# store start time
start_time = datetime.now()
# call the narx function
for startd in range(728-728,728-28+1,1): # 0-700
    res = np.zeros((Ndays*24,2))
    for i in range(2):
        # estimate and compute forecasts of the NARX model
        Res = epf_narx(data[:, :4], Ndays, startd, endd)
        res[:,i] = Res[:,3]
    res_avg[:,728-28-startd] = np.mean(res,axis=1)
# export forecast result
np.savetxt('narxoutput.csv', res_avg, delimiter=',')
# calculate the whole running time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

