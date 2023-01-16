# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:55:35 2023

@author: Admin
"""


from BrukerMRI import *
import pylab as pl
import numpy as np
import re



MainDir = "C:\\Users\\Admin\\Documents\\Bruker_Python\\20220811_134118_Tvrdik_CCM_20220811_MCAO_1_1_1_2"

ExpNum = 16
# load FLASH experiment and reconstruct raw data
Experiment = ReadExperiment(MainDir, ExpNum)

method = ReadParamFile(os.path.join(MainDir, str(ExpNum),'method'))
reco   = ReadParamFile(os.path.join(MainDir, str(ExpNum), "pdata","1","reco"))
acqp   = ReadParamFile(os.path.join(MainDir, str(ExpNum), "acqp" ))



img_size = method["PVM_Matrix"]["value"]


for k in ['ACQ_jobs','ACQ_ScanPipeJobSettings','ACQ_jobs_size','ACQ_ReceiverSelectPerChan', 'BYTORDA']:
    print(k, acqp[k])

# Remake Raw
NI       = acqp['NI']
NR       = acqp['NR']
ACQ_size = acqp['ACQ_size']['value']
numDataHighDim       = np.prod(ACQ_size[1:])
numSelectedReceivers = acqp['ACQ_ReceiverSelectPerChan']['value'].count('Yes')
nRecs = numSelectedReceivers

X  = raw_data = ReadRawData(MainDir + '\\' + str(ExpNum) + "\\rawdata.job0")
jobScanSize = acqp['ACQ_jobs']['value'][0]
dim1= int(X.size / (jobScanSize*nRecs));
        
X = np.reshape(X,[jobScanSize, nRecs, dim1],  order='F');
X = np.transpose(X, (1, 0, 2)) 

X = X[:,0::2,:] + 1j * X[:,1::2,:]


# Reorder into frame
ACQ_obj_order = acqp['ACQ_obj_order']['value']
ACQ_dim = acqp['ACQ_dim']
ACQ_phase_factor = acqp['ACQ_phase_factor']
scanSize=ACQ_size[0]

data = np.reshape(X, (4, 128, 128, -1))

pl.figure()
pl.imshow(np.abs(np.fft.fft(data[0,:,:,2])))

# data=np.reshape(X, (numSelectedReceivers, scanSize, ACQ_phase_factor, NI, int(numDataHighDim/ACQ_phase_factor),NR));
# % Save to output variable:
# % convert to complex:
# X=complex(X(:,1:2:end,:), X(:,2:2:end,:));


# fidFile=np.zeros(
#     ( num_rows, NR*num_columns)
# )

 
# with open(MainDir + '\\' + str(ExpNum) + "\\rawdata.job0", "rb") as f:
#     byte = f.read(1)
#     while byte != b"":
#         byte = f.read(1)
#         total_bytes += 1 
#         data.append(byte)
        
# 
# raw_data = raw_data.newbyteorder('l')
# print(raw_data.dtype.byteorder)
# print(raw_data[0::2]) #+ 1j * raw_data[1::2]

