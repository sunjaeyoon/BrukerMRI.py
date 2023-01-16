#  pylint: disable-msg=C0103
"""This should at some point be a library with functions to import and
reconstruct Bruker MRI data.
2014, Joerg Doepfert
"""


import numpy as np
import os
import re
# ***********************************************************
#  class definition
# ***********************************************************
class BrukerData(object):
    """Class to store and process data of a Bruker MRI Experiment"""
    def __init__(self, path="", ExpNum=0, B0=9.4):
        self.method = {}
        self.acqp = {}
        self.reco = {}

        self.raw_fid = np.array([])
        self.proc_data = np.array([])
        self.k_data = np.array([])
        self.reco_data = np.array([])
        self.reco_data_norm = np.array([]) # normalized reco

        self.B0 = B0 # only needed for UFZ method
        self.GyroRatio = 0 # only needed for UFZ method
        self.ConvFreqsFactor = 0 # reference to convert Hz <--> ppm
        self.path = path
        self.ExpNum = ExpNum

# class BrukerData end **************************************




# ***********************************************************
#  Functions
# ***********************************************************
def ReadExperiment(path, ExpNum):
    """Read in a Bruker MRI Experiment. Returns raw data, processed 
    data, and method and acqp parameters in a dictionary.
    """
    data = BrukerData(path, ExpNum)

    # parameter files
    data.method = ReadParamFile(path + '\\' + str(ExpNum) + "\\method")
    data.acqp   = ReadParamFile(path + '\\' + str(ExpNum) + "\\acqp")
    data.reco   = ReadParamFile(path + '\\' + str(ExpNum) + "\\pdata\\1\\reco")

    # processed data
    data.proc_data = ReadProcessedData(path + '\\' + str(ExpNum) + "\\pdata\\1\\2dseq",
                                       data.reco,
                                       data.acqp)

    # generate complex FID
    raw_data = ReadRawData(path + '\\' + str(ExpNum) + "\\rawdata.job0")
    data.raw_fid = raw_data[0::2] + 1j * raw_data[1::2]

    # calculate GyroRatio and ConvFreqsFactor
    data.GyroRatio = data.acqp["SFO1"]*2*np.pi/data.B0*10**6 # in rad/Ts
    data.ConvFreqsFactor = 1/(data.GyroRatio*data.B0/10**6/2/np.pi)

    data.path = path
    data.ExpNum =ExpNum

    return data


def CalcOptNEchoes(s):
    """Find out how many echoes in an echo train [s] have to be
    included into an averaging operation, such that the signal to 
    noise (SNR) of the resulting averaged signal is maximized. 
    Based on the formula shown in the supporting information of
    the [Doepfert et al. ChemPhysChem, 15(2), 261-264, 2014]
    """

    # init vars
    s_sum = np.zeros(len(s))
    s_sum[0] = s[0]

    TestFn = np.zeros(len(s)) 
    SNR_averaged = np.zeros(len(s)) # not needed for calculation
    count = 1

    for n in range(2, len(s)+1):
        SNR_averaged = np.sum(s[0:n] / np.sqrt(n))
        s_sum[n-1] = s[n-1] + s_sum[n-2]
        TestFn[n-1] = s_sum[n-2]*(np.sqrt(float(n)/(float(n)-1))-1)
        if s[n-1] < TestFn[n-1]:
            break
        count += 1

    return count

def FFT_center(Kspace, sampling_rate=1, ax=0):
    """Calculate FFT of a time domain signal and shift the spectrum
    so that the center frequency is in the center. Additionally
    return the frequency axis, provided the right sampling frequency
    is given.
    If the data is 2D, then the FFT is performed succesively along an
    axis [ax].
    """
    FT = np.fft.fft(Kspace, axis=ax)
    spectrum = np.fft.fftshift(FT, axes=ax)
    n = FT.shape[ax]
    freq_axis = np.fft.fftshift(
        np.fft.fftfreq(n, 1/float(sampling_rate)))

    return spectrum, freq_axis

def fft_image(Kspace):
    
    return np.fft.fftshift(np.fft.fft2(Kspace))


def RemoveVoidEntries(datavector, acqsize0):
    blocksize = int(np.ceil(float(acqsize0)/2/128)*128)

    DelIdx = []
    for i in range(0, len(datavector)/blocksize):
        DelIdx.append(range(i * blocksize
                            + acqsize0/2,
                            (i + 1) * blocksize))
    return  np.delete(datavector, DelIdx)

def ReadRawData(filepath):
    with open(filepath, "r") as f:
        return np.fromfile(f, dtype=np.int32)


def ReadProcessedData(filepath, reco, acqp):
    with open(filepath, "r") as f:
        data = np.fromfile(f, dtype=np.int16)
        
        data = data.reshape(reco["RECO_size"]["value"][0],
                             reco["RECO_size"]["value"][1], -1, order="F")
        if data.ndim == 3:
            data_length = data.shape[2]
        else:
            data_length = 1

        data_reshaped = np.zeros([data.shape[1], data.shape[0], data_length])
        for i in range(0, data_length):
            data_reshaped[:, :, i] = np.rot90(data[:, :, i])

        return data_reshaped



def ReadParamFile(filepath):
    """
    Read a Bruker MRI experiment's method or acqp file to a
    dictionary.
    """
    param_dict = {}
    current_param = None
    with open(filepath, "r") as f:
        
        for line in f:
            
            # End flag Exit Loop
            if line.startswith('##END='):    
                break
            
            # when line contains parameter name
            if line.startswith('##'):
                
                # Parameters
                if line.startswith('##$'):
                    (param_name, current_line) = line[3:].split('=') # split at "="
        
                    # if current entry (current_line) is arraysize
                    if current_line[0:2] == "( " and current_line[-3:-1] == " )":  
                    
                        current_param = param_name 
                        
                        param_dict[param_name] = dict()
                        array_size = current_line.strip('\n').strip('(').strip(')').split(',')
                        array_size = np.array([int(x) for x in array_size])
                        param_dict[param_name]['size'] = array_size
                        
                        value = [] # placeholder for now
                        param_dict[param_name]['value'] = value
                    elif current_line[0] == "(" and current_line[-3:-1] != " )":
                        # if neccessary read in multiple lines
                        while current_line[-2] != ")":
                            current_line = current_line[0:-1] + f.readline()
                        param_dict[param_name] = current_line
                    else:
                        param_dict[param_name] = current_line
                
                # Global Parameters 
                else:
                    (param_name, current_line) = line[2:].split('=') # split at "="
                    param_dict[param_name] = current_line
                
            
            # Comment in file
            elif line.startswith('$$'):
                pass
            
            # line has data, add to current param value list
            else:
                if current_param != None:
                    param_dict[current_param]['value'].append(line)  
    
    
    # Change strings into proper usable values
    for key, value in param_dict.items():
        
        if type(value) == dict:
            #print(key)
            param_dict[key]['value'] = ParseArray("".join(value['value']),value['size'])
            #if key == 'ACQ_RfShapes': break
        elif type(value) == str:
            if value.strip().startswith('(') and value.strip().endswith(')'):
                param_dict[key] = ParseSingleList(value)
            else:
                param_dict[key] = ParseSingleValue(value)
    
    return param_dict


def ParseArray(current_line, arraysize):
    
    # Check if data is a bracket of info
    if arraysize.ndim == 1 and arraysize[0] > 1 and current_line.startswith('('):
        current_line = current_line.replace('\n', '')
        #print(current_line)
        return current_line
    
    # Split into section
    vallist = current_line.split()
    #print(vallist)
    # Find and Expand @x*(y) 
    vallist = parse_amps(vallist)
    #print(vallist)
    
    # Rejoin string and split again
    vallist = " ".join(vallist).strip().replace("  ", " ")
    #print(vallist)
    vallist = vallist.split(' ')
    #print(vallist)
    # if the line was a string, then return it directly
    try:
        float(vallist[0])
    except ValueError:
        out = " ".join(vallist)
        if out.startswith('(') and out.endswith(')') and arraysize.size == 1:
             out = ParseSingleList(out)
           
        return out
    
    # try converting to int, if error, then to float
    #print(vallist)
    try:
        vallist = [int(x) for x in vallist]
    except ValueError:
        vallist = [float(x) for x in vallist]
    # convert to numpy array
    if len(vallist) > 1:
        return np.reshape(np.array(vallist), arraysize)
    # or to plain number
    else:
        return vallist[0]
    
    
def parse_amps(vallist):
    out = []
    #print(vallist)
    for item in vallist:
        #print('\n')
        #print(item)
        x = re.search('@[0-9]+\*\([^)]*\)',item)#"\@.+\*\(.+\)",item)
        if x != None:
            #print(x.group())
            replacement = regex_rule(x.group())
            replacement = item.replace(x.group(), replacement)
            #print(replacement)
        else:
            replacement = item
            
        out.append(replacement)
                 
    return out
                
def regex_rule(string):
    s = string.strip().split('*')
    try:
        m = int(s[0].strip('@'))        
        v = s[1].replace('(', ""). replace(')'," ")
        return (m*v).strip()
    except Exception as e:
        print(e)

def ParseSingleValue(val):
    try: # check if int
        result = int(val)
    except ValueError:
        try: # then check if float
            result = float(val)
        except ValueError:
            # if not, should  be string. Remove  newline character.
            result = val.rstrip('\n').strip()

    return result

def ParseSingleList(val):
    vallist = val.strip().strip('(').strip(')')
    
    # If a list inside a list
    if '(' in vallist:
        return val
    else:
        vallist = vallist.split(',')
        vallist = [ParseSingleValue(i) for i in vallist]
        return vallist


# ***********************************************************
# -----------------------------------------------------------
# ***********************************************************


if __name__ == '__main__':
    # MainDir = "C:\\Users\\Admin\\Documents\\Bruker_Python\\20220811_134118_Tvrdik_CCM_20220811_MCAO_1_1_1_2"

    # ExpNum = 1

    # method = ReadParamFile(os.path.join(MainDir, str(ExpNum),'method'))
    # reco   = ReadParamFile(os.path.join(MainDir, str(ExpNum), "pdata","1","reco"))
    # acqp   = ReadParamFile(os.path.join(MainDir, str(ExpNum), "acqp" ))
    
    #Experiment = ReadExperiment(MainDir, ExpNum)
    
    
    pass

    
