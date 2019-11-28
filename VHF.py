#!/usr/bin/env python
# coding: utf-8




import numpy as np
import tables
import matplotlib.pyplot as plt

# # Конвертирование бинарного файла в несколько сжатых HDF5 файлов
HDF5_EVENTS_AMOUNT = 50
FILES_NUM=2


dtype = np.dtype(
    [('id', "i4"),
     ('event', np.uint32),
     ('x', np.double),
     ('y', np.double),
     ('z', np.double),
     ('time', np.double)])



            
 



def create_file(data, index, start_event):
    current_event = start_event
    id_dtype = np.dtype([("id","i4"), ("count", "i")])
    with tables.open_file("vhf_{}.hdf5".format(index), "w") as h5file:
        for j in range(HDF5_EVENTS_AMOUNT):
            group = h5file.create_group("/", "event_{}".format(current_event))
            event_data = data[data["event"] == current_event][["id", "x", "y", "z", "time"]]
            event_data = np.sort(event_data, order=["id", "time"])
            id_tracks, id_count = np.unique(event_data["id"], return_counts=True)
            table = h5file.create_table(group, "tracks", obj=event_data, filters = tables.Filters(complevel=3, fletcher32=True))
            table.flush()
#             print(id_tracks, id_count)
            id_tracks = np.array([(a,b) for a,b in zip(id_tracks, id_count)], dtype=id_dtype)
            table = h5file.create_table(group, "tracks_id", obj=id_tracks, filters = tables.Filters(complevel=3, fletcher32=True))
            table.flush()
            current_event+=1
    return current_event
    




# ## Вычисление электрического поля




LIGHT_SPEED = 0.3 # m/ns
ELECTRON_CHARGE = 1.6e-19 # SI
EPSILON0 = 1
COULOMB_CONST = 9e+9 # SI
class EMFieldCalculator:
    
    
    
    def __init__(self):
        self.norm = lambda x: np.linalg.norm(x, axis=1)
        self.cross = lambda x, y: np.cross(x, y, axisa=1, axisb=1)
        self.dot = lambda x, y: np.array([np.dot(i, j) for i, j in zip(x, y)])
    def get_n(self, r, r0):
      n = ((r - r0).T / self.norm(r - r0)).T
      return n
    
    def getElectricField(self, r, r0, velocity, acceleration):
        
        R = self.norm(r - r0)
        n = ((r - r0).T / R).T 
        beta = self.norm(velocity) / LIGHT_SPEED
        betaVec = velocity / LIGHT_SPEED
        betaAc =acceleration/ LIGHT_SPEED      # 1/ns 
        temp1 = 1 - beta ** 2 # NO UNIT
        temp1 =((n - betaVec).T * temp1.T).T # NO UNIT
        
        temp2 = R *((1 - self.dot(n, betaVec))) ** 3 # m
        
        firstTerm = (temp1.T/(R*temp2).T).T # 1/m^2
        
        temp3 = self.cross(n, self.cross(n - betaVec, betaAc)) # 1/ns
        secondTerm = (temp3.T/temp2.T).T / LIGHT_SPEED #1/m^2
        
        CONVERT_TO_SI = 1
        return CONVERT_TO_SI*(ELECTRON_CHARGE*COULOMB_CONST)*(firstTerm + secondTerm)


    
CALCULATOR = EMFieldCalculator()





def compute_velocity(track):  # mm/ns 
    diff_x = np.diff(track["x"])
    diff_y = np.diff(track["y"])
    diff_z = np.diff(track["z"])
    return ((np.array([diff_x, diff_y, diff_z])) / (np.diff(track["time"]))).T


def compute_acceleration(velocity, time): ## mm/ns^2
    velocity = velocity.T
    time = time[2:] - time[:-2]
    return (np.diff(velocity) / time).T





def process_track(track, observedPoint):
    outtype = np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
    if (track.size < 3):
        return np.zeros(1, outtype)
    velocity = compute_velocity(track)
    acceleration = compute_acceleration(velocity, track["time"])
    r0 = track[["x", "y", "z"]][1:-1]
    r0 = np.array([r0["x"],r0["y"],r0["z"]]).T
    temp = CALCULATOR.getElectricField(observedPoint, r0, velocity[:-1], acceleration)
    outcome = np.zeros(track.size - 2, outtype)
    outcome["time"] = track["time"][1:-1]+CALCULATOR.norm(observedPoint-r0)/LIGHT_SPEED
    outcome["x"] += temp[:,0]
    outcome["y"] += temp[:, 1]
    outcome["z"] += temp[: ,2]
    return outcome
    





def process_event(h5file,observedPoint, event_number=0, verbose=False):
    tracks = h5file.get_node("/event_{}".format(event_number), "tracks").read()
    id_tracks = h5file.get_node("/event_{}".format(event_number), "tracks_id").read()
   
    tracks_position = np.hstack((np.zeros(1), np.cumsum(id_tracks["count"])))
    result = []
    for i in range(id_tracks.size):
        if verbose:
            print("Process track id:", id_tracks[i])
        

        left, right = int(tracks_position[i]), int(tracks_position[i+1])      
        temp,mask=np.unique(tracks[left:right]['time'],return_index=True)
        temp=tracks[left:right]
        data = process_track(temp[mask],observedPoint)
        result.append(data)
    return result


def find_max_time(listData):
    max_time = 0 
    for item in listData:
        temp = item["time"][-1] 
        if (temp > max_time):
            max_time = temp
    return max_time





def join_track_signal(tracks, time_step = 1):
    max_time = find_max_time(tracks)
    grid = np.arange(0,max_time+time_step, time_step)
    value = np.zeros(grid.size-1, dtype=[("x", "d"),("y", "d"),("z", "d")])
    for track in tracks:
        index = np.searchsorted(grid, track["time"], side="left")
        left = index[0]
        if (index[0] < value.size):
            value["x"][index[0]] += track["x"][0]
            value["y"][index[0]] += track["y"][0]
            value["z"][index[0]] += track["z"][0]
        for i,j in enumerate(index[1:]):
            value["x"][left:j] += track["x"][i+1]
            value["y"][left:j] += track["y"][i+1]
            value["z"][left:j] += track["z"][i+1]
            left = j 
    return value, grid





def join_event_signal(events, time_step = 1):  
    max_size = 0
    for signal in events:
        temp = signal.size
        if temp>max_size:
            max_size=temp
            
    grid = np.arange(0,(max_size+1)*time_step, time_step)
    value = np.zeros(grid.size-1, dtype=[("x", "d"),("y", "d"),("z", "d")])
    for event in events:
        i = event.size
        value["x"][:i] += event["x"]
        value["y"][:i] += event["y"]
        value["z"][:i] += event["z"]
    return value, grid

def saveVectorBorders(xlabel:str,ylabel:str,title:str,arrayVec:np.ndarray,time:np.ndarray,delta:float):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
    left=np.argwhere(module>0)[0][0]
    right=np.argwhere(time<time[left]+delta)[-1][-1]
    np.savetxt('inpTime.txt',time)
    np.savetxt("inpVal.txt",module)    
    time=time[left-1:right]
    module=module[left:right]
        
    plt.plot(time[:-1]/1000,module)
    plt.savefig(title+".png")
    #plt.show()
    np.savetxt("bordTime.txt",time[:-1])
    np.savetxt("bordVal.txt",module)
    plt.clf()
    return time[:-1]/1000,module
def retFFT(step:float,x:np.ndarray,y:np.ndarray):
   vals=2*np.absolute(np.fft.rfft(y))/x.size
   freqs=np.fft.rfftfreq(x.size,d=step)
   return vals,freqs
def saveFFT(xlabel:str,ylabel:str,title:str,arrayVec:np.ndarray,time:np.ndarray,step:float):                
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
   
    left=np.argwhere(module>0)[0][0]       
    right=np.argwhere(module>0)[-1][-1]
    time=time[left:right]
    module=module[left:right]
    time=time-time[0]
    
   
    vals,freqs=retFFT(step,time,module)
    freqs=freqs*10**3
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    #plt.plot([24,24],[0,vals.max()])
    #plt.plot([82,82],[0,vals.max()])  
    
    
    plt.plot(freqs,abs(vals))
    plt.xscale("log")
    plt.savefig(title+" LogScaleAbsFFT.png")
    np.savetxt("freqs.txt",freqs)
    np.savetxt("vals.txt",vals)
    #plt.show()
    plt.clf()
    return freqs,abs(vals)
def saveAmount(step:float,maxtime:float,hdf5filePath:str):
    data=[]
    with tables.open_file(hdf5filePath) as h5file:
        for event in range(0,HDF5_EVENTS_AMOUNT):
            tracks = h5file.get_node("/event_{}".format(event), "tracks").read()
            data.append(tracks)
    
    outcome=np.zeros(int(maxtime/step))
    time=np.arange(0,int(maxtime/step)*step,int(step))
    integral=[]
    for tracks in data:
        for i in range(0,int(maxtime/step)):
            outcome[i]=outcome[i]+np.unique(tracks[(tracks['time']>i*step)*(tracks['time']<=(i+1)*step)]['id']).size
            integral.append(np.unique(tracks['id']).size)
    outcome=outcome/HDF5_EVENTS_AMOUNT      
    print(sum(integral)/len(integral))
    plt.xlabel("time,"+r"$\mu s$")
    plt.ylabel("Total amount")
    plt.title('Amount of particles in RREA')
    plt.plot(time/1000,outcome)
    plt.savefig('impulse range'+".png")
    np.savetxt('countTime.txt',time)
    np.savetxt("countVal.txt",outcome)
    #plt.show()
    plt.clf()
    return time,outcome
def main():
     time_step = 10
    
     signals = []
     
     for fileId in range(0,FILES_NUM):
     
       with tables.open_file("vhf_"+str(fileId)+".hdf5") as h5file:
           for event in range(HDF5_EVENTS_AMOUNT*fileId,HDF5_EVENTS_AMOUNT*(fileId+1)):
              tracks = process_event(h5file, np.array([0,0,-1000]), event_number=event, verbose=False)#500 m under RREA
              signal_from_event, time = join_track_signal(tracks, time_step)
              signals.append(signal_from_event)
     signal, time = join_event_signal(signals, time_step)
     titlePos="1 km under RREA"
     saveVectorBorders("time,"+r"$\mu s$","Electric field, V/m","Electric field,"+titlePos,signal,time,1300)
     saveAmount(time_step,3000,"vhf_0.hdf5")
     saveFFT("frequency, MHz","Electric field, V/m","E field spectrum,"+titlePos,signal,time[:-1],time_step)
       

    
     
     
    
if __name__ == '__main__':
    main()








