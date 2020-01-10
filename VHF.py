#!/usr/bin/env python
# coding: utf-8




import numpy as np
import tables
import matplotlib.pyplot as plt

import logging


HDF5_EVENTS_AMOUNT = 25
FILES_NUM=4
TOTAL_NUM=FILES_NUM

dtype = np.dtype(
    [('event', "i4"),
     ('id', "i4"),
     ('x', np.double),
     ('y', np.double),
     ('z', np.double),
     ('time', np.double),
     ('theta', np.double),
     ('energy', np.double)])
     
from multiprocessing import Pool





            
 



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





def compute_velocity(track):  # m/ns 
    diff_x = np.diff(track["x"])
    diff_y = np.diff(track["y"])
    diff_z = np.diff(track["z"])
    return ((np.array([diff_x, diff_y, diff_z])) / (np.diff(track["time"]))).T


def compute_acceleration(velocity, time): # m/ns^2
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
    value["x"]=value["x"]/len(events)
    value["y"]=value["y"]/len(events)
    value["z"]=value["z"]/len(events)
    return value, grid

def saveVectorBorders(arrayVec:np.ndarray,time:np.ndarray,delta:float,folder:str):
 
    
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
    left=np.argwhere(module>0)[0][0]
    right=np.argwhere(time<time[left]+delta)[-1][-1]
    np.savetxt(folder+'/inpTime.txt',time)
    np.savetxt(folder+"/inpVal.txt",module)    
    time=time[left-1:right]
    module=module[left:right]
        
  
    np.savetxt(folder+"/bordTime.txt",time[:-1])
    np.savetxt(folder+"/bordVal.txt",module)

    return time[:-1]/1000,module
def retFFT(step:float,x:np.ndarray,y:np.ndarray):
   vals=2*np.absolute(np.fft.rfft(y))/x.size
   freqs=np.fft.rfftfreq(x.size,d=step)
   return vals,freqs
def saveFFT(arrayVec:np.ndarray,time:np.ndarray,step:float,folder:str):                
    print("FFT")
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
   
    left=np.argwhere(module>0)[0][0]       
    right=np.argwhere(module>0)[-1][-1]
    time=time[left:right]
    module=module[left:right]
    time=time-time[0]
    
   
    vals,freqs=retFFT(step,time,module)
    freqs=freqs*10**3
  
    np.savetxt(folder+"/freqs.txt",freqs)
    np.savetxt(folder+"/vals.txt",vals)
   
    return freqs,abs(vals)





"electrical current and number of particles"
def getEvents(hdf5Path):
    data=[]
    with tables.open_file(hdf5Path) as h5file:
        for event in range(0,HDF5_EVENTS_AMOUNT):
            tracks = h5file.get_node("/event_{}".format(event), "tracks").read()
            data.append(tracks)
    return(data)  
def currentEvents(time,data,dt):
    out=[]
    for event in data:
      for trackId in np.unique(event['id']):
        track=event[np.searchsorted(event['id'],trackId,side="left"):np.searchsorted(event['id'],trackId,side="right")]
        currentTrackId=np.searchsorted(track["time"],time)
        if currentTrackId<=track.size-1:
          currentTrack=track[currentTrackId]
          if abs(currentTrack['time']-time)<dt:
              out.append(currentTrack) 
    #return len(out)/HDF5_EVENTS_AMOUNT
    return out
    
def getCurrentLocatedEvents(events,time,z,dz,dt):
    eventsOnTime=currentEvents(time,events,dt)
    out=[]
    for track in eventsOnTime:
        if abs(track['z']-z)<dz:
          out.append(track)
    return out
def getCurrentN(events,time,z,dz,dt):
    return len(getCurrentLocatedEvents(events,time,z,dz,dt))/HDF5_EVENTS_AMOUNT
def getI(velocity,charge,dz,N):
    return velocity*charge*N/dz
"electrical current and number of particles - end of code"    
    
    
def main(folder:str,vec:np.ndarray):
     time_step = 10
     signals = []
     
     print(folder,vec)
     
     for fileId in range(0,TOTAL_NUM):
           with tables.open_file("vhfGurevich_"+str(fileId)+".hdf5") as h5file:
               for event in range(HDF5_EVENTS_AMOUNT*fileId,HDF5_EVENTS_AMOUNT*(fileId+1)):
                  event=event%(HDF5_EVENTS_AMOUNT*FILES_NUM)
                  tracks = process_event(h5file, vec, event_number=event, verbose=False)
                  signal_from_event, time = join_track_signal(tracks, time_step)
                  signals.append(signal_from_event)
                  print(folder,fileId,event)
                  logging.info(str(folder)+" "+str(fileId)+" "+str(event))
     signal, time = join_event_signal(signals, time_step)
     saveVectorBorders(signal,time,1300,folder)
     saveFFT(signal,time[:-1],time_step,folder)

    
     
     
    
if __name__ == '__main__':
    
    
    logging.basicConfig(filename="sample.log", level=logging.INFO)
    """
    data=np.fromfile("Electron.bin",dtype=dtype)
    logging.info("Electron.bin")
    for fileId in range(0,TOTAL_NUM):
      
         if fileId%FILES_NUM==0 and fileId>0:
             data=np.fromfile("Electron"+str(int(fileId/FILES_NUM))+".bin",dtype=dtype)
             print("Electron"+str(int(fileId/FILES_NUM))+".bin")
             logging.info("Electron"+str(int(fileId/FILES_NUM))+".bin")
         logging.info(str(fileId)+" "+str((HDF5_EVENTS_AMOUNT*fileId)%(HDF5_EVENTS_AMOUNT*FILES_NUM)))
         create_file(data,fileId,(HDF5_EVENTS_AMOUNT*fileId)%(HDF5_EVENTS_AMOUNT*FILES_NUM))              
    """
    folders=["1kmG","2kmG","5kmG","707mx707mG","866mx500mG","500mx866mG"]
    ranges=[np.array([0,0,-1000]),np.array([0,0,-2000]),np.array([0,0,-5000]),np.array([707,0,-707]),np.array([500,0,-866]),np.array([866,0,-500])]
    args=tuple(zip(folders, ranges))
    
    with Pool(1) as p:
        print(p.starmap(main,args))
        
    
    """
    data=getEvents("vhfGurevich_0.hdf5")
    print(getCurrentN(data,3000,-300,5,5))
    print(getCurrentN(data,3000,-300,10,5))
    print(getCurrentN(data,3000,-300,2,5))
    print(getCurrentN(data,3000,-300,15,5))
    """




