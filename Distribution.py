import numpy as np
from multiprocessing import Pool
import gc
dtype = np.dtype(
    [('event', "i4"),
     ('id', "i4"),
     ('x', np.double),
     ('y', np.double),
     ('z', np.double),
     ('time', np.double)
     ])
EVENTS_NUM=10
data=np.fromfile("ElectronGurevich.bin",dtype=dtype)
#data=np.load('generic.npy')
data=data[np.argsort(data,order=('event',"id","time"))]
data=data[['event','id','z','time']]

def getEvent(data,eventid):
  left=np.searchsorted(data['event'],eventid,side="left")
  right=np.searchsorted(data['event'],eventid,side="right")
  return data[left:right]
def getEventLayer(data,maxeventid):
  right=np.searchsorted(data['event'],maxeventid,side="right")
  return data[0:right]
def getTrack(event,trackid):
  left=np.searchsorted(event['id'],trackid,side="left")
  right=np.searchsorted(event['id'],trackid,side="right")
  return event[left:right]
data=getEventLayer(data,EVENTS_NUM)

gc.collect()
dt=100
dz=25
maxz=500
minz=-750
maxtime=6000
result=np.zeros(    (np.arange(0,maxtime,dt).size , np.arange(minz,maxz,dz).size)   )

def processTracks(event,trackIds,time,dt,z,dz):
      for trackId in trackIds: 
          track=getTrack(event,trackId)
          
          if track.size==1 and abs(track[0]['time']-time)<dt and abs(track[0]['z']-z)<dz:
            return 1
            #result[int(time/dt),int((z-minz)/dz)]=result[int(time/dt),int((z-minz)/dz)]+1
            
          idTime=np.searchsorted(track['time'],time)
          idZ=np.searchsorted(track['z'],z)
          if track.size==0 or idTime==0 or idTime==track.size or idZ==0 or idZ==track.size or abs(track[idTime]['time']-time)>dt or abs(track[idZ]['z']-z)>dz:
            return 0
            #pass
          else:
            return 1
            #result[int(time/dt),int((z-minz)/dz)]=result[int(time/dt),int((z-minz)/dz)]+1
          return 0

def foo(event,time,dt,z,dz):
    print(time,z,event[0]['event'])
    return processTracks(event,np.unique(event['id']),time,dt,z,dz)
      
   
for time in np.arange(0,maxtime,dt):
  for z in np.arange(minz,maxz,dz):
  
    args=[]        
    
    for eventId in range(EVENTS_NUM):
       event=getEvent(data,eventId)
       args.append([event,time,dt,z,dz])

    with Pool(3) as p:
        result[int(time/dt),int((z-minz)/dz)]=result[int(time/dt),int((z-minz)/dz)]+sum(p.starmap(foo,args))
    """
    for eventId in np.unique(data['event']):
      event=getEvent(data,eventId)
      processTracks(event,np.unique(event['id']),time,dt,z,dz)
    """
result=result/EVENTS_NUM
np.save('Distribution',result)
        

