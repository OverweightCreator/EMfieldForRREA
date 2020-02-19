import numpy as np
dtype = np.dtype(
    [('event', "i4"),
     ('id', "i4"),
     ('x', np.double),
     ('y', np.double),
     ('z', np.double),
     ('time', np.double)
     ])

data=np.fromfile("Electron.bin",dtype=dtype)
data=data[np.argsort(data,order=('event',"id","time"))]
data=data[['event','id','z','time']]
def getEvent(data,eventid):
  left=np.searchsorted(data['event'],eventid,side="left")
  right=np.searchsorted(data['event'],eventid,side="right")
  return data[left:right]
def getTrack(event,trackid):
  left=np.searchsorted(event['id'],trackid,side="left")
  right=np.searchsorted(event['id'],trackid,side="right")
  return event[left:right]


dt=100
dz=10
maxz=500
minz=-500
maxtime=15000
result=np.zeros(    (np.arange(0,maxtime,dt).size , np.arange(minz,maxz,dz).size)   )
print(np.arange(maxz,minz,dz))
for time in np.arange(0,maxtime,dt):
  for z in np.arange(minz,maxz,dz):
    print(time,z)
    for eventId in np.unique(data['event']):
      event=getEvent(data,eventId)
      for trackId in np.unique(event['id']):
        track=getTrack(event,trackId)
        idTime=np.searchsorted(track['time'],time)
        idZ=np.searchsorted(track['z'],z)
        if idTime==0 or idTime==track.size-1 or idZ==0 or idZ==track.size-1 or abs(track[idTime]['time']-t)>dt or abs(track[idZ]['z']-z)>dz:
          continue
        else:
          result[int(time/dt),int((z-minz)/dz)]=result[int(time/dt),int((z-minz)/dz)]+1
np.save('Distribution',result)
        

