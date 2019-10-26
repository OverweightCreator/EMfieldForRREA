import numpy as np
import gc
from VHF import EMFieldCalculator
import matplotlib.pyplot as plt
plt.rcParams['font.size']=14
c = 0.3
e = 1.6e-19


class EventData: # requeres sorted data, when ids only increase
  def __init__(self, data):
    self.data = data
    self.ids, number = np.unique(self.data["id"], return_counts=True)
    self.number = np.cumsum(number) 

  def get_id_data(self, idTrack: int) -> np.ndarray:
    idTrack = np.where(self.ids == idTrack)[0]
    if idTrack == 0:
      left = 0
      right = self.number[0]
      
    else:
      left, right = self.number[idTrack - 1], self.number[idTrack]
    outcome=self.data[int(left):int(right)]
    temp,mask=np.unique(outcome['time'],return_index=True)
    return outcome[mask]
  


def getDataById(eventData: EventData):
  count = 0
  max_count = len(eventData.number)
  while count < max_count:
    if count == 0:
      outcome = eventData.data[0:int(eventData.number[count])]
      temp,mask=np.unique(outcome['time'],return_index=True)
    else:
      outcome = eventData.data[int(eventData.number[count - 1]):int(eventData.number[count])]
      temp,mask=np.unique(outcome['time'],return_index=True)
    yield eventData.ids[count], outcome[mask]
    count += 1
    


class SimDataLoader:
  def __init__(self, filename, dtype):
    self.data = np.fromfile(filename, dtype)
   

  def get_event(self, event: int) -> EventData:
    data = self.data[self.data["event"] == event]
    return EventData(np.sort(data, order="id"))
  def sliceData(self,left:int,right:int):
      self.data=self.data[left:right]
      gc.collect()
  def sliceDataByEvents(self,left:int,right:int):
      temp=self.data[self.data['event']<=right]
      self.data=temp[temp['event']>=left]
      del temp
      gc.collect()
  def checkValue(self,height,width):
      args=np.argwhere(self.data['x']**2+self.data['y']**2<width*width/4)
      self.data=self.data[args]
      args=np.argwhere(abs(self.data["z"])<height/2)
      self.data=self.data[args]
      

  
    

def fastV(track):  # m/ns
    diff_x = np.diff(track["x"])
    diff_y = np.diff(track["y"])
    diff_z = np.diff(track["z"])
    return ((np.array([diff_x, diff_y, diff_z])) / (np.diff(track["time"]))).T


def fastW(track):
    velocity = fastV(track).T
    time = track["time"][2:] - track["time"][:-2]
    return (np.diff(velocity) / time).T


CALCULATOR = EMFieldCalculator()


def pointingTrack(track,observedPoint,wFunc,efieldFunc):
  outtype = np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
  if (track.size < 3):
    return np.zeros(1, outtype),np.zeros(1, outtype)
  r0 = track[["x", "y", "z"]][1:-1]
  r0 = np.array([r0["x"],r0["y"],r0["z"]]).T
  v = fastV(track[0:-1])
  w = wFunc(track)
  outcomeE = np.zeros(track.size - 2, outtype)
  outcomeH = np.zeros(track.size - 2, outtype)
  outcomeE["time"] = np.sort(track[1:-1],order="time")['time']+CALCULATOR.norm(observedPoint-r0)/c
  outcomeH["time"] = outcomeE["time"]
  temp = efieldFunc(observedPoint, r0, v, w)
  outcomeE["x"] += temp[:,0]
  outcomeE["y"] += temp[:, 1]
  outcomeE["z"] += temp[: ,2]
  eVec=outcomeE[["x", "y", "z"]]
 
  eVec=eVec.view(np.float).reshape(eVec.shape+(-1,))[:,:3]# FROM NP ARRAY WITH DTYPE XYZ TO 2-DIMENSIONAL ARRAY

  outcomeHvec=CALCULATOR.cross(CALCULATOR.get_n(observedPoint,r0),eVec)
  outcomeHvec = np.core.records.fromarrays(outcomeHvec.transpose(), 
                                             names='x, y, z',
                                             formats = 'f8, f8, f8')
  outcomeH['x']=outcomeHvec['x']
  outcomeH['y']=outcomeHvec['y']
  outcomeH['z']=outcomeHvec['z']
 
  return outcomeE,outcomeH


def approximation(time, array):
    
    index = np.searchsorted(array["time"], time, side="left") - 1
    if (index >= array.size-1):
        ret=np.array([0.0, 0.0, 0.0, time])
        ret.dtype=np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
        return ret
    if(index<1):
        ret=np.array([0.0, 0.0, 0.0, time])
        ret.dtype=np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
        return ret
    else:
        array[index]['time']=time
     
        return array[index]


def pointingAvalache(data: EventData, observedPoint: np.ndarray, step: float,timelimit:float,start:float,wFunc,eFunc):

    outtype = np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
    outcomeE = np.zeros(int(timelimit / step),dtype=outtype)
    outcomeH = np.zeros(int(timelimit / step),dtype=outtype)
    for (idTrack, data) in getDataById(data):
        EH = pointingTrack(data, observedPoint,wFunc,eFunc)

        # Добавь код здесь, обрабатывай сразу весь трек, а потом уже складывай треки
        for j in range(0, int(timelimit / step)):
          time = j * step+start
          aproxE = approximation(time, EH[0])
          aproxH = approximation(time, EH[1])
          outcomeE[j]['x']=outcomeE[j]['x']+aproxE['x']
          outcomeE[j]['y']=outcomeE[j]['y']+aproxE['y']
          outcomeE[j]['z']=outcomeE[j]['z']+aproxE['z']
          outcomeE[j]['time']=time
          outcomeH[j]['x']=outcomeH[j]['x']+aproxH['x']
          outcomeH[j]['y']=outcomeH[j]['y']+aproxH['y']
          outcomeH[j]['z']=outcomeH[j]['z']+aproxH['z']
          outcomeH[j]['time']=time
    
    return outcomeE,outcomeH

def pointingSimulation(simDat:SimDataLoader,step:float,observedPoint:np.ndarray,wFunc,eFunc)->np.ndarray:
     maxtime=simDat.data['time'].max()+np.linalg.norm(observedPoint)/c
     start=np.linalg.norm(observedPoint)/c
   
     outtype = np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
     outcomeE = np.zeros(int(maxtime / step),dtype=outtype)
     outcomeH= np.zeros(int(maxtime / step),dtype=outtype)
     print("events num=",np.unique(simDat.data['event']).size)
     for i in np.unique(simDat.data['event']):
         tempEH=pointingAvalache(simDat.get_event(i), observedPoint, step,maxtime,start,wFunc,eFunc)
         outcomeE['x']+=tempEH[0]['x']
         outcomeE['y']+=tempEH[0]['y']
         outcomeE['z']+=tempEH[0]['z']
         outcomeE['time']+=tempEH[0]['time']
         outcomeH['x']+=tempEH[1]['x']
         outcomeH['y']+=tempEH[1]['y']
         outcomeH['z']+=tempEH[1]['z']
         outcomeH['time']+=tempEH[1]['time']
         print("event",i,"processing")
     size=np.unique(simDat.data['event']).size
     outcomeE['x']=outcomeE['x']/size
     outcomeE['y']=outcomeE['y']/size
     outcomeE['z']=outcomeE['z']/size
     outcomeE['time']=outcomeE['time']/size
     outcomeH['x']=outcomeH['x']/size
     outcomeH['y']=outcomeH['y']/size
     outcomeH['z']=outcomeH['z']/size
     outcomeH['time']=outcomeH['time']/size
     return outcomeE,outcomeH

def saveAmount(step:float,maxtime:float,data:np.array):
    outcome=np.zeros(int(maxtime/step))
    time=np.zeros(int(maxtime/step))
    for i in np.unique(data['event']):
        avalanche=data[data['event']==i]
        for j in range(0,int(maxtime/step)):
            temp=avalanche[avalanche['time']>j*step]
            temp=temp[temp['time']<=(j+1)*step]
            outcome[j]=outcome[j]+np.unique(temp['id']).size
            time[j]=j*step
    plt.xlabel("time, ns")
    plt.ylabel("counts")
    plt.title('impulse range')
    plt.plot(time/1000,outcome)
    plt.savefig('impulse range'+".png")
    plt.clf()
    return time,outcome
def saveVector(xlabel:str,ylabel:str,title:str,arrayVec:np.ndarray):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
    
    left=np.argwhere(module>0)[0][0]
    right=np.argwhere(module>0)[-1][-1]
    time=arrayVec['time'][left:right]
    module=module[left:right]
    plt.plot(time,module)
    plt.savefig(title+".png")
    plt.clf()
    return time,module
    
def retFFT(step:float,x:np.ndarray,y:np.ndarray):
   vals=2*np.absolute(np.fft.rfft(y))/x.size
   freqs=np.fft.rfftfreq(x.size,d=step)
   return vals,freqs
def saveFFT(xlabel:str,ylabel:str,title:str,arrayVec:np.ndarray,step:float,logMode:bool):                
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
   
    left=np.argwhere(module>0)[0][0]       
    right=np.argwhere(module>0)[-1][-1]
    time=arrayVec['time'][left:right]
    module=module[left:right]
    time=time-time[0]
    
   
    vals,freqs=retFFT(step,time,module)
    freqs=freqs*10**3
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title+"Abs")
    plt.plot([24,24],[0,vals.max()])
    plt.plot([82,82],[0,vals.max()])  
    
    for i in range(0,2):
      plt.plot(freqs,abs(vals))
      if logMode:
        plt.xscale("log")
        plt.savefig(title+" LogScaleAbsFFT"+str(i)+".png")
      else:
        plt.savefig(title+" AbsFFT"+str(i)+".png")
      plt.clf()


def pointingVec(E:np.ndarray,H:np.ndarray)->np.ndarray:
  eVec=E[["x", "y", "z"]]
  hVec=H[["x", "y", "z"]]
  eVec=eVec.view(np.float).reshape(eVec.shape+(-1,))[:,:3]
  hVec=hVec.view(np.float).reshape(hVec.shape+(-1,))[:,:3]
  temp=CALCULATOR.cross(eVec,hVec)*(c/(4*np.pi))
  temp = np.core.records.fromarrays(temp.transpose(), 
                                             names='x, y, z',
                                             formats = 'f8, f8, f8')
  outtype = np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
  outcome= np.zeros(temp.size,outtype)
  outcome['x']=temp['x']
  outcome['y']=temp['y']
  outcome['z']=temp['z']
  outcome['time']=E['time']
  return outcome



def saveVectorBorders(xlabel:str,ylabel:str,title:str,arrayVec:np.ndarray,delta:float):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    module=(arrayVec['x']**2+arrayVec['y']**2+arrayVec['z']**2)**0.5
    left=np.argwhere(module>0)[0][0]
    right=np.argwhere(arrayVec['time']<arrayVec['time'][left]+delta)[-1][-1]
        
    time=arrayVec['time'][left:right]
    module=module[left:right]
        
    plt.plot(time,module)
    plt.savefig(title+".png")
    plt.clf()
def main():
    
    dtype = np.dtype(
        [('id', "i4"),
         ('event', np.uint32),
         ('x', np.double),
         ('y', np.double),
         ('z', np.double),
         ('time', np.double)])
    path = "Electron.bin"
    step = 10 # ns
 
    simDat=SimDataLoader(path,dtype)
    simDat.sliceDataByEvents(0,100)
    simDat.checkValue(1000,9000)
    event=simDat.data[simDat.data['event']==4]
    z=event[(event["time"]>500)*(event["time"]<=510)]['z']
   
    print(max(z),min(z))
  
    time,outcome=saveAmount(step,3500,simDat.data)
    plt.xlabel("time,ns")
    plt.ylabel("I,A")
    plt.title("ELECTRIC CURRENT , A")
    plt.plot(time,e*outcome/600)
    plt.show()
    plt.plot(time[:-1],10**9*e*np.diff(outcome/600)/np.diff(time))
    plt.xlabel("time,ns")
    plt.ylabel("dI/dt,A")
    plt.title("di/dt , A")
    
    plt.show()
        
    
    
    
    

if __name__ == '__main__':
    main()
