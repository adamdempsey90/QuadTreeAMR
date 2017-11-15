import numpy as np
import h5py

class Quantity():
    def __init__(self,name,fname=None):
        self.name = name
        self.load(fname)
    def load(self,fname):
        if fname is None:
            self.avg = None
            self.std = None
            self.rms = None
            return
        self.avg = float(fname['avg'][...])
        self.std = float(fname['std'][...])
        self.rms = float(fname['rms'][...])
    def copy(self):
        newpar = Quantity(self.name)
        newpar.avg = self.avg
        newpar.rms = self.rms
        newpar.std = self.std
        return newpar


    def save(self,fname):
        grp = fname.create_group(self.name)
        grp.create_dataset('avg',data=self.avg)
        grp.create_dataset('std',data=self.std)
        grp.create_dataset('rms',data=self.rms)
    def __eq__(self,par2):
        res = True
        for c in ['avg','std','rms']:
            p1 = getattr(self,c)
            p2 = getattr(par2,c)
            res &= p1 == p2
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)

class Average():
    cols=['a1', 'a2', 'a3',
          'e1', 'e2', 'e3',
          'n1', 'n2', 'n3',
          'p12', 'p14', 'p23',
          'phi1', 'phi2', 'phi3',
          'phi4', 'phi5', 'phi6',
          'phi7', 'phi8', 'phiL',
          'w1', 'w2', 'w3']
    def __init__(self,fname=None):
        self.load(fname)

    def load(self,fname):
        if fname is None:
            for c in self.cols:
                setattr(self,c,None)
            return
        for c in self.cols:
            dset = fname[c]
            setattr(self,c,Quantity(c,dset))
    def copy(self):
        newpar = Average()
        for c in self.cols:
            setattr(newpar,c,getattr(self,c).copy())
        return newpar
    def save(self,fname):
        for c in self.cols:
            getattr(self,c).save(fname)
    def __eq__(self,par2):
        res = True
        for c in self.cols:
            p1 = getattr(self,c)
            p2 = getattr(par2,c)
            res &= p1 == p2
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)
class ICs():
    cols = ['p0', 'p1', 'p2', 'p3']
    def __init__(self,fname=None):
        self.load(fname)
    def load(self,fname):
        if fname is None:
            for c in self.cols:
                setattr(self,c,None)
            return
        for c in self.cols:
            dset = fname[c]
            names = []
            dset.visit(names.append)
            dat = {}
            for n in names:
                dat[n] = float(dset[n][...])
            setattr(self,c,dat)
    def save(self,fname):
        for c in self.cols:
            grp = fname.create_group(c)
            for key,val in getattr(self,c).items():
                grp.create_dataset(key,data=val)
    def copy(self):
        newpar = ICs()
        for c in self.cols:
            setattr(newpar,c,getattr(self,c))
        return newpar
    def __eq__(self,par2):
        res = True
        for c in self.cols:
            d1 = getattr(self,c)
            d2 = getattr(par2,c)
            for key in d1.keys():
                res &= d1[key] == d2[key]
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)
class Chaos():
    cols = ['lyap', 'megno']
    def __init__(self,fname=None):
        self.load(fname)
    def load(self,fname):
        if fname is None:
            for c in self.cols:
                setattr(self,c,None)
            return
        for c in self.cols:
            setattr(self,c,float(fname[c][...]))

    def copy(self):
        newpar = Chaos()
        for c in self.cols:
            setattr(newpar,c,getattr(self,c))
        return newpar
    def save(self,fname):
        for c in self.cols:
            fname.create_dataset(c,data=getattr(self,c))
    def __eq__(self,par2):
        res = True
        for c in self.cols:
            p1 = getattr(self,c)
            p2 = getattr(par2,c)
            res &= p1 == p2
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)
class Bools():
    cols = ['inres', 'unst']
    def __init__(self,fname=None):
        self.load(fname)
    def load(self,fname):
        if fname is None:
            for c in self.cols:
                setattr(self,c,None)
            return
        for c in self.cols:
            setattr(self,c,bool(fname[c][...]))
    def copy(self):
        newpar = Bools()
        for c in self.cols:
            setattr(newpar,c,getattr(self,c))
        return newpar
    def save(self,fname):
        for c in self.cols:
            fname.create_dataset(c,data=getattr(self,c))
    def __eq__(self,par2):
        res = True
        for c in self.cols:
            p1 = getattr(self,c)
            p2 = getattr(par2,c)
            res &= p1 == p2
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)
class Parameters():
    cols = ['K', 'delta', 'endt', 'integrator', 'nperi', 'nt', 'ntau', 'smooth', 'tau_a', 'tau_e_1']
    dtypes = [float, float, float, str,int, int , int,bool, float, float]
    def __init__(self,fname=None):
        self.load(fname)
    def load(self,fname):
        if fname is None:
            for c in self.cols:
                setattr(self,c,None)
            return

        for c,d in zip(self.cols,self.dtypes):
            setattr(self,c,d(fname[c][...]))

    def save(self,fname):
        for c in self.cols:
            fname.create_dataset(c,data=getattr(self,c))
    def copy(self):
        newpar = Parameters()
        for c in self.cols:
            setattr(newpar,c,getattr(self,c))
        return newpar
    def __eq__(self,par2):
        res = True
        for c,d in zip(self.cols,self.dtypes):
            p1 = getattr(self,c)
            p2 = getattr(par2,c)
            if d == float:
                res &= np.isclose(p1,p2)
            else:
                res &= p1 == p2
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)

class Sim():
    cols = ['Averages','ICs','Chaos','Bools','Parameters']
    def __init__(self,fname=None,data=None,node=None):
        if data is not None:
            data.copy(newsim=self)
            self.name

        self.name = node.indx
        self.load(fname)
    def get_refinement_data(self):
        return self.Averages.phiL.rms
    def get_data(self):
        return self.Averages.phiL.rms
    def copy(self,newsim=None):
        return_flag = newsim is None
        if not return_flag:
            newsim = Sim(fname=None)
        newsim.Averages = self.Averages.copy()
        newsim.ICs = self.ICs.copy()
        newsim.Chaos = self.Chaos.copy()
        newsim.Bools = self.Bools.copy()
        newsim.Parameters = self.Parameters.copy()
        if return_flag:
            return newsim
    def load(self,fname):
        if fname is None:
            self.Averages = None
            self.ICs = None
            self.Chaos = None
            self.Bools = None
            self.Parameters = None
            return
        with h5py.File(fname,'r') as f:
            self.Averages = Average(fname=f['Averages'])
            self.ICs = ICs(fname=f['ICs'])
            self.Chaos = Chaos(fname=f['Chaos'])
            self.Bools = Bools(fname=f['Bools'])
            self.Parameters = Parameters(fname=f['Parameters'])
    def save(self,fname=None):
        with h5py.File(fname,'w') as f:
            grp = f.create_group('Averages')
            self.Averages.save(grp)
            grp = f.create_group('ICs')
            self.ICs.save(grp)
            grp = f.create_group('Chaos')
            self.Chaos.save(grp)
            grp = f.create_group('Bools')
            self.Bools.save(grp)
            grp = f.create_group('Parameters')
            self.Parameters.save(grp)
    def __eq__(self,par2):
        res = True
        for c in self.cols:
            p1 = getattr(self,c)
            p2 = getattr(par2,c)
            res &= p1 == p2
        return res
    def __neq__(self,par2):
        return ~self.__eq__(par2)

