
from mc_base import _MC_Base
from _mc import integrate_points

__all__ = [ "integrate_from_points" ]

class _Integrator_From_Points(_MC_Base):

    def __init__(self,f,points,args=(),nprocs=None,batch_size=None,
            weight=1.0):
        self.f = f
        self.points = points
        self.args = args
        self.weight = weight
        _MC_Base.__init__(self,nprocs,batch_size)

    @property
    def npoints(self):
        return len(self.points)

    def create_batches(self):
        _MC_Base.create_batches(self)
        self.batch_start = []
        self.batch_end = []
        accum = 0
        for batch_size in self.batch_sizes:
            self.batch_start.append(accum)
            self.batch_end.append(accum+batch_size)
            accum += batch_size

    def make_integrator(self):
        f = self.f
        def func(batch_number):
            start = self.batch_start[batch_number]
            end = self.batch_end[batch_number]
            #print "Start: ",start, "End: ",end
            return integrate_points(f,self.points[start:end],self.weight,self.args)
        return func


def integrate_from_points(f,points,args=(),nprocs=1,batch_size=None,weight=1.0):
    return _Integrator_From_Points(
            f,points,args,nprocs,batch_size,weight).run()

