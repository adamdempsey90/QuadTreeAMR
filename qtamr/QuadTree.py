import numpy as np
import h5py
from .DataClass import Sim as Data

class Tree():
    """
        The tree holds high level functions for interacting with
        the nodes.
    """
    def __init__(self,fname=None):
        """
            Initialize a tree from the hdf5 file fname
        """
        self.file = fname
        if fname is None:
            return

        try:
            self.open(fname)

        except OSError:
            print('File not found')
            self.file = None

    def build(self,data=None):
        """
            Build the tree from the hdf5 file
        """
        if self.file is None:
            self.root = Node('0',data=data)
            return
        self.root = Node('0')
        grp = self.file['0']
        self.root.build(grp)

    def open(self,fname):
        self.file = h5py.File(fname,'r')
    def close(self):
        self.file.close()
    def depth(self):
        lvls = []
        self.walk(func=lambda x: lvls.append(len(x.indx)-1))
        return max(lvls)
    def compression(self):
        leaves = self.list_leaves()
        maxlvl = max([len(n)-1 for n in leaves])
        npoints = 2**maxlvl * 2**maxlvl
        return 1-len(leaves)/float(npoints)
    def empty_leaves(self):
        """
            Build a list of leaves with no data.
        """
        nodes =[]
        self.walk(func=lambda x: nodes.append(x.indx) if x.leaf and x.data is None else None)
        return nodes
    def list_leaves(self):
        """
            Build a list of all leaves.
        """
        nodes = []
        self.walk(func=lambda x: nodes.append(x.indx))
        return nodes

    def walk(self,**kargs):
        """
            Walk the tree starting from the root.
        """
        return self.root.walk(**kargs)
    def refine(self,corners=False,**kargs):
        leaves = self.list_leaves()
        # Reorder based on level
        lvls = np.array([len(x) for x in leaves])
        inds = np.argsort(lvls)[::-1]
        leaves = list(np.array(leaves)[inds])
        for leaf in leaves:
            self.find(leaf).refine(corners=corners,**kargs)
        for leaf in leaves:
            node = self.find(leaf)
            if node.rflag:
                _,nuppers = node.find_neighbors(corners=corners)
                for upper in [i for x in nuppers for i in x]:
                    if upper is not None:
                        n = node.find(upper)
                        if n.leaf:
                            n.rflag = True

    def start_refine(self):
        for leaf in self.list_leaves():
            node = self.find(leaf)
            if node.rflag:
                node.split()
                node.rflag = False
        # Return a list of new, empty leaves
        return self.empty_leaves()

    def find(self,name):
        """
            Find node starting from the root
        """
        current_node = self.root
        for c in name[1:]:
            current_node = current_node.child[int(c)]
        return current_node
    def plot(self,fig=None,ax=None,vmin=0,vmax=180,cmap='viridis',**kargs):
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt


        if ax is None:
            fig,ax = plt.subplots()
        norm = colors.Normalize(vmin=vmin,vmax=vmax)
        self.root.plot(ax,norm=norm,cmap=cmap,**kargs)
    def save(self,fname):
        """
            Save the whole tree to an hdf5 file.
        """
        with h5py.File(fname,'w') as f:
            self.root.save(f)


class Node():
    """
        Nodes either point to their children or they have no
        children and instead hold some data.
    """
    def __init__(self,indx,file=None,parent=None,data=None):
        self.indx = indx
        self.name = '/'.join(self.indx)
        self.level = len(indx)-1
        self.global_index = (0,0,0)
        self.parent = parent
        if parent is not None:
            self.global_index = self.index_from_name(self.indx)
        self.leaf = True
        self.rflag = False
        self.child = [None]*4
        self.file = file
        self.data = data
    def save(self,file):
        """
            Write this node to the hdf5 group/file.
        """
        grp = file.create_group(self.indx[-1])
        if self.leaf:
            # We are a leaf, so we should dump our data
            dset = grp.create_group('Data')
            self.data.save(dset)
        else:
            # We are not a group, so call the children
            for c in self.child:
                c.save(grp)
        return

    def build(self,f):
        """
            Look in the hdf5 group f for child cells
        """


        for i in range(4):
            try:
                grp = f[str(i)]
                self.leaf = False
                self.child[i] = Node(self.indx+str(i),parent=self,file=grp)
                self.child[i].build(grp)
            except KeyError:
                self.leaf = True
                self.datastr = self.name + '/' + str(i) + '/Data'
                self.data=Data(fname=f['Data'],node=self)
                return

        return
    def split(self):
        """
            Split the node into four children, and pass the data to the
            first born.
        """
        self.leaf=False
        self.child[0] = Node(self.indx+'0',parent=self,data=self.data.copy())
        for i in range(1,4):
            self.child[i] = Node(self.indx+str(i),parent=self)
        return
    def name_from_index(self,k,i,j):
        """
            Calculate the name of the cell corresponding to
            global index (k,i,j)
        """
        name = []
        icurr = i
        jcurr = j
        for curr_level in range(k+1)[::-1]:
            name.append(str(2*(icurr%2)+jcurr%2))
            icurr = icurr // 2
            jcurr = jcurr // 2
        return ''.join(name[::-1])
    def index_from_name(self,name):
        """
            Calculate the index of the cell corresponding to
            the given name
        """
        icurr = 0
        jcurr = 0
        for k,c in enumerate(name):
            icurr = 2*icurr + int(c)//2
            jcurr = 2*jcurr + int(c)%2

        return (len(name)-1,icurr,jcurr)
    def get_xy(self):
        """
            Get the x,y coordinates on the unit square for this
            node.
        """
        k,i,j = self.global_index
        dx = 1./2**k
        return dx*i,dx*j
    def set_data(self,data):
        self.data = data.copy()
    def get_data(self):
        """
            Retrieve the data from the hdf5 file.
        """
        return self.data.get_data()
       # try:
       #     return self.file['Data'][...]
       # except:
       #     return self.data.get_data()
    def plot(self,ax,**kargs):
        """
            The recursive plotting function.
            Only draw if this is a leaf.
        """
        if self.leaf:
            self.draw(ax,**kargs)
            return
        for c in self.child:
            c.plot(ax,**kargs)
    def draw(self,ax,cmap='Spectral',plot_refined=True,edges=False,**kargs):
        """
            Draw a rectangle for this cell and color by the given
            data field.
            If needed, indicate the cell is tagged for refinement.
        """
        import matplotlib.cm
        import matplotlib.patches as patches
        import matplotlib.collections as collections

        norm = kargs.pop('norm',None)
        try:
            dat = self.data.get_data()
            cmap = matplotlib.cm.get_cmap(cmap)
            if norm is not None:
                c = cmap(norm(dat))
            else:
                c = cmap(dat)
        except:
            dat = np.nan
            c = 'w'


        k,i,j=self.global_index
        dx = 1./2**k
        dy = dx
        x = dx *i
        y = dy*j
        rect = patches.Rectangle((x,y),dx,dy)
        if edges:
            ax.add_collection(collections.PatchCollection([rect],facecolor=c,edgecolor='k'))#,lw=1))
        else:
            ax.add_collection(collections.PatchCollection([rect],facecolor=c))#,lw=1))
        if plot_refined and self.rflag:
            ax.plot(x+dx/2,y+dy/2,'ro',ms=2)

    def find(self,name):
        """
           Find the next step towards the desired
           node with name name.
        """
        len_myself = len(self.indx)
        len_name = len(name)
        if self.indx == name:
            # Found it!
            #print('Node ', self.indx, ' found ',name)
            return self
        if len_myself < len_name:
            if self.indx == name[:len_myself]:
                # It's below us in the tree
                child = name[:len_myself+1][-1]
                #print('Node ', self.indx, ' is going to child ',child)
                return self.down(int(child)).find(name)
        # It's not below us, so move up
        #print('Node ', self.indx, ' is going up to find ',name)

        return self.up().find(name)
    def refine(self,refine_all=False,corners=False,**kargs):
        """
            Check neighbors to see if this node should
            be refined.
        """

        # First check if already tagged

        neighbors, upper_neighbors = self.find_neighbors()

        # Even if already tagged, still need to check new neighbors
        final_list = [[None,None,None],[None,None,None],[None,None,None]]
        for i in range(3):
            for j in range(3):
                if upper_neighbors[i][j] is not None:
                    node = self.find(upper_neighbors[i][j])
                    if not node.leaf:
                        node = node.find(neighbors[i][j])
                    final_list[i][j] = node


        res = self.check_refinement(final_list,**kargs)

        for i in range(3):
            for j in range(3):
                if final_list[i][j] is not None:
                    final_list[i][j].rflag |= res[i][j]

        # Checked all neighbors. Broadcast result if True
        if refine_all:
            if self.rflag:
                for i in range(3):
                    for j in range(3):
                        final_list[i][j].rflag |= True


        return


    def find_neighbors(self):
        """
            Find the neighbors and their parents.
        """
        k,i,j = self.global_index
        max_indx = 2**k
        max_indx_up = 2**(k-1)
        neighbors = []
        upper_neighbors = []


        neighbors = [ [None,None,None],[None,self.indx,None],[None,None,None]]
        upper_neighbors = [ [None,None,None],[None,None if self.parent is None else self.parent.indx,None],[None,None,None]]
        stencil = [(-1,0),(1,0),(0,-1),(0,1)]
        stencil += [(-1,1),(1,-1),(1,1),(-1,-1)]

        for di,dj in stencil:
            ii = i + di
            jj = j + dj
            if ii>=0 and jj>=0 and ii<max_indx and jj<max_indx:
                neighbors[1+di][1+dj] = self.name_from_index(k,ii,jj)
            iu = ii//2
            ju = jj//2
            ku = k-1
            if iu>=0 and ju>=0 and iu<max_indx_up and ju<max_indx_up:
                upper_neighbors[1+di][1+dj] = self.name_from_index(ku,iu,ju)
        return neighbors, upper_neighbors
    def check_refinement(self,nodes,**kargs):
        """Given the neighbors, check for refinement."""
        return self.refinement_lohner(nodes,**kargs)
    def refinement_lohner(self,nodes,tol=.8,corners=True,**kargs):
        ans = [[False,False,False],[False,False,False],[False,False,False]]
        u = np.zeros((3,3))

        u1 = self.data.get_refinement_data()
        for i in range(3):
            for j in range(3):
                try:
                    u[i,j] = nodes[i][j].data.get_refinement_data()
                except:
                    u[i,j] = u1

        numerator = 0
        denominator = 0
        numerator += (abs(u[2,1] - 2*u[1,1] + u[0,1]))**2
        numerator += (abs(u[1,2] - 2*u[1,1] + u[1,0]))**2
        if corners:
            numerator += (.25*abs( u[2,2] + u[0,0] - u[0,2] - u[2,0]))**2

        denominator += (abs(u[2,1]-u[1,1]) + abs(u[0,1]-u[1,1]))**2
        denominator += (abs(u[1,2]-u[1,1]) + abs(u[1,0]-u[1,1]))**2

        resx = np.sqrt(numerator/denominator)
        if resx >= tol:
            for i in range(3):
                for j in range(3):
                    ans[i][j] = True
        return ans
    def up(self):
        """
            Move up the tree
        """
        return self.parent
    def down(self,i=0):
        """
            Move down the tree to child i
        """
        return self.child[i]
    def walk(self,printname=False,func=None):
        """
            Recursively walk the tree, applying the function func if this
            is a leaf.
        """
        if self.leaf:
            if func is not None:
                func(self)
            if printname:
                print(self)
            return self
        for c in self.child:
            c.walk(printname=printname,func=func)
    def __repr__(self):
        return self.indx
    def __str__(self):
        return self.indx

