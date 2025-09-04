import numpy as np
import pandas as pd
from skimage import morphology as morph
#import matplotlib.pyplot as plt

def tiler(imc,prots,Tsize,step,K=1,tissueT=0.5,TT=1,dT=10**5):
    
    ' ' 'IMC image (nr,nc,prots), protein name in each channel, Tile size, step size, K=Dilation radius, tissue T: minimum ratio of tissue in tile to consider tile, TT threshold IMC signal to consider tissue ' ' '
    
    ##now create channel pair colocs
    imt=np.sum(imc,axis=2)
    BW=np.asarray(1*(imt>TT))
    imcK=np.zeros(np.shape(imc),dtype=np.float32)
    
    nr,nc,prot_num=np.shape(imc)
    
    for i in range(prot_num):
        imcK[:,:,i]=morph.dilation(imc[:,:,i],footprint=morph.disk(K))
    
    combnum=int(0.5*prot_num*(prot_num-1))
    imdK=np.zeros((nr,nc,combnum),dtype=np.float32)
    
    ccomb=0
    prot_pairs=[]
    for chan1 in range(0,prot_num):
        for chan2 in range(chan1+1,prot_num):
            imdK[:,:,ccomb]=imcK[:,:,chan1]*imcK[:,:,chan2]
            prot_pairs.append(prots[chan1]+'-'+prots[chan2])
            ccomb=ccomb+1
    
    ###############################################
    ###################now tile the IMC
    f1=np.zeros((dT,prot_num),dtype=np.float32)
    f2K=np.zeros((dT,combnum),dtype=np.float32)
    locs=np.zeros((dT,2),dtype=np.uint32)
    
    
    tcnt=-1
    for ir in range(0,nr-Tsize-1,step):
        for jc in range(0,nc-Tsize-1,step):
            
            
            cBW=BW[ir:ir+Tsize,jc:jc+Tsize]
                
                
            if np.sum(cBW)/Tsize**2>tissueT:
                
                tcnt=tcnt+1
                cim=imc[ir:ir+Tsize,jc:jc+Tsize,:]
                
                cimdK=imdK[ir:ir+Tsize,jc:jc+Tsize,:]
                
                
                f1[tcnt,:]=np.mean(cim,axis=(0,1))
                f2K[tcnt,:]=np.mean(cimdK,axis=(0,1))
                
                locs[tcnt,0]=ir
                locs[tcnt,1]=jc
        
    
    f1=f1[:tcnt+1,:]
    f2K=f2K[:tcnt+1,:]
    locs=locs[:tcnt+1,:]
    
    featmat=np.concatenate((locs,f1,f2K),axis=1)
    col_names=['row','col']+prots+prot_pairs
    
    df=pd.DataFrame(featmat,columns=col_names)
    
    return df



#####TSI
def sep_train(df,DS,DG={},DGchan={},MaxIter=200,step_size=0.01,link='sqr',EPS=10**-10,cnum_norm='sqrt'):
    
    ' ' ' separating major tissue types, such as tumor, stroma, and immune. df: data dataframe, DS: a dictionary for the structure.      ' ' '
    
    classes=list(DS.keys())
    
    #########################
    ##if there is no gain for a class, make it 1
    DGkeys=list(DG.keys())
    for ckey in classes:
        if not (ckey in DGkeys):
            DG[ckey]=1
    #####create an initial guess for weights
    WD={}
    
    for cclass in classes:
        cw=1/np.sqrt(len(DS[cclass]))*np.ones((1,len(DS[cclass])))
        if cclass in list(DG.keys()):
            WD[cclass]=DG[cclass]*cw
        else:
            WD[cclass]=cw
    ###convert df to numpy array for simple processing down the line
    X=np.asarray(df)
    prots=list(df.columns)
    
    ###################indices of prots in each class
    DI={}
    for ckey in list(DS.keys()):
        DI[ckey]=[]
    
    for i in range(len(prots)):
        cprot=prots[i]
        for ckey in list(DS.keys()):
            if cprot in DS[ckey]:
                clist=DI[ckey]
                clist.append(i)
                DI[ckey]=clist
    
    #############train separator
    cnum=len(classes)
    scoresD={}
    tscoresD={}
    LabD={}
    tnum=len(df)
    class_scores=np.zeros((tnum,cnum),dtype=np.float32)
    tclass_scores=np.zeros((tnum,cnum),dtype=np.float32)
    for itercnt in range(MaxIter):
        #print(itercnt)
    
        if itercnt<100:
            my_step_size=step_size
        else:
            my_step_size=step_size/(itercnt-99)
    
        #####compute class scores
        for cclass_cnt in range(cnum):
            cclass=classes[cclass_cnt]
            cscore=WD[cclass]*X[:,DI[cclass]]
            if link=='lin':
                tscore=cscore
            if link=='sqr':
                tscore=cscore**2
            
                    
            class_scores[:,cclass_cnt]=np.sum(cscore,axis=1)            
            tclass_scores[:,cclass_cnt]=np.sum(tscore,axis=1)
            scoresD[cclass]=cscore
            tscoresD[cclass]=tscore
        tot_scores=np.sum(class_scores,axis=1)
        tot_scores=tot_scores[:,np.newaxis]
            
        tot_tscores=np.sum(tclass_scores,axis=1)
        tot_tscores=tot_tscores[:,np.newaxis]
        ############
        ##find current labs
        clabs=np.argmax(class_scores,axis=1)
        clabs=clabs[:,np.newaxis]
        ###now take grad
        gradD={}
        gradvec_list=[]
        for cclass_cnt in range(cnum):
            cclass=classes[cclass_cnt]
            
            #print(cclass)
        
            cclass_scores=class_scores[:,cclass_cnt]
            cclass_scores=cclass_scores[:,np.newaxis]
            
            ctclass_scores=tclass_scores[:,cclass_cnt]
            ctclass_scores=ctclass_scores[:,np.newaxis]
            
            if link=='lin':
                cgrad=(  (X[:,DI[cclass]]*tot_scores) -  cclass_scores*X[:,DI[cclass]]      )/(tot_scores**2+EPS)
            if link=='sqr':
                cgrad=( 2*cclass_scores*X[:,DI[cclass]]*tot_tscores - 2*(cclass_scores**3)*X[:,DI[cclass]]    )/(tot_tscores**2+EPS)
                
                
            cgrad=cgrad*(clabs==cclass_cnt)
            cgrad=np.sum(cgrad,axis=0)/(np.sum(1*(clabs==cclass_cnt))+EPS)
                
            gradD[cclass]=cgrad
                                
            gradvec_list=gradvec_list+list(cgrad)
        ################
        ##update weights
        for cclass in classes:
            if cnum_norm=='lin':
                WD[cclass]=WD[cclass]+(my_step_size/len(cclass))*gradD[cclass]/(np.linalg.norm(gradvec_list)+EPS)
            else:
                WD[cclass]=WD[cclass]+(my_step_size/np.sqrt(len(cclass)))*gradD[cclass]/(np.linalg.norm(gradvec_list)+EPS)
            #cWD[cclass]=cWD[cclass]+0.01*gradD[cclass]
        #print(gradvec_list)
        
    ####################
    ##now compute prob vecs
    class_scores=np.zeros((tnum,cnum),dtype=np.float32)
    for cclass_cnt in range(cnum):
        cclass=classes[cclass_cnt]
        cscore=WD[cclass]*X[:,DI[cclass]]
        class_scores[:,cclass_cnt]=np.sum(cscore,axis=1)
    for cclass_cnt in range(cnum):
        cclass=classes[cclass_cnt]
        LabD[cclass]=1*(np.argmax(class_scores,axis=1)==cclass_cnt)
        #print('num tiles in '+cclass+' is '+str(np.sum(LabD[cclass])))
    
    
    Wall={}
    Wall['WD']=WD
    Wall['DI']=DI
    Wall['classes']=classes
    Wall['link']=link
    
    return Wall

def sep_predict(df,Wall):
    
    X=np.asarray(df)
    
    DI=Wall['DI']
    WD=Wall['WD']
    classes=Wall['classes']
    link=Wall['link']
    
    tnum=len(X)
    cnum=len(classes)
    EPS=10**-10
    
    pmat=np.zeros((tnum,cnum),dtype=np.float32)
    for i in range(cnum):
        pmat[:,i]=np.sum(WD[classes[i]]*X[:,DI[classes[i]]],axis=1)
        if link=='sqr':
            pmat=pmat**2
    
    psum=np.squeeze(np.sum(pmat,axis=1))
    psum=psum[:,np.newaxis]
    pmat=pmat/(psum+EPS)
    pmat[pmat<0]=0
    pmat[pmat>1]=1
    pmat[np.isnan(pmat)]=0
    
    return pmat

###################
##preprocess NMF
def feat_sel(df,TE=0.05,TR=0.05,prots_to_remove=[]):
    
    EPS=0.00000001 ###some small epsilon for mathemtical correctness, not a big deal
    X=np.asarray(df)
    prots=list(df.columns)
    
    remove_chan_ids=[]
    for i in range(len(prots)):
        cprot=prots[i]
        if cprot in prots_to_remove:
            remove_chan_ids.append(i)
            
    
    rats=np.mean(1*(X>TE),axis=0)
    if len(remove_chan_ids)>0:
        rats[remove_chan_ids]=-1
    
    comchans=np.squeeze(np.asarray(np.where(rats>TR-EPS)))
    
    Xr=X[:,comchans]
    protsr=[]
    for i in range(len(comchans)):
        protsr.append(prots[i])
    
    df=pd.DataFrame(Xr,columns=protsr)
    return df, comchans

###Hmap
def hmapgen(X,locs,Tsize,shape,tissue_mask):
            
    nr=shape[0]
    nc=shape[1]
    tnum,fdim=np.shape(X)
    
    hmap=np.zeros((nr,nc,fdim),dtype=np.float32)
    nmap=np.zeros((nr,nc),dtype=np.float32)
    
    for t in range(tnum):
        cr=int(locs[t,0])
        cc=int(locs[t,1])
        cX=X[t,:]
        hmap[cr:cr+Tsize,cc:cc+Tsize,:]=hmap[cr:cr+Tsize,cc:cc+Tsize,:]+cX
        nmap[cr:cr+Tsize,cc:cc+Tsize]=nmap[cr:cr+Tsize,cc:cc+Tsize]+1
        
    hmap=hmap/(nmap[:,:,np.newaxis]+0.0000000001)
    for i in range(fdim):
        chmap=hmap[:,:,i]
        chmap[nmap<0.5]=0
        chmap=chmap*tissue_mask
        hmap[:,:,i]=chmap
    
    return hmap

####cluster map
def cluster(hmap,labmap,tmask,nmf_dims,background=0):
    
    nr,nc,dummy=np.shape(hmap)
    
    tnum=len(nmf_dims)
    
    cum_dims=[0]
    for i in range(tnum):
        cum_dims.append(cum_dims[i]+nmf_dims[i])
    
    clmap=np.zeros((nr,nc),dtype=np.uint32)
    for i in range(tnum):
        ctnmf=hmap[:,:,cum_dims[i]:cum_dims[i+1]]
        to_include=1*(labmap==i)*tmask
        clmap=clmap+tmask*to_include*(cum_dims[i]+1+np.argmax(ctnmf,axis=2))
    
    clmap=clmap+background
    
    return clmap

#########################################
###plot shades of clusters
def clmap_viser(cl_map,colormap,background=0):
    
    cl_map=cl_map-background
    
    nr,nc=np.shape(cl_map)
    clust_num=len(colormap)
    
    img=np.zeros((nr,nc,3),dtype=np.float32)
    
    for i in range(clust_num):
        img[cl_map==i+1,:]=colormap[i,:]
    
    return img
    


def comp_area(clmap,clust_num=-1,background=0):
    
    clmap=clmap-background
    
    if clust_num<0:
        clust_num=np.max(clmap)
    
    area=np.zeros((clust_num,))
    
    for i in range(1,clust_num+1):
        area[i-1]=np.sum(1*(clmap==i))
    
    return area
    

###########################
###interpret nmf
def interpret_nmf(nmf_model,top_dim=10,normalize=False,prots=[],df=[]):
    
    if len(prots)==0:
        prots=list(df.columns)
    
    comps=nmf_model.components_
    
    if normalize:
        Scomps=np.sum(comps,axis=0)
        comps=comps/Scomps[np.newaxis,:]
    
    weights_df=pd.DataFrame(comps,columns=prots)
    
    top_prots_all=[]
    cT={}
    cols=[]
    for i in range(len(comps)):
        cw=comps[i,:]
        top_prots=[]
        if np.sum(cw)<0.000001:
            for j in range(top_dim):
                top_prots.append('NONE')
        else:
            topinds=np.flip(np.argsort(cw))
            for j in range(top_dim):
                top_prots.append(prots[topinds[j]])
        
        top_prots_all=top_prots_all+top_prots
        cT['dim'+str(i)]=top_prots
        cols.append('dim'+str(i))
        
    top_prots_df=pd.DataFrame.from_dict(cT)
    
    
    return weights_df, top_prots_df
    
##############################################3
####extract average expression on each mf cluster
def measure_expr(imc,cl_map,clust_num=-1,prots=[],background=0,normalize=True):
    
    if clust_num<0:
        clust_num=np.max(cl_map)
    
    nr,nc,marker_num=np.shape(imc)
    
    exprs=np.zeros((clust_num,marker_num))
    
    for i in range(clust_num):
        mask=1*(cl_map==1+i+background)
        area=np.sum(mask)
        if area>0:
            for j in range(marker_num):
                if normalize:
                    exprs[i,j]=np.sum(imc[:,:,j]*mask)/area
                else:
                    exprs[i,j]=np.sum(imc[:,:,j]*mask)
    
    if len(prots)>0:
        exprs=pd.DataFrame(exprs,columns=prots)
        
    
    return exprs
            
        
        
  
    
    
