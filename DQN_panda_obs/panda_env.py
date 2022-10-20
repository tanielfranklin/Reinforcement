#Adding a cuboid obstacle
from spatialgeometry import Cuboid
import math
import numpy as np
import random
import spatialmath as sm
import spatialgeometry as sg
import swift
import roboticstoolbox as rp
from roboticstoolbox.tools.urdf.urdf import Collision
class Panda_RL(object):
    
    def __init__(self,delta=0.1):
        self.scene = swift.Swift()
        self.scene.launch(realtime=True)
        self.panda = rp.models.Panda()
        self.panda_end = rp.models.Panda()
        self.m=100 #magnification factor
        self.renderize=True
        
        self.obstacle = Cuboid([0.2, 0.2, 0.8], pose=sm.SE3(0.3, 0, 0)) 
        #self.floor = Cuboid([0.2, 0.2, 0.8], pose=sm.SE3(-0.2, 0, 0)) 
        self.obs_floor = Cuboid([2., 2., 0.01], pose=sm.SE3(0, 0, 0), color=[100,100,100,0])#"black") #color=[100,100,100,0]
        self.scene.add(self.obs_floor)
        self.scene.add(self.obstacle)
        self.scene.add(self.panda, robot_alpha=0.6)
        self.delta=delta
        #End joints positions
        j=[0.8,-1.5,1] 
        self.q_end=[0., j[0], 0.,j[1], 0., j[2], 0.]
        self.Tep = self.panda.fkine(self.q_end)
        self.f0=self.fitness()
        # set end position
        self.panda.q = self.q_end
        # get goal position
        self.p_goal=self.get_position()
        #set initial position
        self.panda.q = self.panda.qz
        self.set_end_target(self.q_end)
        
        self.p_rob=self.get_position()
        self.d_1=self.distance()
        
        self.mu_1=100

        
    def set_end_target(self,q):        
        #Add to workspace
        Tep = self.panda.fkine(q)
        axes=sg.Axes(length=0.1,pose=Tep)
        self.scene.add(axes)
        
        
    def reset(self):
        #j1 range -1.7 a 1.7
        #j2 range 0.0 a -3.
        #j3 range 0.0 a 3.7
        
        j1=[-1.7,1.7]
        j2=[0.0, -3.]
        j3=[0.0, 3.7]
        
        collision=True
        
        #initial states without collisions
        while collision:
            self.panda.q[1]=round(random.uniform(j1[0],j1[1]),2)
            self.panda.q[3]=round(random.uniform(j2[0],j2[1]),2)
            self.panda.q[5]=round(random.uniform(j3[0],j3[1]),2)
            if self.renderize:
                self.scene.step()
            collision,_=self.detect_collision()
            
            
                
        
        return np.array([self.panda.q[1],self.panda.q[3],self.panda.q[5]])
        
    def get_position(self):
        #Get matrix 
        pos=self.panda.fkine(self.panda.q)
        #Get position vector
        pos=np.array(pos)[0:-1,-1]
        return pos
    
    def get_q(self):
        #get free active joints
        q=[1,3,5]
        return [self.panda.q[i] for i in q]
        
        
        
        
    def step(self,a):
        #change joint angles by delta, do nothing or -delta
        #print(a)
        info=""
        d=self.distance()
        s=self.panda.q
        self.panda.q[1]+=a[0]*self.delta
        self.panda.q[3]+=a[1]*self.delta
        self.panda.q[5]+=a[2]*self.delta
        if self.renderize:
            self.scene.step()
        next_state=np.array([self.panda.q[1],self.panda.q[3],self.panda.q[5]])
        info=["",""]
        done =False
        
        
        if self.detect_collision()[0]:
            # next_state=s
            # next_state=np.array([self.panda.q[1],self.panda.q[3],self.panda.q[5]])
            r=-100
            done=True
            info=["Done","Collided"]    
        if self.distance()<0.005:
            done=True
            info=["Done","Completed"]
            r=100
        else:
            r=-2*self.distance() + 20*(d-self.distance())   
        return next_state,r , done,info
    
    def reach_joint_limit(self):
        #j3 range -0.08 a 3.75  #j2 range -0.07 a -3. #j1 range -1.8 a 1.76
        j_lim=[(-1.8,1.76), (-3,0),(0,3.75)]
        q=[1,3,5]
        joint=[]
        for k,j in enumerate(j_lim):        
            if (self.panda.q[q[k]])>j[1] or (self.panda.q[q[k]])<j[0]:
                joint.append(k)
                return True,joint
            else:
                return False,joint
            
    def detect_collision(self):
        collision = [self.panda.links[i].collided(self.obstacle) for i in range(9)]
        # Discarding collisions among first and second links with the floor
        collision_floor = [self.panda.links[i].collided(self.obs_floor) for i in np.arange(2,9)]
        collision.append(self.reach_joint_limit()[0])
        if sum(collision)+sum(collision_floor)!=0:
            return True, [collision, collision_floor]
        else:
            return False, [collision, collision_floor]         
        
        
    
    def reward(self,d):
        r_col=0
        col=False
        r=0
        if self.detect_collision()[0]:
            r_col=-200
            col=True
        else:
            r=-self.distance()*self.mu_1-r_col            
        self.d_1=self.distance()
        return r,col
    
    def reward2(self):        
        r=(math.atan(self.f0-self.fitness())*math.pi/(2*self.fg))*self.m
        return r       
        
    def fitness(self):
        value=np.sum(np.array(self.panda.q-self.q_end)**2)
        return value
    def distance(self):
        self.p_rob=self.get_position()
        value=np.sum(np.array(self.p_rob-self.p_goal)**2)
        return value
        
    # def render(self, mode='human', close=False):
    # # Render the environment to the screen