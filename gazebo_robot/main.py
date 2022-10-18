# Program for Forward Kinematics of 2R Robotic Arm 
 
import math 
import matplotlib.pyplot as plt 
 
 
# Length of link 1 
L1 = 1
# Length of link 2 
L2 = 0.5 
 
# Defining angular range of link1 & link2
theta_start = 0			# starting angle = 0 degrees
theta_end = 90*(math.pi / 180)		# end angle = 90 degrees 
 
# Defining number of intervals between theta_start & theta_end 
n_theta = 10 
 
# Angle traced in Radians by link 1 with respect to Horizontal X axis(ranges from 0 to 90 degrees) 
theta1 = []
# Angle traced in Radians by link 2 with respect to Horizontal X axis(ranges from 0 to 90 degrees) 
theta2 = [] 
 
# for loop to calculate values of theta1 & theta2
for i in range(0,n_theta):		  
	angle_in_degrees = theta_start + i*(theta_end - theta_start) / (n_theta - 1)		#calculating value of interval in n_theta  
	theta1.append(angle_in_degrees)		# Incrementing theta1   
	theta2.append(angle_in_degrees)		# Incrementing theta2 
	
# Initial co-ordinates of link 1 at fixed pivot end in X & Y axis will be (0,0) 
x0 = 0 
y0 = 0 
 
ct = 1 
 
for THETA1 in theta1:		# for loop for any value THETA1 in range of theta1 values  
	for THETA2 in theta2:		# for loop for any value of THETA2 in range if theta2 values 

		# Calculating the coordinates of end point of link-1 
		x1 = L1 * math.cos(THETA1)		# X co-ordinate of amother end of link 1 which is starting co-ordinate of link2   
		y1 = L1 * math.sin(THETA1)		# Y co-ordinate of amother end of link 1 which is starting co-ordinate of link2 

		# Calculating the coordinates of end point of link-2
		# End coordinates for link-1 are the start coordinates for link-2 
		x2 = x1 + L2*math.cos(THETA2)		# Final X co-ordinate of link 2   
		y2 = y1 + L2*math.sin(THETA2)		# Final Y co-ordinate of link 2 
 
		filename = '%05.d.png' %ct 		# defining filename, so that plot created can be stitched sequentially to create animation   
		ct = ct + 1 		# interval between two images 
 
		plt.figure() # To plot each figure in new frame so as to get clear idea about movement of Arm for different combination of theta1 & theta2    
		plt.plot([x0,x1],[y0,y1])   
		plt.plot([x1,x2],[y1,y2])   
		plt.xlim([-0.5,1.8])	
		plt.ylim([0,1.8])
		plt.title("Simulation of Forward Kinematics of a 2R Robotic arm using Python")	
		plt.savefig(filename)