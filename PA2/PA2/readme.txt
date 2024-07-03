Global Variables

control_point : save control point of cows
final_point : save control_point[0], so if animation terminate, cow2wld is final_point
currnt : how many control points are saved
aniinitial : check animation initialized, it is changed every rotate
aniStarttime : animation start time. it is changed every rotate
rotate_count : check rotate count

Functions

def display():

when current is less than 6, it show cows on control point 
so, use drawCow for every control_point and setting cursor false

when current is 6, it need to start animation.
so initializing is needed, 
cow2wld and final_point are first control_point
aniStratTime is current time, and aniinitial is set to 0 and return

and start next part (already initialized and rotate_count is less than 3)
animTime is current time - aniStarttime
rotation time is 6.0 , because control_point are 6. It is easy to implement

when animTime < rotation time
drawCow, 
i is index So, flooring the animTime
and t is need to 0<= t <= 1 in spline, so animTime % 1.0

p is location, it use only translation part, but rotation part is changed later, 
so just use 4*4 matrix for spline function arguments

r is rotation matrix, get rotation matrix by spline_deperative

after get values, rotate and translate cow2wld
and draw it

if animTime > rotation time, 
set rotate_count increase, and aniinitial to 0
so, start at initialized first

if 3 rotate is terminated,
then initialize control_point and aniinitial, rotate_count, cow2wld

def spline(t, p_1, p0, p1, p2) :
spline formula, t is 0<=t <= 1
and use 4 points

def spline_deperative(t, p_1, p0, p1, p2)
Differentiate spline function and get rotate matrix
by only translation part (rotation part is not used it need to change)
and get pitch and yaw

in this program, y-axis is like z-axis, 
so I use rotate Y at yaw
and rotate Z at pitch
and rotate matrix is rotateY @ rotateZ

def onMouseButton :
when clicked it need to save control points
so add save control point code if currnet < 6

def onMouseDrag :
if currnet >= 6 , it is animating so it need to be not used
and at V_drag. It is need to make a plane perpendicular to ray direction
so, I make a plane with normalized ray direction(plane's normal vector)





