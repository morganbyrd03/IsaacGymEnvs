from odio_urdf import *

rope = Robot("rope")

num_links = 5
radius = 0.2
length = 1.0

links = []
for i in range(num_links):
    name = "link" + str(i)
    origin = "{} {} {}".format(0., 0., radius*2)
    # origin = "{} {} {}".format(0. * i, 0. * i, 0.5 * i)
    inertial = Inertial(Mass(value=0.1), Origin(xyz=origin))
    visual = Visual(Geometry(Cylinder(length=length, radius=radius)), Origin(xyz=origin))
    # visual = Visual(Geometry(Sphere(radius=radius)), Origin(xyz=origin))
    # collision = Collision(Geometry(Sphere(radius=radius)), Origin(xyz=origin))
    collision = Collision(Geometry(Cylinder(length=length, radius=radius)), Origin(xyz=origin))
    links.append(Link(name, inertial, visual, collision))
    links.append(Link(name + "_x"))  # Needed for multiple joint axes

joints = []
for i in range(num_links-1):
    # origin = "{} {} {}".format(0., 0., radius*2)
    origin = "{} {} {}".format(0., 0., length)
    # joints.append(Joint("joint"+str(i+1), Parent("link"+str(i)), Child("link"+str(i+1)), Dynamics(damping=0.),
    #                     Limit(lower=-2.5, upper=2.5), Axis(xyz=("0 1 0")), Origin(xyz=origin), type="revolute",))
    joints.append(Joint("joint" + str(i+1) + "_y", Parent("link" + str(i)), Child("link" + str(i) + "_x"), Dynamics(damping=0.),
                        Limit(lower=-2.5, upper=2.5), Axis(xyz=("0 1 0")), Origin(xyz=origin), type="revolute", ))
    joints.append(Joint("joint" + str(i+1) + "_x", Parent("link" + str(i) + "_x"), Child("link" + str(i + 1)), Dynamics(damping=0.),
                        Limit(lower=-2.5, upper=2.5), Axis(xyz=("1 0 0")), Origin(xyz="0, 0, 0"), type="revolute", ))

joints.append(Joint("joint0", Parent("panda_hand"), Child("link0"), Dynamics(damping=0.),
                        Limit(lower=-2.5, upper=2.5), Axis(xyz=("0 1 0")), Origin(xyz=origin), type="revolute",))

rope(*links)
rope(*joints)

# Save to file
with open("/home/morgan/rope.urdf", "w") as f:
    f.write(str(rope))