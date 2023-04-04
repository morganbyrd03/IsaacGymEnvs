from odio_urdf import *

rope = Robot("rope")

num_links = 25
radius = 0.01

links = []
for i in range(num_links):
    name = "link" + str(i)
    # print((Capsule().allowed_attributes))
    origin = "{} {} {}".format(0., 0., radius*2)
    # cap_origin = "{} {} {}".format(0., 0., 1.0+radius)
    # origin = "{} {} {}".format(0. * i, 0. * i, 0.5 * i)
    inertial = Inertial(Mass(value=0.1), Origin(xyz=origin))
    # visual = Visual(Geometry(Cylinder(length=1, radius=radius)), Origin(xyz=origin))
    visual = Visual(Geometry(Sphere(radius=radius)), Origin(xyz=origin))
    collision = Collision(Geometry(Sphere(radius=radius)), Origin(xyz=origin))
    # collision = Collision(Geometry(Cylinder(length=1, radius=radius)), Origin(xyz=origin))
    # visual = Visual(Geometry(Box(size="1 0.1 0.1")), Origin(xyz=origin))
    # cap_visual = Visual(Geometry(Sphere(radius=radius*2)), Origin(xyz=cap_origin))
    # cap_collision = Collision(Geometry(Sphere(radius=radius*2)), Origin(xyz=cap_origin))
    # links.append(Link(name+"_cap", inertial, cap_visual, cap_collision))
    links.append(Link(name, inertial, visual, collision))

joints = []
for i in range(num_links-1):
    origin = "{} {} {}".format(0., 0., radius*2)
    # origin = "{} {} {}".format(0., 0., 1.0)
    joints.append(Joint("joint"+str(i+1), Parent("link"+str(i)), Child("link"+str(i+1)), Dynamics(damping=0.),
                        Limit(lower=-2.5, upper=2.5), Axis(xyz=("0 1 0")), Origin(xyz=origin), type="revolute",))

joints.append(Joint("joint0", Parent("panda_hand"), Child("link0"), Dynamics(damping=0.),
                        Limit(lower=-2.5, upper=2.5), Axis(xyz=("0 1 0")), Origin(xyz=origin), type="revolute",))

rope(*links)
rope(*joints)

# Save to file
with open("/home/morgan/rope.urdf", "w") as f:
    f.write(str(rope))