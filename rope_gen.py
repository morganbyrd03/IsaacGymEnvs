from odio_urdf import *

rope = Robot("rope")

num_links = 48
radius = 0.02
length = 1.0 / (num_links)
mass = 0.3 / (num_links)
# length_true = 1.0 / (num_links//2)

links = []
joints = []
# An attribute specifying the lower joint limit (in radians for revolute joints, in metres for prismatic joints). Omit if joint is continuous.
joint_limit_lower  = -2.5
joint_limit_upper  = 2.5
joint_limit_effort = 20 # (required) attribute for enforcing the maximum joint effort
joint_limit_velocity = 20 # (required)m/s 

joint_limit =Limit( lower=joint_limit_lower, 
                    upper=joint_limit_upper,
                    effort=joint_limit_effort,
                    velocity=joint_limit_velocity)

for i in range(num_links):
    # Add link
    name = "link" + str(i)
    origin = "{} {} {}".format(0., 0., length/2)
    inertial = Inertial(Mass(value=mass), Origin(xyz=origin))
    visual = Visual(Geometry(Cylinder(length=length, radius=radius)), Origin(xyz=origin))
    collision = Collision(Geometry(Cylinder(length=length, radius=radius)), Origin(xyz=origin))
    links.append(Link(name, inertial, visual, collision))

    # add joint
    if (i==0):
        origin = "{} {} {}".format(0., 0., 0)
        joints.append(Joint("joint0", Parent("panda_hand"), Child("link0"), Dynamics(damping=0.),
                        joint_limit, Axis(xyz=("0 1 0")), Origin(xyz=origin), type="revolute",))
    else:
        if i % 2 == 0:
            axis = "0 1 0"
        else:
            axis = "1 0 0"

        origin = "{} {} {}".format(0., 0., length)
        joints.append(Joint("joint"+str(i), 
                        Parent("link"+str(i-1)), 
                        Child("link"+str(i)), 
                        Dynamics(damping=0.),
                        joint_limit,
                        Axis(xyz=axis), Origin(xyz=origin), 
                        type="revolute")
                        )

rope(*links)
rope(*joints)

# Save to file
with open("./rope.urdf", "w") as f:
    f.write(str(rope))