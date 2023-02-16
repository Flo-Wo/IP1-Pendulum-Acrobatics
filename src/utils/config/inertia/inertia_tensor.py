import numpy as np

# see here: https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=capsule#body-joint
# for notes on the parameter choices

# the 2 parameters include: radius of the cylinder and half-height/fromto


def capsule(height=0.3, radius=0.015):
    # see here https://www.gamedev.net/articles/programming/math-and-physics/capsule-inertia-tensor-r3856/
    # for the computation of the tensor
    m_cy = height * radius**2 * np.pi
    m_hs = 2 * radius**3 * np.pi / 3
    m = m_cy + 2 * m_hs
    print("mass?", m)

    # return the diag tensor
    return np.array(
        [
            m_cy * (height**2 / 12 + radius**2 / 4)
            + 2
            * m_hs
            * (2 * radius**2 / 5 + height**2 / 2 + 3 * height * radius / 8),
            m_cy * (radius**2 / 2) + 2 * m_hs * (2 * radius**2 / 5),
            m_cy * (height**2 / 12 + radius**2 / 4)
            + 2
            * m_hs
            * (2 * radius**2 / 5 + height**2 / 2 + 3 * height * radius / 8),
        ]
    )


def cylinder(mass=0.1, height=0.1, radius=0.1):
    # see: https://scienceworld.wolfram.com/physics/MomentofInertiaCylinder.html
    m_r2 = mass * radius**2
    m_h2 = mass * height**2
    return np.array(
        [1 / 12 * m_h2 + 1 / 4 * m_r2, 1 / 12 * m_h2 + 1 / 4 * m_r2, 1 / 2 * m_r2]
    )


np.set_printoptions(suppress=True)

print("base plate")
cylinder_diag = cylinder(0.01, 0.005, 0.02)
print(cylinder_diag)

print("pendulum mount polt")
print(cylinder(0.05, 0.05, 0.01))

print("pendulum cylinder approx")
print(cylinder(0.5, 0.3, 0.15))

print("pendulum capsule")
print(capsule())
