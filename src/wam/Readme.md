# Attention

For evaluation purposes it is really important that we keep the order of the first two sensors
inside any ``.xml``-file we want to evaluate the same:
1) the position of the pendulum's tip in the global frame
2) the rotation of the pendulum's tip in the global frame expresses as a quaternion
--> Thus, we can always read the first 7 values of the mj_sensordata in our results class to obtain the position and the rotation of the 
trajectory

```xml
<framepos name="pend_glob_end" objtype="site" objname="wam/sensor_sites/pend_endeff" reftype="site" refname="wam/ref_sites/global_origin"/>
<framequat name="pend_glob_end_quat" objtype="site" objname="wam/sensor_sites/pend_endeff" reftype="site" refname="wam/ref_sites/global_origin"/>
<framepos name="pend_begin_end" objtype="site" objname="wam/sensor_sites/pend_begin" reftype="site" refname="wam/ref_sites/global_origin"/>
```


# Configurations
The configs are given by
```python
q_config = np.array([0, -0.3, 0, 0.3, 0, 0])
q_config = np.array([0, -1.7, 0, 1.7, 0, 0])
q_config = np.array([0, -0.78, 0, 2.37, 0, 0])
q_config = np.array([0, -1.6, 1.55, 1.6, 0, 1.55])
```