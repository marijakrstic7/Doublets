import numpy as np

ALMOST_SPHERICAL_ELLIPSOID_DIMENSIONS = np.array((1,0.9999,0.9999), dtype=float)

class _YLZUtils:
    """
    Functions to create YLZ structures and include them in structure_init.
    Taken from Adam Prada's input scripts for experiments with YLZ membranes.
    """

    @staticmethod
    def _unit_vector(vector):
        """ Returns the unit vector of the vector. """
        if max(abs(vector))>1e-12:
            return vector / np.linalg.norm(vector)
        else:
            return vector

    @staticmethod
    def _angle_between(v1, v2):
        """ Returns the angle between vectors 'v1' and 'v2' """
        v1_u = _YLZUtils._unit_vector(v1)
        v2_u = _YLZUtils._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def _get_vrot_ang(orient):
        """ Get rotation axis and angle for quaternion

        YLZ particle "dipoles" point along the x-axis of the ellipsoids. Ellipsoids 
        have their internal axes by default aligned with the box axes, i.e. the
        "dipoles" are pointing along the x directions. This function takes in the
        desired orientation of the "dipole" vector, finds the axis along which it has
        to rotate the quaternion and then the angle by which it has to rotate it in
        the right hand rule sense.
        """
        xhat = np.array([1,0,0])
        vrot = _YLZUtils._unit_vector(np.cross(xhat,orient))
        ang = _YLZUtils._angle_between(xhat,orient)

        # Edge case for orient [-1,0,0], where the rotation axis can be any in yz plane
        # In that case, the obtained vrot is [0,0,0] to within numerical precision,
        # which causes problems
        if np.linalg.norm(vrot)<0.9:
            vrot = np.array([0,1,0])
        
        return vrot,ang
    
    @staticmethod
    def get_zylinder_quaternion(input_orient):
        """ Get quaternion values for ellipsoid data file in LAMMPS

        As described in LAMMPS documentation (https://docs.lammps.org/read_data.html
        All quaternions are initialised to be aligned with the axes of the box.
    
        """

        # If nonzero orient
        orient = _YLZUtils._unit_vector(np.array(input_orient))
        # Orientation is expected in cartesians
        vrot, theta = _YLZUtils._get_vrot_ang(orient)
        # vrot in cartesians, theta in radians

        a = vrot[0]
        b = vrot[1]
        c = vrot[2]
        quatw = np.cos(theta/2)
        quati = a*np.sin(theta/2)
        quatj = b*np.sin(theta/2)
        quatk = c*np.sin(theta/2)
        return quatw, quati, quatj, quatk
    
    @staticmethod
    def get_ylz_centered_cylinder(
        n_total_approx, 
        n_circumference, 
        particle_diameter=1.0,
    ):
        """Atoms and orientations around a cylinder.
        
        Generates a number of rings n_rings containing n_circumference atoms each.
        """
        # TODO: this function should make cylinders in any orientation, not just along y
        # atom_list = []
        box_dimensions = []

        alpha = 2*np.pi/n_circumference
        cylinder_radius = (particle_diameter/2)/np.sin(alpha/2)

        # For simplicity keep Nl even
        n_rings = n_total_approx//n_circumference
        if n_rings==0:
            n_rings = 2
        elif n_rings%2==1:
            n_rings = n_rings+1

        n_total_exact = n_circumference*n_rings
        # ring spacing, diameter*cos(30) apart
        dx = particle_diameter*np.sqrt(3)/2
        box_length = n_rings*dx
        box_dimensions.append(4*cylinder_radius) # x
        box_dimensions.append(box_length) # y
        box_dimensions.append(4*cylinder_radius) # z

        xmin = -box_length/2
        coordinates = np.empty((n_total_exact,3), dtype=float)
        orientations = np.empty((n_total_exact,3), dtype=float)
        quaternions = np.empty((n_total_exact,4), dtype=float)

        # atom index, 1-based for lammps
        atom_count = 1
        # iterate over rings
        for i_x in range(n_rings):
            y = xmin+i_x*dx

            # every other rings is shifted by delta_angle/2
            # builds a hexagonal structure
            if i_x%2==0:
                offset = 0
            else:
                offset = alpha/2

            # iterate over individual atoms in rings
            for i_n in range(n_circumference):
                angle = offset + i_n*alpha
                mux = np.sin(angle)
                muy = 0
                muz = np.cos(angle)

                x = cylinder_radius*mux
                z = cylinder_radius*muz

                coordinates[atom_count-1] = [x,y,z]
                orientations[atom_count-1] = [mux,muy,muz]

                # get around obscure special case of sitting at 0?
                if max(abs(orientations[atom_count-1]))>1e-6:
                    quaternions[atom_count-1] = _YLZUtils.get_zylinder_quaternion(
                        orientations[atom_count-1]
                    )
                else:
                    quaternions[atom_count-1] = [1,0,0,0]

                # atom_list.append(atomobj([x,y,z],
                #     orient=[mux,muy,muz],
                #     diameter=almost_sphere,
                #     atom_ID=atom_count,
                #     human_ID=["membrane"]
                # ))            
                atom_count += 1

        return coordinates, orientations, quaternions, box_dimensions